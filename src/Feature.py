import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import deque
from imageio import imread, imwrite
from math import floor
from time import time

from util import eq_paste, crop_zeros


def uint16_to_uint8(im):
    im = (im / (2**16-1) * 255).astype('uint8')
    return im


def get_akaze_keypoints(im):
    akaze = cv2.AKAZE_create()
    return akaze.detectAndCompute(im, None)


def get_sift_keypoints(im):
    im = uint16_to_uint8(im)
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(im, None)


def get_surf_keypoints(im):
    im = uint16_to_uint8(im)
    surf = cv2.xfeatures2d.SURF_create()
    return surf.detectAndCompute(im, None)


def stitch(im1, im2, matcher, get_keypoints):
    # start timer
    start = time()

    # Find feature points
    (kp_1, desc_1) = get_keypoints(im1)
    (kp_2, desc_2) = get_keypoints(im2)

    # Match descriptors.
    matches = matcher.knnMatch(desc_1, desc_2, k=2)
    matches = sorted(matches, key = lambda x:x[0].distance)
    better_matches = [ m[0] for m in matches if m[0].distance < 0.75 * m[1].distance]

    # make sure we have enough matches.
    if len(better_matches) < 10:
        # cache feature points
        print('could not find enough matches')
        return None, None, None

    # Warp the second image to best match the first.
    src_pts = np.float32([ kp_2[m.trainIdx].pt for m in better_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_1[m.queryIdx].pt for m in better_matches ]).reshape(-1,1,2)

    # find the transformation given the amount of points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = im2.shape

    # use the homography offsets to determine the size for the stitched image
    x_offset, y_offset = M[0:2, 2]
    new_size = (int(w+x_offset), int(h+y_offset))
    affine_M = M[:2,:3]

    # warp and paste
    warped = cv2.warpAffine(im2, affine_M, new_size, flags=cv2.INTER_CUBIC)
    base = eq_paste(im1, warped)
    base = crop_zeros(base)
    return (base, affine_M, time() - start)


def stitch_blocks(blocks, method):
    bf_matcher = cv2.BFMatcher()

    A,B,C,D = blocks

    AB, M1, t1 = stitch(A, B, bf_matcher, method)
    CD, M2, t2 = stitch(C, D, bf_matcher, method)
    if t1 == None or t2 == None:
        return (None, None, None)
    E, M3, t3 = stitch(AB, CD, bf_matcher, method)

    base = crop_zeros(E)
    return (base, [M1, M2, M3], sum([t1,t2,t3]))


def AKAZE(blocks):
    return stitch_blocks(blocks, get_akaze_keypoints)


def SIFT(blocks):
    return stitch_blocks(blocks, get_sift_keypoints)


def SURF(blocks):
    return stitch_blocks(blocks, get_surf_keypoints)
