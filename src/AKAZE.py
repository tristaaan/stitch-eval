import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import deque
from imageio import imread, imwrite
from math import floor
from time import time

from util import paste, crop_zeros

def get_akaze_keypoints(im):
    akaze = cv2.AKAZE_create()
    return akaze.detectAndCompute(im, None)

def AKAZE(blocks):
    bf = cv2.BFMatcher()
    queue = deque(blocks)
    base = queue.popleft()
    ind = 0
    total_time = 0
    while len(queue) > 0:
        # load images
        im1 = base
        im2 = queue.popleft()

        # start timer
        start = time()

        # Find feature points
        (kp_1, desc_1) = get_akaze_keypoints(im1)
        (kp_2, desc_2) = get_akaze_keypoints(im2)

        # Match descriptors.
        matches = bf.knnMatch(desc_1, desc_2, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance)
        better_matches = [ m[0] for m in matches if m[0].distance < 0.75 * m[1].distance]

        # make sure we have enough matches.
        if len(better_matches) < 10:
            # cache feature points
            print('could not find enough matches')
            continue

        # Warp the second image to best match the first.
        src_pts = np.float32([ kp_2[m.trainIdx].pt for m in better_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_1[m.queryIdx].pt for m in better_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = im1.shape

        # use the homography offsets to determine the size for the stitched image
        x_offset, y_offset = M[0:2, 2]
        new_size = (int(max(w, w+x_offset)), int(max(h, h+y_offset)))

        # warp!
        warped = cv2.warpAffine(im2, M[:2,:3], new_size, flags=cv2.INTER_CUBIC)
        print('stitched %d with %d' % (ind, ind + 1))
        ind += 1
        base = paste(im1, warped)
        total_time += time() - start

    base = crop_zeros(base)
    imwrite('../data/stitched.tif', base)
    average_time = total_time / (len(blocks) - 1)
    return (base, average_time)
