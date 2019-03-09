import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from collections import deque
from imageio import imread, imwrite
from math import floor
from time import time


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


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side
    This could use PIL.Image.paste, but it complains about tuples.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img


def visualize_features(im1, im2):
    im3 = concat_images(im1, im2)

    fig=plt.figure(figsize=(18, 16), dpi=72)
    plt.imshow(im3)

    ax = plt.gca()

    for m in better_matches[:50]:
        kp_a = kp_1[m.queryIdx].pt
        kp_b = kp_2[m.trainIdx].pt
        l = mlines.Line2D([kp_a[0], kp_b[0]+im1.shape[0]],
                          [kp_a[1], kp_b[1]], color='r', marker='.', markevery=(1,1))
        ax.add_line(l)

    plt.show()


def paste(canvas, paint):
    '''
    Paste an image on to another using masks.
    '''
    ch, cw = canvas.shape
    ph, pw = paint.shape
    nw, nh = (0,0)
    if cw < pw:
        nw = pw - cw
    if ch < ph:
        nh = ph - ch
    output = np.copy(canvas)
    output = np.pad(output, ((0, nh), (0, nw)), 'constant', constant_values=(0,0))

    # mask magic! This is very fast considering the size of what's being merged
    o_mask = np.ma.equal(output, 0)
    output[o_mask] = paint[o_mask] # for places where output is 0: paint.
    return output


def crop_zeros(im):
    '''
    Crop zeros around an image
    '''
    r,c = im.shape
    top = 0
    left = 0
    bottom = r-1
    right = c-1
    while sum(im[top,:]) == 0:
        top += 1

    while sum(im[bottom,:]) == 0:
        bottom -=1

    while sum(im[:,left]) == 0:
        left +=1

    while sum(im[:,right]) == 0:
        right -=1

    return im[top:bottom+1,left:right+1]

