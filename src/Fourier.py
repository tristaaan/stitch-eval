import cv2

import numpy as np
import scipy as sp

from collections import deque
from imageio import imread, imwrite
from math import sqrt
from numpy.fft import fft2, ifft2
from operator import mul
from scipy import conj, ndimage
from time import time

from util import bounds_equalize, crop_zeros, eq_paste, tuple_sub


def cross_power_spectrum(i1, i2):
    '''
    normalized cross-power spectrum
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], 'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / abs(f2 * conj(f2))

def log_polar(img, radius=2.0):
    '''
    Convert image into log-polar space
    '''
    center = (img.shape[0] / 2, img.shape[1] / 2)
    radius = round(sqrt(center[0] ** radius + center[1] ** radius))
    return cv2.warpPolar(img, img.shape, center, radius, cv2.WARP_POLAR_LOG)

def NCC(i1, i2):
    '''
    Normalized cross correlation, maximize this
    '''
    assert sum(i1.shape) == sum(i2.shape), 'i1 and i2 are different shapes "%s" v "%s"' % (i1.shape, i2.shape,)
    f_err = i1 - i1.mean()
    t_err = i2 - i2.mean()
    nom = (f_err * t_err).sum()
    dnom = np.square(t_err).sum() * np.square(f_err).sum()
    if sqrt(dnom) == 0:
        return 0
    return nom / sqrt(dnom)

def test_corners(ref, mov, pt):
    '''
    Find the corner which matches to the found x,y point
    '''
    window = 16; # window size
    mw,mh = mov.shape
    rw,rh = ref.shape
    y,x = pt
    corrs = [] # tl, tr, br, bl

    # moving image corners
    tl = mov[0:window, 0:window]
    tr = mov[0:window, mw-window:mw]
    br = mov[mh-window:mh, mw-window:mw]
    bl = mov[mh-window:mh, 0:window]

    # reference corners around pt
    ref_tl = ref[y:min(y+window, rh), x:min(x+window, rw)]
    ref_tr = ref[y:min(y+window, rh), max(x-window,0):x]
    ref_br = ref[max(y-window,0):y,   max(x-window,0):x]
    ref_bl = ref[max(y-window,0):y,   x:min(x+window, rw)]

    # compare corners
    if mul(*tl.shape) > 0 and mul(*tl.shape) == mul(*ref_tl.shape):
        corrs.append(NCC(ref_tl, tl))
    else:
        corrs.append(-1)

    if mul(*tr.shape) > 0 and mul(*tr.shape) == mul(*ref_tr.shape):
        corrs.append(NCC(ref_tr, tr))
    else:
        corrs.append(-1)

    if mul(*br.shape) > 0 and mul(*br.shape) == mul(*ref_br.shape):
        corrs.append(NCC(ref_br, br))
    else:
        corrs.append(-1)

    if mul(*bl.shape) > 0 and mul(*bl.shape) == mul(*ref_bl.shape):
        corrs.append(NCC(ref_bl, bl))
    else:
        corrs.append(-1)

    # return the best one
    return np.argmax(corrs)


def prepare_paste(i1,i2, x,y):
    corner = test_corners(i1, i2, (y,x)) # tl, tr, bl, br
    padded = None
    if corner == 0: # top left corner
        row_p = y
        col_p = x
        padded = np.pad(i2, ((row_p,0), (col_p,0)), 'constant', constant_values=(0,0))
    elif corner == 1: # top right corner
        row_p = y
        col_p = max(i2.shape[1] - x, 0)
        padded = np.pad(i2, ((row_p,0), (0,col_p)), 'constant', constant_values=(0,0))
    elif corner == 2: # bottom right corner
        row_p = max(i2.shape[0] - y, 0)
        col_p = max(i2.shape[1] - x, 0)
        padded = np.pad(i2, ((0,row_p), (0,col_p)), 'constant', constant_values=(0,0))
    elif corner == 3: # bottom left corner
        row_p = max(i2.shape[0] - y, 0)
        col_p = x
        padded = np.pad(i2, ((0,row_p), (col_p,0)), 'constant', constant_values=(0,0))
    else:
        raise ValueError('This should never happen, corner=%d' % corner)
    return padded


def Fourier(blocks):
    queue = deque(blocks)
    base = queue.popleft()
    ind = 0
    total_time = 0
    while len(queue) > 0:
        # load images
        im1 = base
        im2 = queue.popleft()

        if sum(im2.shape) > sum(im1.shape):
            # this will run on the first block if im2 is rotated
            im1 = bounds_equalize(im2, im1)
        else:
            im2 = bounds_equalize(im1, im2)

        # start timer
        start = time()

        # convert to log-polar coordinates
        im1_p = log_polar(im1)
        im2_p = log_polar(im2)

        # find correlation
        impulse = ifft2(cross_power_spectrum(im1_p, im2_p))
        theta, _ = np.unravel_index(np.argmax(impulse), impulse.shape)

        # calculate angle in degrees
        angle = theta * (360 / impulse.shape[0]) % 180
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180

        # rotate the moving image to the correct angle
        if angle != 0:
            im2 = ndimage.rotate(im2, -angle, reshape=False)

        im2 = crop_zeros(im2)
        # find the x,y translation
        im2_orig = im2.copy()
        im2 = bounds_equalize(im1, im2)
        impulse = ifft2(cross_power_spectrum(im1, im2))
        y,x = np.unravel_index(np.argmax(impulse), impulse.shape)

        # due to the padding we need to shift the x and y
        ny = y+int((im2.shape[0] - im2_orig.shape[0])/2)
        nx = x+int((im2.shape[1] - im2_orig.shape[1])/2)

        # The corner to which this method matches can be ambiguous
        # find the matching corner and prepare the image to be pasted
        im2 = prepare_paste(im1, im2_orig, nx, ny)
        base = eq_paste(im1, im2)
        total_time += time() - start
        print('stitched %d with %d' % (ind, ind + 1))
        ind += 1

    base = crop_zeros(base)
    imwrite('../data/stitched.tif', base)
    average_time = total_time / (len(blocks) - 1)
    return (base, average_time)
