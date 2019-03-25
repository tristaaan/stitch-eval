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


def norm_cross_power_spectrum(i1, i2):
    '''
    normalized cross power spectrum, Faroosh et. al.
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], 'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / abs(f2 * conj(f2))

def cross_power_spectrum(i1, i2):
    '''
    Cross power spectrum,
    Image Mosaic Based on Phase Correlation and Harris Operator, F. Yang et al.
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], 'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / abs(f1 * conj(f2))

def phase_correlation(i1, i2):
    '''
    Image Alignment and Stitching: A Tutorial, R. Szeliski
    "here the spectrum of the two signals is **whitened** by dividing each per frequency product
    by the magnitudes of the Fourier transforms"
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], 'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / (abs(f1) * abs(f2))


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
    assert i1.shape == i2.shape, \
           'i1 and i2 are different shapes "%s" v "%s"' % (i1.shape, i2.shape,)
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
    if mul(*tl.shape) > 0 and tl.shape == ref_tl.shape:
        corrs.append(NCC(ref_tl, tl))
    else:
        corrs.append(-1)

    if mul(*tr.shape) > 0 and tr.shape == ref_tr.shape:
        corrs.append(NCC(ref_tr, tr))
    else:
        corrs.append(-1)

    if mul(*br.shape) > 0 and br.shape == ref_br.shape:
        corrs.append(NCC(ref_br, br))
    else:
        corrs.append(-1)

    if mul(*bl.shape) > 0 and bl.shape == ref_bl.shape:
        corrs.append(NCC(ref_bl, bl))
    else:
        corrs.append(-1)

    # return the best one
    return np.argmax(corrs), max(corrs)


def find_best_corner(i1, i2, old_shape, x,y):
    '''
    offset the x and y depending on which corner matches best
    return new x, new y, corner
    '''
    corner_scores = []
    corners = []
    xys = []

    # tl
    ny = y+int((old_shape[0] - i2.shape[0])/2)
    nx = x+int((old_shape[1] - i2.shape[1])/2)
    c,cs = test_corners(i1, i2, (ny,nx))
    corner_scores.append(cs)
    corners.append(c)
    xys.append((nx,ny))

    # tr
    ny = y-int((old_shape[0] - i2.shape[0])/2)
    nx = x+int((old_shape[1] - i2.shape[1])/2)
    c,cs = test_corners(i1, i2, (ny,nx))
    corner_scores.append(cs)
    corners.append(c)
    xys.append((nx,ny))

    # br
    ny = y-int((old_shape[0] - i2.shape[0])/2)
    nx = x-int((old_shape[1] - i2.shape[1])/2)
    c,cs = test_corners(i1, i2, (ny,nx))
    corner_scores.append(cs)
    corners.append(c)
    xys.append((nx,ny))

    # bl
    ny = y+int((old_shape[0] - i2.shape[0])/2)
    nx = x-int((old_shape[1] - i2.shape[1])/2)
    c,cs = test_corners(i1, i2, (ny,nx))
    corner_scores.append(cs)
    corners.append(c)
    xys.append((nx,ny))

    # tl, tr, bl, br
    corner = corners[np.argmax(corner_scores)]
    nx, ny = xys[np.argmax(corner_scores)]
    return corner, nx, ny


def prepare_paste(i1, i2, old_shape, x,y):
    '''
    Pad the image depending on which corner matches best with the image
    The corner to which this method matches can be ambiguous
    '''
    corner, x, y = find_best_corner(i1, i2, old_shape, x, y)
    padded = None
    if corner == 0: # top left corner
        row_p = abs(y)
        col_p = abs(x)
        padded = np.pad(i2, ((row_p,0), (col_p,0)), 'constant', constant_values=(0,0))
    elif corner == 1: # top right corner
        row_p = abs(y)
        col_p = max(i2.shape[1] - x, 0)
        padded = np.pad(i2, ((row_p,0), (0,col_p)), 'constant', constant_values=(0,0))
    elif corner == 2: # bottom right corner
        row_p = max(i2.shape[0] - y, 0)
        col_p = max(i2.shape[1] - x, 0)
        padded = np.pad(i2, ((0,row_p), (0,col_p)), 'constant', constant_values=(0,0))
    elif corner == 3: # bottom left corner
        row_p = max(i2.shape[0] - y, 0)
        col_p = abs(x)
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
        im2 = queue.popleft()
        base, im2 = bounds_equalize(base, im2)

        # start timer
        start = time()

        # convert to log-polar coordinates
        base_p = log_polar(base)
        im2_p = log_polar(im2)

        # find correlation
        impulse = ifft2(phase_correlation(base_p, im2_p))
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

        im2 = crop_zeros(im2, zero=500)
        base = crop_zeros(base, zero=500)
        # find the x,y translation
        im2_orig = im2.copy()
        base, im2 = bounds_equalize(base, im2)
        impulse = ifft2(phase_correlation(base, im2))
        y,x = np.unravel_index(np.argmax(impulse), impulse.shape)

        # prepare the image to be pasted
        im2 = prepare_paste(base, im2_orig, im2.shape, x, y)
        # paste
        base = eq_paste(base, im2)
        total_time += time() - start
        print('stitched %d with %d' % (ind, ind + 1))
        ind += 1

    base = crop_zeros(base, zero=100)
    imwrite('../data/stitched.tif', base)
    average_time = total_time / (len(blocks) - 1)
    return (base, average_time)
