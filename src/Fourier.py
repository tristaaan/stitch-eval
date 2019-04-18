import cv2
import math

import numpy as np
import scipy as sp

from collections import deque
from imageio import imread, imwrite
from imutils import rotate_bound as rotate
from math import pi,sqrt
from numpy.fft import fft2, ifft2
from operator import mul
from scipy import conj
from scipy.signal import convolve2d
from time import time

from util import bounds_equalize, crop_zeros, eq_paste, \
                 pad, square_off, tuple_sub


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
    "here the spectrum of the two signals is whitened by dividing each
    per frequency product by the magnitudes of the Fourier transforms"
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], 'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / (abs(f1) * abs(f2))


def log_polar(img):
    '''
    Convert image into log-polar space
    '''
    rad_size = 2.0
    center = (img.shape[0] / 2, img.shape[1] / 2)
    radius = round(sqrt(center[0] ** rad_size + center[1] ** rad_size))
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
    if math.sqrt(dnom) == 0:
        return 0
    return nom / math.sqrt(dnom)

def test_corners(ref, mov, pt):
    '''
    Find the corner which matches to the found x,y point
    '''
    window = 24; # window size
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
    best_corner = np.argmax(corner_scores)
    corner = corners[best_corner]
    nx, ny = xys[best_corner]
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

def apply_hamming_window(im):
    h = np.hamming(im.shape[0])
    ham2d = np.sqrt(np.outer(h,h))
    return im * ham2d

def is_square(im):
    return im.shape[0] == im.shape[1]

def F_stitch(im1, im2):
    # start timer
    start = time()
    zero = 500

    im1_orig = im1.copy()
    im2_orig = im2.copy()

    if not is_square(im1):
        im1 = square_off(im1)
    if not is_square(im2):
        im2 = square_off(im2)

    if im1.shape != im2.shape:
        im1, im2 = bounds_equalize(im1, im2)

    # convert to log-polar coordinates
    im1_p = log_polar(im1)
    im2_p = log_polar(im2)

    # find correlation, enhance fft
    impulse = ifft2(phase_correlation(im1_p, im2_p))
    theta, _ = np.unravel_index(np.argmax(impulse), impulse.shape)

    # calculate angle in degrees
    angle = (theta * 360 / impulse.shape[0]) % 180
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    # rotate the moving image to the correct angle
    if angle != 0:
        im2 = rotate(im2_orig, angle)
        im2_orig = crop_zeros(im2.copy(), zero=zero)

    im1 = crop_zeros(im1, zero=zero)
    if not is_square(im1):
        im1 = square_off(im1)
    im2 = crop_zeros(im2, zero=zero)
    if not is_square(im2):
        im2 = square_off(im2)

    # find the x,y translation
    im1, im2 = bounds_equalize(im1, im2)
    print(im1.shape, im2.shape)
    impulse = ifft2(phase_correlation(im1, im2))
    y,x = np.unravel_index(np.argmax(impulse), impulse.shape)

    if y > im1.shape[0] // 2:
        y -= im1.shape[0]
    if x > im1.shape[1] // 2:
        x -= im1.shape[1]

    print(x,y, angle)
    # prepare the image to be pasted
    im1 = crop_zeros(im1, zero=zero)
    # im2 = pad(im2_orig, x, y)
    im2 = pad(im2_orig, x, y)
    # paste
    base = eq_paste(im1, im2)
    base = crop_zeros(base, zero=zero)
    return (base, [x,y,angle], time() - start)

def Fourier(blocks):
    A,B,C,D = blocks

    AB, M1, t1 = F_stitch(A, B)
    CD, M2, t2 = F_stitch(C, D)

    E,  M3, t3 = F_stitch(AB, CD) # argument order matters here

    E = crop_zeros(E, zero=100)
    return (E, [M1, M2, M3], sum([t1,t2,t3]))

def Frequency(blocks):
    return Fourier(blocks)
