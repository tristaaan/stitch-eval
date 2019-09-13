import cv2
import math
import gc

import numpy as np
import scipy as sp

from imutils import rotate_bound
from math import pi,sqrt
from numpy.fft import fft2, ifft2
from operator import mul
from scipy import conj
from scipy.signal import convolve2d
from time import time

from util import bounds_equalize, ul_equalize, crop_zeros, merge, \
                 square_off, tuple_sub


def norm_cross_power_spectrum(i1, i2):
    '''
    normalized cross power spectrum, Faroosh et. al.
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], \
        'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / abs(f2 * conj(f2))


def cross_power_spectrum(i1, i2):
    '''
    Cross power spectrum,
    Image Mosaic Based on Phase Correlation and Harris Operator, F. Yang et al.
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], \
        'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
    f1 = fft2(i1)
    f2 = fft2(i2)
    return (f1 * conj(f2)) / abs(f1 * conj(f2))


def phase_correlation(i1, i2):
    '''
    Image Alignment and Stitching: A Tutorial, R. Szeliski
    "here the spectrum of the two signals is whitened by dividing each
    per frequency product by the magnitudes of the Fourier transforms"
    '''
    assert i1.shape[0] == i2.shape[0] and i1.shape[1] == i2.shape[1], \
        'images are different sizes: %s v %s' % (i1.shape, i2.shape,)
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

    # save originals for rotation later
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

    # cropped values are actually x,y translations
    tx,ty = (0,0)
    # rotate the moving image to the correct angle
    if angle != 0:
        pre_size = im2_orig.shape
        rotated = rotate_bound(im2_orig, angle)
        post_size = rotated.shape
        im2,(cy,_,cx,_) = crop_zeros(rotated, crop_vals=True)
        rot_diff = tuple_sub(post_size, pre_size)
        tx -= cx - rot_diff[1]//2
        ty -= cy - rot_diff[0]//2
        im2_orig = im2.copy()

    im1 = im1_orig.copy()
    if not is_square(im1):
        im1 = square_off(im1)
    if not is_square(im2):
        im2 = square_off(im2)

    # find the x,y translation
    im1, im2 = ul_equalize(im1, im2)

    impulse = ifft2(phase_correlation(im1, im2))
    y,x = np.unravel_index(np.argmax(impulse), impulse.shape)

    if y > im1.shape[0] // 2:
        y -= im1.shape[0]
    if x > im1.shape[1] // 2:
        x -= im1.shape[1]

    # t_mod has already been applied for merging
    # print('x:%d+%d, y:%d+%d, theta:%.02f' % (x, tx, y, ty, angle))
    h_offset, w_offset = im1_orig.shape
    base = merge(im1_orig, im2_orig, x, y, zero=zero)

    return (base, [x+tx, y+ty, angle], time() - start)

def Fourier(blocks):
    A,B,C,D = blocks

    AB, M1, t1 = F_stitch(A, B)
    CD, M2, t2 = F_stitch(C, D)

    E,  M3, t3 = F_stitch(AB, CD) # argument order matters here

    E = crop_zeros(E, zero=100)
    return (E, [M1, M2, M3], sum([t1,t2,t3]))

def Frequency(blocks):
    return Fourier(blocks)
