import cv2

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

from util import bounds_equalize, crop_zeros, eq_paste, pad, tuple_sub


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


def apply_hamming_window(im):
    h = np.hamming(im.shape[0])
    ham2d = np.sqrt(np.outer(h,h))
    return im * ham2d


def F_stitch(im1, im2):
    # start timer
    start = time()

    if im1.shape != im2.shape:
        im1, im2 = bounds_equalize(im1, im2)

    # convert to log-polar coordinates
    im1_p = log_polar(im1)
    im2_p = log_polar(im2)

    # find correlation, enhance fft
    impulse = ifft2(phase_correlation(im1_p, im2_p))
    theta, _ = np.unravel_index(np.argmax(impulse), impulse.shape)

    # calculate angle in degrees
    angle = theta * (360 / impulse.shape[0]) % 180
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    # rotate the moving image to the correct angle
    if angle != 0:
        im2 = rotate(im2, -angle)

    im2 = crop_zeros(im2, zero=500)
    im1 = crop_zeros(im1, zero=500)
    # find the x,y translation
    im2_orig = im2.copy()
    im1, im2 = bounds_equalize(im1, im2)
    impulse = ifft2(phase_correlation(im1, im2))
    y,x = np.unravel_index(np.argmax(impulse), impulse.shape)

    # prepare the image to be pasted
    im2 = pad(im2, x,y)
    # paste
    base = eq_paste(im1, im2)
    base = crop_zeros(base, zero=100)
    return (base, time() - start)

def Fourier(blocks):
    A,B,C,D = blocks

    AB, t1 = F_stitch(A, B)
    CD, t2 = F_stitch(C, D)
    E,  t3 = F_stitch(AB, CD) # argument order matters here

    E = crop_zeros(E, zero=100)
    # imwrite('../data/tmp/stitched-%s.tif' % (str(time())), E)
    imwrite('../data/stitched.tif', E)
    return (E, sum([t1,t2,t3]))

def Frequency(blocks):
    return Fourier(blocks)
