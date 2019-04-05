import imageio

import numpy as np
import scipy as sp

from collections import deque
from imageio import imread, imwrite
from math import sqrt
from PIL import Image
from scipy.ndimage import rotate
from skimage.transform import resize
from sklearn.linear_model import LinearRegression
from time import time

from util import bounds_equalize, crop_zeros, eq_paste, paste

def fill_fn(fn, size):
    name = fn.__name__
    if name == 'NCC' or name == 'MI' or name == 'SRD':
        return np.zeros(size)
    elif name == 'SSD' or name == 'SAD':
        return np.full(size, np.inf)
    else:
        raise ValueError('Unrecognized measurement method: %s' % name)

def objective_function(measurement):
    name = measurement.__name__
    if name == 'NCC' or name == 'MI' or name == 'SRD':
        return np.argmax
    elif name == 'SSD' or name == 'SAD':
        return np.argmin
    else:
        raise ValueError('Unrecognized measurement method: %s' % name)

def convolve(im, template, op):
    h_len, w_len = im.shape
    th, tw = template.shape
    offset_h, offset_w = (int(th/2), int(tw/2))
    op_res = fill_fn(op, (h_len, w_len))
    for i in range(h_len-th+1):
        for j in range(w_len-tw+1):
            # slice im to be the area under the template
            op_res[i+offset_h, j+offset_w] = op(im[i:i+th, j:j+tw], template)
    return op_res


# distances
def SSD(i1, i2):
    '''
    Sum squared differences, minimize this
    '''
    assert sum(i1.shape) == sum(i2.shape), 'i1 and i2 are different shapes "%s" v "%s"' % (i1.shape, i2.shape,)
    sum_square = np.square(i2 - i1).sum() # (A - B)^2
    return sum_square

def SAD(i1, i2):
    '''
    sum absolute differences, minimize this
    '''
    assert sum(i1.shape) == sum(i2.shape), 'i1 and i2 are different shapes "%s" v "%s"' % (i1.shape, i2.shape,)
    sum_absolute = (abs(i2 - i1)).sum() # | A - B |
    return sum_absolute

# correlation
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

def lpc(a,b):
    '''
    create a 2d linear prediction and return the coefficients
    '''
    lr = LinearRegression()
    lr.fit(a,b)
    return lr.coef_

def NCC_whiten(im1, im2):
    '''
    whiten the images based on a '2D prediction error filter' of the reference image (im1)
    '''
    a_pred = lpc(im1, im2)
    im1 += np.square(abs(im1 - a_pred))
    im2 += np.square(abs(im2 - a_pred))
    return (im1, im2)

# mutual information
def MI(X,Y):
    '''
    Mutual information, maximize this
    '''
    bins = 256
    X = np.ravel(X)
    Y = np.ravel(Y)
    c_XY = np.histogram2d(X,Y, bins=bins)[0]
    c_X = np.histogram(X, bins=bins)[0]
    c_Y = np.histogram(Y, bins=bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    return H_X + H_Y - H_XY

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    return -sum(c_normalized* np.log2(c_normalized)) # same as scipy.stats.entropy(c_norm, base=2)

def iterative(ref_src, mov_src, measurement=NCC, theta_range=(-45,45), im_filter=None):
    ref = ref_src
    mov = mov_src
    # when the image is rotated it may be larger, fill this value in the new places.
    inorm = 2**15 # middle value of uint16

    # initialize transformation parameters
    prev_x, prev_y = (np.inf, np.inf)
    x,y = (0,0)
    prev_theta = np.inf
    total_theta = 0

    # terminate when there are no more translational and rotational changes
    while x != prev_x or y != prev_y: #or total_theta != prev_theta:
        # rotate from previous transform
        mov = mov_src.copy()
        if total_theta != 0:
            mov = rotate(mov_src, total_theta, reshape=False)
            mov[mov == 0] = inorm
        if im_filter != None:
            ref = im_filter(ref)
            mov = im_filter(mov)
        # find best x,y
        p = int(mov.shape[0])
        ref_padded = np.pad(ref, ((p,p), (p,p)), 'constant', constant_values=(0,0))
        xy_corr = convolve(ref_padded, mov, measurement)
        prev_x, prev_y = (x,y)
        y,x = np.unravel_index(objective_function(measurement)(xy_corr), xy_corr.shape)
        p2 = int(p/2)

        # # find best theta
        # thetas = list(range(theta_range[0], theta_range[1]+1))
        # theta_corr = []
        # for theta in thetas:
        #     # rotate moving image, may change here if reshape=True.
        #     mov_r = rotate(mov_src, total_theta-theta, reshape=False)
        #     mov_r[mov_r == 0] = inorm
        #     if im_filter != None:
        #         mov_r = im_filter(mov_r)
        #     orig_ref = ref.shape[0]
        #     p2 = int(orig_ref / 2)
        #     diff = mov_r.shape[0] - orig_ref
        #     margin_s = int(diff / 2)
        #     margin_e = mov_r.shape[0] - margin_s
        #     ref_range = ref_padded[y-p2:y+p2, x-p2:x+p2]
        #     if ref_range.shape != mov_r[margin_s:margin_e, margin_s:margin_e].shape:
        #         ref_range = ref_padded[y-p2:y+p2+1, x-p2:x+p2+1]
        #     c = measurement(ref_range, mov_r[margin_s:margin_e, margin_s:margin_e])
        #     # store measurement value
        #     theta_corr.append(c)
        # # recover the best theta
        # prev_theta = total_theta
        # delta_theta = thetas[objective_function(measurement)(theta_corr)]
        # total_theta -= delta_theta
        mov_r = rotate(mov_src, total_theta, reshape=False)

    # print(x-p, y-p, -total_theta)
    return (x-p, y-p, -total_theta)

def iterative_generic(blocks, measurement=NCC):
    queue = deque(blocks)
    base = queue.popleft()
    ind = 0
    total_time = 0
    while len(queue) > 0:
        # load images
        im2 = queue.popleft()
        # base, im2 = bounds_equalize(base, im2)

        # start timer
        start = time()
        x,y,angle = iterative(base, im2, measurement=measurement)

        if angle != 0:
            im2 = rotate(im2, angle, reshape=False)

        im2 = crop_zeros(im2, zero=100)
        base = crop_zeros(base, zero=100)
        half_y = int(im2.shape[0] / 2)
        half_x = int(im2.shape[1] / 2)
        ny = max(y - half_y, 0)
        nx = max(x - half_x, 0)
        im2 = np.pad(im2, ((ny, 0), (nx, 0)), 'constant', constant_values=(0,0))
        # paste
        base = eq_paste(base, im2)
        total_time += time() - start
        print('stitched %d with %d' % (ind, ind + 1))
        ind += 1

    base = crop_zeros(base, zero=100)
    imwrite('../data/stitched.tif', base)
    average_time = total_time / (len(blocks) - 1)
    return (base, average_time)

def iterative_ssd(blocks):
    return iterative_generic(blocks, measurement=SSD)

def iterative_ncc(blocks):
    return iterative_generic(blocks, measurement=NCC)

def iterative_mi(blocks):
    return iterative_generic(blocks, measurement=MI)