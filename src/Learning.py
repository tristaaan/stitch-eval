import math
from time import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from imutils import rotate_bound
from imageio import imwrite

from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Dense, Flatten, MaxPooling2D, Input, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD

from util import pad, merge, crop_zeros, uint16_to_uint8

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

def conv_block(m, filters):
    kernel = (3, 3)
    m = Conv2D(filters, kernel, padding='same', activation='relu')(m)
    return BatchNormalization()(m)

def conv_group(m, filters):
    m = conv_block(m, filters)
    m = conv_block(m, filters)
    return MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m)

def homography_regression_model(input_dims, translation=False):
    '''
    initialize model, if translation only four_pt=False
    '''
    input_shape=(*input_dims, 2)

    input_layer = Input(shape=input_shape, name='input_layer')

    # 4x 64
    x = conv_group(input_layer, 64) # 1,2
    x = conv_group(x,           64) # 3,4
    # 4x 128
    x = conv_group(x,           128) # 1,2
    x = conv_block(x,           128) # 3
    x = conv_block(x,           128) # 4
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, name='FC_1024')(x)
    x = Dropout(0.5)(x)

    if translation:
        out = Dense(2, name='output')(x)
    else:
        out = Dense(8, name='output')(x)

    return Model(inputs=input_layer, outputs=[out])


def homography_net():
    opt = SGD(lr=0.005, momentum=0.9, decay=0.0)
    model = homography_regression_model((128,128))
    model.compile(optimizer=opt, loss=euclidean_distance)
    model.load_weights('weights/hnet-weights-rigid-train.hdf5')
    return model


def translation_net():
    opt = SGD(lr=0.001, momentum=0.9, decay=0.0)
    model = homography_regression_model((128,128), translation=True)
    model.compile(optimizer=opt, loss=euclidean_distance)
    model.load_weights('weights/tnet-weights-multiscale.hdf5')
    return model

def invert(im):
    return abs(255 - im)

def points_to_affine(shape, H_4pt):
    h,w = shape
    pts1 = np.array([[0, 0],
                     [0, h],
                     [w, h],
                     [w, 0]], dtype='float32')
    pts2 = H_4pt + pts1
    return cv2.findHomography(pts1, pts2)[0][:2,:3]

def warp_merge(im1, im2, h):
    '''
    Grab the translation and rotation components and apply them.
    This is more reliable than warpAffine because no parts of the image are lost
    '''
    x, y = h[0:2, 2]
    # x = x/128 * im1.shape[1]
    # y = y/128 * im1.shape[0]
    th1 = math.atan(-h[0,1] / h[0,0]) * (180 / math.pi) # atan(-b/a)
    th2 = math.atan(h[1,0] / h[1,1]) * (180 / math.pi)  # atan(c/d)
    th = (th1 + th2) / 2
    warped = rotate_bound(im2, th)
    return merge(im1, warped, x, y), [x,y,th]

def resize(im, size):
    return cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)

def est_transform(im1, im2, model, orig_size):
    '''
    Input two images into the model and return the estimated transform
    this could be more optimized by batching the transformation estimations
    instead of one at a time.
    '''
    stack = np.dstack( (invert(im1), invert(im2)) )
    stack = stack.reshape(1, *stack.shape)
    H = model.predict(stack)[0]
    if len(H) == 8:
        return H.reshape(4,2)
    return H * list(orig_size)[::-1] # scale results

def stitch_blocks(blocks, model, size):
    # save original size
    orig_size = blocks[0].shape

    # cast to uint8 if necessary
    # in either case scale down images
    if blocks[0].dtype == 'uint16':
        A,B,C,D = map(lambda x: resize(x, size), map(uint16_to_uint8, blocks))
    else:
        A,B,C,D = map(lambda x: resize(x, size), blocks)

    start = time()
    # get transforms
    t_AB = est_transform(A, B, model, orig_size)
    t_CD = est_transform(C, D, model, orig_size)

    # for the vertical component take the mean of AC and BD
    t_AC = est_transform(A, C, model, orig_size)
    t_BD = est_transform(B, D, model, orig_size)
    t_v = (t_AC + t_BD) / 2

    # use original images from here on
    A,B,C,D = blocks

    # translation net
    if t_AB.size == 2:
        # stitch
        im_ab = merge(A, B, *t_AB)
        im_cd = merge(C, D, *t_CD)
        final = merge(im_ab, im_cd, *t_v)

        # add theta component to transform
        t_AB = t_AB.tolist() + [0]
        t_CD = t_CD.tolist() + [0]
        t_v  = t_v.tolist()  + [0]
        return (final, [t_AB, t_CD, t_v], time() - start)

    # regular hnet
    shape = A.shape
    h_AB = points_to_affine(shape, t_AB)
    h_CD = points_to_affine(shape, t_CD)
    h_AC = points_to_affine(shape, t_AC)
    h_BD = points_to_affine(shape, t_BD)
    h_v = (h_AC + h_BD) / 2

    # stitch
    im_ab, t1 = warp_merge(A, B, h_AB)
    im_cd, t2 = warp_merge(C, D, h_CD)
    final, t3 = warp_merge(im_ab, im_cd, h_v)

    # affine transforms kept as np arrays
    return (final, [t1,t2,t3], time() - start)

def TN(blocks, model):
    return stitch_blocks(blocks, model, (128,128))

def DHN(blocks, model):
    return stitch_blocks(blocks, model, (128,128))
