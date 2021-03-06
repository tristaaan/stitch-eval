import argparse
import math    # sqrt
import imageio # imread, imwrite
import imutils # rotate, resize
import skimage # noise

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from Fiducials import Fiducial_corners
from util import uint16_to_uint8

def im_split(fname, overlap=0.2, rotation=0, noise=0, downsample=-1, \
             fiducials=False, **kwargs):
    '''
    Takes an image `im` and splits it into four blocks given some overlap
    percentage. Optionally rotation and noise can also be applied
    a downsample argument can be passed to downsample the original image.
    '''
    assert overlap >= 0 and overlap <= 1, 'overlap must be in the range 0...1 inclusive'

    im = imageio.imread('%s' % fname)
    if downsample > 0:
        im = imutils.resize(im, width=downsample)
    rows = 2
    height, width = im.shape[:2]

    stride = None
    output_w = 0
    output_h = 0

    # block_side = L / 2-p
    base_block_w = width / rows
    base_block_h = height / rows

    output_w = width  / (2 - overlap)
    output_h = height / (2 - overlap)

    stride_w = output_w - (output_w - base_block_w)*2
    stride_h = output_h - (output_h - base_block_h)*2

    output_images = []
    ground_truth_corners = [] # ground truth to compare to
    initial_corners      = [] # initial fiducials to transform
    for r in range(rows):
        r_start = int(stride_h * r)
        r_end   = int(min(r_start + output_h, height))
        for c in range(rows):
            c_start = int(stride_w * c)
            c_end   = int(min(c_start + output_w, width))
            block = im[r_start:r_end,c_start:c_end]
            base_shape = (r_end-r_start, c_end-c_start)
            # Do not transform the first block.
            if r+c == 0:
                output_images.append(block)
                ground_truth_corners.append(Fiducial_corners(base_shape,
                                        initial_transform=[0, 0, 0]))
                initial_corners.append(Fiducial_corners(base_shape,
                                        initial_transform=[0, 0, 0]))
                continue
            if noise > 0:
                mean = 0.0
                std = noise
                orig_dtype = block.dtype
                # adding noise converts image scale from 0..1
                block = skimage.util.random_noise(block, mode='gaussian', \
                                                  mean=mean, var=std,     \
                                                  seed=1234)
                # scale pixels back to range depending on dtype
                if orig_dtype == 'uint16':
                    block *= (2**16)-1
                elif orig_dtype == 'uint8':
                    block *= (2**8)-1
                block = block.astype(orig_dtype)

            if rotation != 0:
                block = imutils.rotate_bound(block, rotation)

            output_images.append(block)
            ground_truth_corners.append(Fiducial_corners(base_shape,
                                        initial_transform=[c_start, r_start, 0]))
            initial_corners.append(Fiducial_corners(base_shape,
                                        initial_transform=[0,0,rotation],
                                        unit='d'))

    if fiducials:
        return output_images, ground_truth_corners, initial_corners
    return output_images

def im_split_blocks(fname, blocks=4, downsample=-1):
    '''
    Takes an image `im` and a parameter `blocks` to determine how many blocks
    it should    be split into. `blocks` must be a square number greater than 1.
    There is no overlap for images
    generated from this method. It is just for splitting up images.
    '''
    assert blocks > 1, 'blocks must be greater than one'
    assert math.sqrt(blocks).is_integer(), '√blocks must be a whole integer'

    im = imageio.imread('%s' % fname)
    if downsample > 0:
        im = imutils.resize(im, width=downsample)
    rows = int(math.sqrt(blocks))
    height, width = im.shape[:2]

    output_w = im.shape[1] / rows
    output_h = im.shape[0] / rows

    output_images = []
    for r in range(rows):
        r_start = int(output_h * r)
        r_end   = int(min(r_start + output_h, height))
        for c in range(rows):
            c_start = int(output_w * c)
            c_end   = int(min(c_start + output_w, width))
            block = im[r_start:r_end,c_start:c_end]
            base_shape = (r_end-r_start, c_end-c_start)

            block = im[r_start:r_end,c_start:c_end]
            output_images.append(block)

    return output_images


def plot_blocks(imgs, bars=False):
    fig = plt.figure(figsize=(9, 8), dpi=72)
    rows = int(math.sqrt(blocks))
    for i in range(1, len(imgs)+1):
        fig.add_subplot(rows, rows, i)
        plt.imshow(imgs[i-1], cmap=plt.cm.gray)
        if bars:
            # left
            axs = plt.gca()
            ax,ay = (imgs[i-1].shape[0] * overlap, 0)
            bx,by = (imgs[i-1].shape[0] * overlap, imgs[i-1].shape[1])
            l = mlines.Line2D([ax, bx],
                              [ay, by], color='r')
            axs.add_line(l)
            # right
            ax,ay = (imgs[i-1].shape[0] * (1-overlap), 0)
            bx,by = (imgs[i-1].shape[0] * (1-overlap), imgs[i-1].shape[1])
            l = mlines.Line2D([ax, bx],
                              [ay, by], color='r')
            axs.add_line(l)
            # top
            ax,ay = (0, imgs[i-1].shape[1] * overlap)
            bx,by = (imgs[i-1].shape[1], imgs[i-1].shape[0] * overlap)
            l = mlines.Line2D([ax, bx],
                              [ay, by], color='r')
            axs.add_line(l)
            # bottom
            ax,ay = (0, imgs[i-1].shape[1] * (1-overlap))
            bx,by = (imgs[i-1].shape[1], imgs[i-1].shape[0] * (1-overlap))
            l = mlines.Line2D([ax, bx],
                              [ay, by], color='r')
            axs.add_line(l)
    plt.show()


def write_ims(ims, prefix, rotation=False, noise=False, **kwargs):
    flags = ''
    if rotation:
        flags += '_rot'
    if noise:
        flags += '_noise'
    for (ind, im) in enumerate(ims):
        imageio.imwrite('../data/%s_segment%s_%d.tif' % (prefix, flags, ind+1), im[:, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split an image into blocks' \
                                            + 'with some overlap, rotation,'  \
                                            + 'and/or noise')
    parser.add_argument('-noise', '-n', help='add noise', type=float, default=0)
    parser.add_argument('-rotation', '-r', help='add rotation', type=int, default=0)
    parser.add_argument('-overlap', '-o', help='overlap percentage [0...1]', type=float, default=0.2)
    parser.add_argument('-blocks', '-b', help='number of blocks, must be a square number', type=int)
    parser.add_argument('-downsample', '-ds', help='downsample output images', type=int, default=-1)
    parser.add_argument('-file', '-f', help='image filename', type=str, default='../data/T1_Img_002.00.tif')
    args = parser.parse_args()
    kw = vars(args)
    fname = kw['file']
    if kw['blocks'] :
        imgs = im_split_blocks(fname, blocks=kw['blocks'], downsample=kw['downsample'])
    else:
        imgs = im_split(fname, **kw)
    write_ims(imgs, fname.split('_')[0], **kw)
