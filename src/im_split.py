import argparse
import math    # sqrt
import imageio # imread, imwrite
import imutils # image rotate bounds
import skimage # noise

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def im_split(fname, overlap=0.2, blocks=4, rotation=0, noise=0, **kwargs):
    '''
    Takes an image `im` and optionally a parameter `blocks` to determine how many blocks it should
    be split into. `blocks` must be a square number greater than 1. Overlap determines the
    percentage overlap between the blocks, for example, if `p=0.2` 20% of an image will
    be common with its neighboring image.
    '''
    assert blocks > 1, 'blocks must be greater than one'
    assert math.sqrt(blocks).is_integer(), '√blocks must be a whole integer'
    assert overlap >= 0 and overlap <= 1, 'overlap must be in the range 0...1'

    im = imageio.imread('%s' % fname)

    rows = int(math.sqrt(blocks))
    width = im.shape[0]
    height = im.shape[1]

    ratio = 1 / ((1/overlap) + (rows-1))
    output_w = height * ratio*2
    offset = output_w * overlap
    output_h = output_w

    output_images = []
    # print('b:%d o:%d' % (output_w, offset))
    for r in range(rows):
        r_start = int(max(output_w * r - offset * r, 0))
        r_end   = int(min(r_start + output_w, height))
        for c in range(rows):
            c_start = int(max(output_w * c - offset * c, 0))
            c_end   = int(min(c_start + output_w, width))
            # print("%d:%d , %d:%d" % (r_start, r_end, c_start, c_end))
            block = im[r_start:r_end,c_start:c_end]
            # Do not transform the first block.
            if r+c == 0:
                output_images.append(block)
                continue
            if noise > 0:
                mean = 0.0
                std = noise
                block = skimage.util.random_noise(block, mode='gaussian', mean=mean, var=std)
                block *= (2**16)-1               # scale pixels back to 0..65535
                block = block.astype('uint16')   # cast back to uint16
            if rotation > 0:
                block = imutils.rotate_bound(block, rotation)

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
    parser.add_argument('-blocks', '-b', help='number of blocks, must be a square number', type=int, default=4)
    parser.add_argument('-file', '-f', help='image filename', type=str, default='../data/T1_Img_002.00.tif')
    args = parser.parse_args()
    kw = vars(args)
    fname = kw['file']
    imgs = im_split(fname, **kw)
    write_ims(imgs, fname.split('_')[0], **kw)
