import argparse
import math
import random
import numpy as np
import pandas as pd

from imageio import imread, imwrite
from imutils import resize
from operator import mul

from im_split import im_split
from AKAZE import AKAZE
from Fourier import Fourier
from util import clamp_uint16, equalize_image


def eval_method(image_name, method, **kwargs):
    original = imread(image_name)
    if kwargs['downsample'] > 0:
        original = resize(original, width=kwargs['downsample'])
    blocks = im_split(image_name, **kwargs)
    stitched, duration = method(blocks)
    acc_result = i_RMSE(stitched, original)
    suc_result = success_measurement(stitched, original)
    return (duration, acc_result, suc_result)


def eval_param(image_name, method, param, data_range, downsample):
    table = pd.DataFrame(columns=[param, 'time', 'error', 'success'])
    print('%s evaluation with %s' % (param, method.__name__))
    for val in data_range:
        kw = { param: val, 'downsample': downsample }
        duration, err, suc = eval_method(image_name, method, **kw)
        print("param: %0.2f, t: %0.2f, err: %0.2f suc: %0.2f" % (val, duration, err, suc))
        table = table.append(
        {param: val, 'time': duration, 'error': err, 'success': suc}, ignore_index=True)
    return table


def run_eval(image_name, method, noise=False, rotation=False, overlap=False, \
             downsample=False, **kwargs):
    # if no arguments are provided all are run.
    if not noise and not rotation and not overlap:
        noise    = True
        rotation = True
        overlap  = True

    out = []
    if noise:
        dr = [d / 100 for d in range(0,11,2)]
        out.append(eval_param(image_name, method, 'noise', dr, downsample))

    if rotation:
        dr = range(0,181,45)
        out.append(eval_param(image_name, method, 'rotation', dr, downsample))

    if overlap:
        dr = [o/100 for o in range(5, 51, 5)]
        out.append(eval_param(image_name, method, 'overlap', dr, downsample))

    return out

def i_RMSE(stitched, original):
    oh,ow = original.shape
    # make sure these are the same size.
    if stitched.shape[0] != oh or stitched.shape[1] != ow:
        stitched = equalize_image(stitched, original.shape)

    total_px = ow * oh
    stitched = stitched.astype('float64') / (2**16 -1)
    original = original.astype('float64') / (2**16 -1)
    abs_err = (original - stitched) ** 2
    return math.sqrt((1/total_px) * abs_err.sum())


def success_measurement(stitched, original):
    '''
    The two images should be the same size or very close.
    If they are not, there can be some measurement of success applied based
    on their difference in number of pixels
    '''
    s = stitched.shape
    o = original.shape
    if s[0] != o[0] or mul(*s) != mul(*o):
        diff  = abs(mul(*s) - mul(*o))
        total =     mul(*s) + mul(*o)
        return 1.0 - (diff / total)
    return 1.0


def method_picker(name):
    methods = [Fourier, AKAZE]
    method_names = list(map(lambda x: x.__name__.lower(), methods))
    return methods[method_names.index(name.lower())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluations with a method')
    # eval parameters
    parser.add_argument('-noise',    action='store_true', help='run noise evaluations')
    parser.add_argument('-rotation', action='store_true', help='run rotation evaluations')
    parser.add_argument('-overlap',  action='store_true', help='run overlap evaluations')

    # other options
    parser.add_argument('-file', '-f',       help='image filename', type=str,
                        default='../data/T1_Img_002.00.tif')
    parser.add_argument('-method', '-m',     help='method to evaluate', type=str, default='akaze')
    parser.add_argument('-downsample', '-ds', help='downsample images', type=int, default='-1')

    args = parser.parse_args()
    kw = vars(args)
    kw['method'] = method_picker(kw['method'])
    x = run_eval(kw['file'], **kw)
    for t in x:
        print(t)