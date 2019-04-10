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
from Direct import iterative_ssd, iterative_ncc, iterative_mi
from Fourier import Frequency, Frequency_highpass
from util import clamp_uint16, equalize_image


def eval_method(image_name, method, **kwargs):
    original = imread(image_name)
    if kwargs['downsample'] > 0:
        original = resize(original, width=kwargs['downsample'])
    blocks = im_split(image_name, **kwargs)
    stitched, duration = method(blocks)
    if duration == None:
        return (0, np.NAN, np.NAN)
    acc_result = i_RMSE(stitched, original)
    suc_result = acc_result < 0.10
    return (duration, acc_result, suc_result)


def eval_param(image_name, method, param, data_range, downsample=False, overlap=0.2):
    row = []
    if param != 'overlap':
        print('%s: %s, overlap: %0.2f' % (method.__name__, param, overlap))
    else:
        print('%s: %s, overlap: %0.2f' % (method.__name__, param, data_range[0]))

    for val in data_range:
        kw = { param: val, 'downsample': downsample, }
        if param != 'overlap':
            kw['overlap'] = overlap
        duration, err, suc = eval_method(image_name, method, **kw)
        print("%s: %0.2f, t: %0.2f, err: %0.2f suc: %0.2f" % (param, val, duration, err, suc))
        row.append('(%.02f, %0.02fs)' % (err, duration))
    return row


def run_eval(image_name, method, noise=False, rotation=False, overlap=False, \
             downsample=False, **kwargs):
    # if no arguments are provided all are run.
    if not noise and not rotation and not overlap:
        noise    = True
        rotation = True
        overlap  = True

    overlap_range = [o/100 for o in range(10, 101, 10)]

    out = {}
    if noise:
        noise_range = [d / 100 for d in range(0,11,2)]
        table = []
        for o in overlap_range:
            kw = {'overlap': o, 'downsample': downsample}
            table.append(eval_param(image_name, method, 'noise', noise_range, **kw))

        df = pd.DataFrame(table, columns=noise_range, index=overlap_range)
        df.index.names = ['overlap']
        df.columns.names = ['noise']
        df.columns = map(lambda x: '%.02fdb' % (x), noise_range)
        out['noise'] = df

    if rotation:
        rot_range = range(-45,46,15)
        table = []
        for o in overlap_range:
            kw = {'overlap': o, 'downsample': downsample}
            table.append(eval_param(image_name, method, 'rotation',
                                    rot_range, **kw))
        df = pd.DataFrame(table, columns=rot_range, index=overlap_range)
        df.index.names = ['overlap']
        df.columns.names = ['rotation']
        df.columns = map(lambda x: '%d°' % (x), rot_range)
        out['rotation'] = df

    if overlap:
        table = eval_param(image_name, method, 'overlap', overlap_range, downsample)
        df = pd.DataFrame(table).transpose()
        df.index.names = ['overlap']
        df.columns = map(lambda x: '%.0f%%' % (x*100), overlap_range)
        out['overlap'] = df

    return out

def i_RMSE(stitched, original):
    oh,ow = original.shape
    # make sure these are the same size.
    if stitched.shape[0] != oh or stitched.shape[1] != ow:
        stitched = equalize_image(stitched, original.shape)

    stitched = stitched.astype('float64') / (2**16 -1)
    original = original.astype('float64') / (2**16 -1)
    abs_err = (original - stitched) ** 2
    return math.sqrt(abs_err.mean())

def method_picker(name):
    methods = [Frequency, Frequency_highpass, AKAZE, iterative_ssd, iterative_ncc, iterative_mi]
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
    parser.add_argument('-output', '-o', action='store_true', help='output results to LaTeX table')

    args = parser.parse_args()
    kw = vars(args)
    # get method to evaluate
    method = kw['method']
    kw['method'] = method_picker(method)

    # evaluate!
    results = run_eval(kw['file'], **kw)

    # write output
    for k in results.keys():
        print(results[k])
        if kw['output']:
            latex_str = results[k].to_latex()
            with open('%s_%s.tex' % (method, k), 'w') as fi:
                caption = '%s results for %s method.' % (k, method)
                if kw['downsample']:
                    ds = kw['downsample']
                    caption += ' Base image downsampled to $%d\\times%d$ pixels.' % (ds, ds)
                fi.write('\\begin{table}\n\\centering\n')
                fi.write(latex_str)
                fi.write('\\caption{%s}\n' % caption)
                fi.write('\\end{table}\n')
