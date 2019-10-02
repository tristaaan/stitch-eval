import argparse
import copy
import gc
import glob
import math
import os
import time

import numpy as np
import pandas as pd

from cv2 import imread
from os import path

from im_split import im_split
from Fiducials import Fiducial_corners, group_transform, \
                      group_transform_affine, zero_group
from measurements import fiducial_point_error
from visualization import saveimfids, plot_results

def build_fiducials(initial, transforms, affine=False):
    '''
    Given initial fiducials and transformations from the stitching process
    apply transformations to initial fiducials.
    '''
    # no transforms to reference images
    A = initial[0]
    C = initial[2]

    # transform moving images
    if affine:
        B = initial[1].transform_affine(transforms[0])
        D = initial[3].transform_affine(transforms[1])
    else:
        B = initial[1].transform(*transforms[0], unit='d')
        D = initial[3].transform(*transforms[1], unit='d')
        # keep cd coordinates relative
        zero_group([C,D])


    # get the center point of moving image CD
    min_x = min(f.min_x() for f in [C,D])
    max_x = max(f.max_x() for f in [C,D])

    min_y = min(f.min_y() for f in [C,D])
    max_y = max(f.max_y() for f in [C,D])

    temp_center = (max_x - (max_x-min_x) / 2, max_y - (max_y-min_y) / 2)
    # transform CD
    if affine:
        group_transform_affine([C,D], transforms[2], temp_center=temp_center)
    else:
        x,y,th = transforms[2]
        x += B.min_x() if round(B.min_x()) <= round(A.min_x()) else 0
        y += B.min_y() if round(B.min_y()) <= round(A.min_y()) else 0
        group_transform([C,D], x,y,th , unit='d', temp_center=temp_center)
    return [A, B, C, D]

def header_row(param):
    if param == 'overlap':
        return ['overlap', 'error', 'time', 'success', 'total']

    return ['overlap', param, 'error', 'time', 'success', 'total']

def result_row(overlap, param, p_val, err, duration, success, total):
    if param == 'overlap':
        return { 'overlap': p_val, 'error': err, 'time': duration,
            'success': success, 'total': total }

    return { 'overlap': overlap, param: p_val, 'error': err,
        'time': duration, 'success': success, 'total': total }

def eval_method(image_name, method, debug=False, **kwargs):
    '''
    Evaluate image stitching given an image, a method, and the splitting params
    '''
    blocks, ground_truth, initial = im_split(image_name, fiducials=True, **kwargs)
    stitched, transforms, duration = method(blocks)

    # there was a catastrophic failure
    if duration == None:
        return (0, np.NAN)
    # with the transformations, construct estimated fiducials
    est_fiducials = build_fiducials(initial, transforms, \
                                    affine=hasattr(transforms[0],'shape'))

    # compare the estimated ones with the ground_truth ones.
    acc_result = fiducial_point_error(ground_truth, est_fiducials)

    # if debug, write the stitched image.
    if debug:
        # print(transforms[0], '\n', transforms[1], '\n', transforms[2])
        # fname = '../data/tmp/%s_%d_%.02f.tif' % \
        #         (method.__name__, int(kwargs['overlap']*100), acc_result)
        # saveimfids(fname, stitched, copy.deepcopy(est_fiducials), truthy=ground_truth)
        gc.collect() # cleans up matplot lib junk

    return (duration, acc_result)


def eval_param(inputs, method, param, data_range, overlap=0.2,
               downsample=False, record_all=False, debug=False):
    '''
    Run a study on a method given a single range of parameters with overlap
    '''
    row = []
    if param != 'overlap':
        print('%s: %s, overlap: %0.2f' % (method.__name__, param, overlap))
    else:
        print('%s: %s, overlap: %0.2f' % (method.__name__, param, data_range[0]))

    multi_file = type(inputs) == list
    if not multi_file:
        image_name = inputs

    if downsample:
        max_dim = downsample
    else:
        max_dim = max(imread(inputs[0]).shape)
    err_thresh = max_dim / 10

    for val in data_range:
        kw = { param: val, 'downsample': downsample, 'debug': debug }
        if param != 'overlap':
            kw['overlap'] = overlap
        # perform evaluations over a directory of images
        if multi_file:
            success_count = 0
            duration_sum = 0
            err_sum = 0
            # evaluate each image
            for f in inputs:
                record = False
                duration, err = eval_method(f, method, **kw)
                # tally success
                if err <= err_thresh and err != np.NAN:
                    success_count += 1
                    record = True
                # add duration and error to sums
                if record or record_all:
                    duration_sum += duration
                    err_sum += err

            # calculate average error
            if not record_all and success_count == 0:
                err = np.NAN
                duration = np.NAN
            elif record_all:
                err = err_sum / len(inputs)
                duration = duration_sum / len(inputs)
            else:
                err = err_sum / success_count
                duration = duration_sum / success_count
            # append to dataframe
            row.append(
                result_row(overlap, param, val, err, duration, success_count, len(inputs))
            )
            print("%s: %0.2f, t: %0.2f, err: %0.2f, suc: %d/%d" %
                (param, val, duration, err, success_count, len(inputs)))
        # perform evaluations on a single image
        else:
            duration, err = eval_method(image_name, method, **kw)
            print("%s: %0.2f, t: %0.2f, err: %0.2f" % (param, val, duration, err))
            row.append(result_row(overlap, param, val, err, duration,
                                  int(err <= err_thresh), 1))
    return row

def run_eval(inputs, method, noise=False, rotation=False, overlap=False,
             downsample=False, o_range=[20,80,10], r_range=[60,61,10], **kwargs):
    '''
    Run a study on a method with a variety of parameters on a single image:
    overlap + [noise, rotation]
    returns a single dataframe with statistics.
    '''
    multi_file = path.isdir(inputs)
    if path.isdir(inputs):
        files = get_images_from_directory(inputs)
    else:
        image_name = inputs

    # if no arguments are provided all are run.
    if not noise and not rotation and not overlap:
        noise    = True
        rotation = True
        overlap  = True

    overlap_range = [o/100 for o in range(o_range[0], o_range[1]+1, o_range[2])]
    # overlap_range = [0.60, 0.75] # for smaller debug runs
    out = {}
    kw = {'downsample': downsample,
          'debug': kwargs['debug'],
          'record_all': kwargs['record_all']}
    if noise:
        param = 'noise'
        noise_range = [d / 1000 for d in range(0,51,10)]
        df = pd.DataFrame([], columns=header_row(param))
        for o in overlap_range:
            kw['overlap'] = o
            if multi_file:
                df = df.append(eval_param(files, method, param, noise_range, **kw))
            else:
                df = df.append(eval_param(image_name, method, param, noise_range, **kw))
        out[param] = df

    if rotation:
        param = 'rotation'
        r_range[1] += 1
        rot_range = range(*r_range)
        # rot_range = [-15, 0, 15] # for smaller debug runs

        df = pd.DataFrame([], columns=header_row(param))
        for o in overlap_range:
            kw['overlap'] = o
            if multi_file:
                df = df.append(eval_param(files, method, param, rot_range, **kw))
            else:
                df = df.append(eval_param(image_name, method, param, rot_range, **kw))
        out[param] = df

    if overlap:
        param = 'overlap'
        if multi_file:
            table = eval_param(files, method, param, overlap_range, **kw)
        else:
            table = eval_param(image_name, method, param, overlap_range, **kw)

        df = pd.DataFrame(table, columns=header_row(param))
        out[param] = df

    return out

def get_images_from_directory(directory):
    types = ('*.tif*', '*.png', '*.jpeg', '*.jpg') # cover most common file types
    files = []
    for t in types:
        files.extend(glob.glob(path.join(directory, t)))
    return files

def method_picker(name):
    from Feature import AKAZE, SIFT, SURF
    from amd_alpha import amd_alpha
    # from Direct import iterative_ssd, iterative_ncc, iterative_mi
    from Fourier import Frequency

    methods = [AKAZE, SIFT, SURF,
        amd_alpha,
        Frequency,
        # iterative_ssd, iterative_ncc, iterative_mi
    ]

    # Do not import learning methods unless necessary, no need to instantiate
    # Tensorflow and Keras unless we need to.
    if name == 'TN' or name == 'DHN':
        from Learning import DHN, TN
        methods += [DHN, TN]

    method_names = list(map(lambda x: x.__name__.lower(), methods))
    return methods[method_names.index(name.lower())]

def arg_range(s):
    arr = list(map(int, s.split(':')))
    assert len(arr) >= 2, 'there must be start and end to the range: %s' % s

    start,end = arr[:2]
    if len(arr) == 3:
        stride = arr[2]
    else:
        stride = 10
    return [start, end, stride]

def make_results_folder(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluations with a method')
    # eval parameters
    parser.add_argument('-noise',    action='store_true', help='run noise evaluations')
    parser.add_argument('-rotation', action='store_true', help='run rotation evaluations')
    parser.add_argument('-overlap',  action='store_true', help='run overlap evaluations')

    parser.add_argument('-o_range', help='range of overlap start:end:stride', type=arg_range, \
                        action='store', default=[20,80,10])
    parser.add_argument('-r_range', help='range of rotation, start:end:stride, for negative values use -r_range="..."',
                        type=arg_range, action='store', default=[-45,45,15])

    # other options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-file', '-f', help='image filename', type=str,
                             default='../data/T1_Img_002.00.tif')
    input_group.add_argument('-dir', '-d',
                             help='run evaluations on all images in a directory')

    parser.add_argument('-method', '-m',     help='method to evaluate', \
                        type=str, default='akaze')
    parser.add_argument('-downsample', '-ds', help='downsample images', \
                        type=int, default='-1')
    parser.add_argument('-output', '-o', action='store_true',   \
                        help='output results to csv')
    parser.add_argument('-debug', action='store_true',   \
                        help='write the stitched image after each stitch')
    parser.add_argument('-record_all', '-r', help='record all errors and average ' +
                        'them regardless if they are under the error threshold', \
                        action='store_true', default='akaze')
    parser.add_argument('-tex', action='store_true',   \
                        help='output results to LaTeX table')
    parser.add_argument('-viz', '-vis', action='store_true',   \
                        help='create a heatmap of the results')

    pd.options.display.float_format = '{:,.2f}'.format
    args = parser.parse_args()
    kw = vars(args)
    # get method to evaluate
    method = kw['method']
    kw['method'] = method_picker(method)

    # evaluate!
    if kw['dir']:
        results = run_eval(kw['dir'], **kw)
    else:
        results = run_eval(kw['file'], **kw)

    # write output
    for k in results.keys():
        print(results[k])

        # create output folder if needed
        folder_name = 'results'
        outname = os.path.join(folder_name, '%s_%s_%s' % \
                               (method, k, time.strftime('%d-%m_%H:%M')))
        # create output folder
        if kw['viz'] or kw['tex'] or kw['output']:
            make_results_folder(folder_name)

        # save csv
        if kw['output']:
            results[k].to_csv(outname + '.csv', float_format='%0.03f')

        # output visualization
        if kw['viz']:
            if kw['downsample'] > 0:
                image_size = kw['downsample']
            elif kw['dir']:
                files = get_images_from_directory(kw['dir'])
                image_size = max(imread(files[0]).shape)
            else:
                image_size = max(imread(kw['file']).shape)
            plot_results(outname, results[k], k, image_size=image_size)

        # create latex table output
        if kw['tex']:
            latex_str = results[k].to_latex() \
                                  .replace('°', '\\degree') # usepackage{gensymb}
            results[k].to_csv(outname + '.csv')
            with open(outname + '.tex', 'w') as fi:
                caption = ('%s results for %s method.' % (k[0].upper() + k[1:],
                                                         method)).replace('_', '\\_')
                if kw['downsample']:
                    ds = kw['downsample']
                    caption += ' Base image downsampled to $%d\\times%d$ pixels.' % (ds, ds)
                fi.write('\\begin{table}\n\\centering\n')
                fi.write(latex_str)
                fi.write('\\caption{%s}\n' % caption)
                fi.write('\\end{table}\n')

    print('done')

