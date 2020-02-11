import argparse
import copy
import gc
import glob
import math
import os
import time
import json

import numpy as np
import pandas as pd

from functools import reduce
from cv2 import imread
from os import path

from im_split import im_split
from Fiducials import Fiducial_corners, group_transform, \
                      group_transform_affine, zero_group
from measurements import average_corner_error
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
        return ['overlap', 'results', 'size']

    return ['overlap', param, 'results', 'size']

def result_row(overlap, param, p_val, results, size):
    if param == 'overlap':
        return { 'overlap': p_val, 'results': json.dumps(results), 'size': size }

    return { 'overlap': overlap, param: p_val, 'results': json.dumps(results),
        'size': size }

# model placeholder
learning_model = None

def eval_method(image_name, method, debug=False, **kwargs):
    '''
    Evaluate image stitching given an image, a method, and the splitting params
    '''
    global learning_model
    blocks, ground_truth, initial = im_split(image_name, fiducials=True, **kwargs)
    # it's expensive to reinitialize the learning models every time.
    method_name = method.__name__.lower()
    if method_name in ['tn', 'dhn']:
        if learning_model == None:
            if method_name == 'tn':
                from Learning import translation_net
                learning_model = translation_net()
            elif method_name == 'dhn':
                from Learning import homography_net
                learning_model = homography_net()
            else:
                raise Error('unknown method name')
        stitched, transforms, duration = method(blocks, learning_model)
    else:
        stitched, transforms, duration = method(blocks)

    # there was a catastrophic failure
    if duration == None:
        return (0, 'inf', 'inf', 'inf')
    # with the transformations, construct estimated fiducials
    est_fiducials = build_fiducials(initial, transforms, \
                                    affine=hasattr(transforms[0],'shape'))

    # compare the estimated ones with the ground_truth ones.
    average_err, min_err, max_err = average_corner_error(ground_truth, est_fiducials)

    # if debug, write the stitched image.
    if debug:
        print(transforms[0], '\n', transforms[1], '\n', transforms[2])
        print(average_err, min_err/4, max_err/4)
        fname = '../data/tmp/%s_%d_%.02f_%.02f.tif' % \
                (method.__name__, int(kwargs['overlap']*100), average_err, max_err/4)
        saveimfids(fname, stitched, copy.deepcopy(est_fiducials), truthy=ground_truth)
        gc.collect() # cleans up matplot lib junk

    return (duration, average_err, min_err, max_err)


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
    input_size = len(inputs)
    if not multi_file:
        image_name = inputs
        input_size = 1

    if downsample != False:
        image_size = downsample
    elif multi_file:
        image_size = max(imread(inputs[0]).shape)
    else: # not multi_file, no downsample
        image_size = max(imread(image_name).shape)

    for val in data_range:
        kw = { param: val, 'downsample': downsample, 'debug': debug }
        if param != 'overlap':
            kw['overlap'] = overlap
        # perform evaluations over a directory of images
        if multi_file:
            json_record = {}
            # evaluate each image
            print('Progress: 0/%d' % (input_size), end = '\r')
            duration_sum = 0
            success_sum = 0
            for i, f in enumerate(inputs):
                duration, err, min_err, max_err = eval_method(f, method, **kw)
                json_record[f] = {
                    'err': err,
                    'time': duration,
                    'min': min_err,
                    'max': max_err
                }
                if max_err != 'inf' and max_err/4 <= 1:
                    success_sum += 1

                # print progress and estimated time remaining
                duration_sum += duration
                average_duration = duration_sum / (i + 1)
                m, s = divmod((input_size - (i+1)) * average_duration, 60)
                print('progress: {}/{} (t: {:.1f}, {:01d}:{:02}) (s: {:d}) (e: {:.02f})'
                    .format(i + 1, input_size, average_duration, int(m), int(s), success_sum, err),
                    end = '\r')
            # append results to table
            row.append(
                result_row(overlap, param, val, json_record, image_size)
            )
            # end of run output
            # calculate average duration
            avg_duration = reduce(lambda prev,cur: prev+cur['time'], json_record.values(), 0)
            # retrieve values from records for output
            vals = list(filter(lambda val: type(val) != str,
                map(lambda cur: cur['err'], json_record.values())
            ))
            # all inf
            if len(vals) == 0:
                min_err = np.inf
                max_err = np.inf
            # some inf
            elif len(vals) < input_size:
                min_err = min(vals)
                max_err = np.inf
            # no inf
            else:
                min_err = min(vals)
                max_err = max(vals)
            print('{}: {:0.2f}, t: {:0.2f}, s: {}/{}, min: {:0.2f}, max: {:0.2f}'
                .format(param, val, avg_duration, success_sum, input_size,
                    min_err, max_err))
        # perform evaluations on a single image
        else:
            duration, err, min_err, max_err = eval_method(image_name, method, **kw)
            print('{}: {:0.2f}, t: {}, err: {}'.format(param, val, duration, err))
            record = {
                'err': err,
                'time': duration,
                'min': min_err,
                'max': max_err
            }
            row.append(result_row(overlap, param, val, record, image_size))
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
    if name.lower() == 'tn' or name.lower() == 'dhn':
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
                        action='store_true')
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
    for param in results.keys():
        # create output folder if needed
        folder_name = 'results'
        outname = os.path.join(folder_name, '%s_%s_%s' % \
                               (method, param, time.strftime('%d-%m_%H:%M')))
        # create output folder
        if kw['viz'] or kw['output']:
            make_results_folder(folder_name)

        # save csv
        if kw['output']:
            results[param].to_csv(outname + '.csv')
            print('csv output saved')

        # output visualization
        if kw['viz']:
            # plot with an error threshold of 1
            plot_results(outname, results[param], param, 1)
            print('results visualized and saved')

    print('done')

