import argparse
import copy
import gc
import math
import os

import numpy as np
import pandas as pd

from im_split import im_split
from Fiducials import Fiducial_corners, group_transform, \
                      group_transform_affine, zero_group
from visualization import saveimfids, reindex, plot_results

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

def eval_method(image_name, method, debug=False, **kwargs):
    '''
    Evaluate image stitching given an image, a method, and the splitting params
    '''
    blocks, ground_truth, initial = im_split(image_name, fiducials=True, **kwargs)
    stitched, transforms, duration = method(blocks)

    # there was a catastrophic failure
    if duration == None:
        return (0, np.NAN, False)
    # with the transformations, construct estimated fiducials
    est_fiducials = build_fiducials(initial, transforms, \
                                    affine=hasattr(transforms[0],'shape'))
    # compare the estimated ones with the ground_truth ones.
    acc_result = fiducial_point_error(ground_truth, est_fiducials)
    suc_result = acc_result < 100
    # if debug, write the stitched image.
    if debug:
        print(transforms[0], transforms[1], transforms[2])
        fname = '../data/tmp/%s_%d_%.02f.tif' % \
                (method.__name__, int(kwargs['overlap']*100), acc_result)
        saveimfids(fname, stitched, copy.deepcopy(est_fiducials))#, truthy=ground_truth)
        gc.collect() # cleans up matplot lib junk
    return (duration, acc_result, suc_result)


def eval_param(image_name, method, param, data_range, overlap=0.2,
               downsample=False, debug=False):
    '''
    Run a study on a method given a single range of parameters with overlap
    '''
    row = []
    if param != 'overlap':
        print('%s: %s, overlap: %0.2f' % (method.__name__, param, overlap))
    else:
        print('%s: %s, overlap: %0.2f' % (method.__name__, param, data_range[0]))

    for val in data_range:
        kw = { param: val, 'downsample': downsample, 'debug': debug }
        # print(kw)
        if param != 'overlap':
            kw['overlap'] = overlap
        duration, err, suc = eval_method(image_name, method, **kw)
        print("%s: %0.2f, t: %0.2f, err: %0.2f suc: %0.2f" % (param, val, duration, err, suc))
        row.append('(%.02f, %0.02fs)' % (err, duration))
    return row


def run_eval(image_name, method, noise=False, rotation=False, overlap=False, \
             downsample=False, **kwargs):
    '''
    Run a study on a method with a variety of parameters:
    overlap + [noise, rotation]
    '''
    # if no arguments are provided all are run.
    if not noise and not rotation and not overlap:
        noise    = True
        rotation = True
        overlap  = True

    # overlap_range = [o/100 for o in range(20, 81, 10)]
    overlap_range = [0.60, 0.75]

    out = {}
    kw = {'downsample': downsample, 'debug': kwargs['debug']}
    if noise:
        noise_range = [d / 1000 for d in range(0,51,10)]
        table = []
        for o in overlap_range:
            kw['overlap'] = o
            table.append(eval_param(image_name, method, 'noise', noise_range, **kw))

        df = pd.DataFrame(table, columns=noise_range, index=overlap_range)
        df.index.names = ['overlap']
        df.columns.names = ['noise']
        df.columns = map(lambda x: '%.03f' % (x), noise_range)
        out['noise'] = df

    if rotation:
        rot_range = range(-60,61,10)
        # rot_range = [-15, 0, 15]

        table = []
        for o in overlap_range:
            kw['overlap'] = o
            row = eval_param(image_name, method, 'rotation', rot_range, **kw)
            table.append(row)
        df = pd.DataFrame(table, columns=rot_range, index=overlap_range)
        df.index.names = ['overlap']
        df.columns.names = ['rotation']
        df.columns = map(lambda x: '%d°' % (x), rot_range)
        out['rotation'] = df

    if overlap:
        table = eval_param(image_name, method, 'overlap', overlap_range, **kw)
        df = pd.DataFrame(table).transpose()
        df.index.names = ['overlap']
        df.columns = map(lambda x: '%.0f%%' % (x*100), overlap_range)
        out['overlap'] = df

    return out

def dist(p1, p2):
    L2 = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
    return math.sqrt(L2)

def edge_distances(fgroup):
    '''
    given a fiducial group calculate the distances between corresponding corners
    '''
    labels = ['A', 'B', 'C', 'D']
    g = dict(zip(labels, fgroup))
    ab1 = dist(g['A'].tr, g['B'].tl)
    ab2 = dist(g['A'].br, g['B'].bl)

    ac1 = dist(g['A'].bl, g['C'].tl)
    ac2 = dist(g['A'].br, g['C'].tr)

    cd1 = dist(g['C'].tr, g['D'].tl)
    cd2 = dist(g['C'].br, g['D'].bl)

    bd1 = dist(g['B'].bl, g['D'].tl)
    bd2 = dist(g['B'].br, g['D'].tr)
    return [ab1, ab2, ac1, ac2, cd1, cd2, bd1, bd2]

def fiducial_edge_error(gt, est):
    '''
    Compare edge distances, rotation invariant
    '''
    gt_dist = edge_distances(gt)
    est_dist = edge_distances(est)
    return abs(np.subtract(gt_dist, est_dist)).sum()

def fiducial_point_error(gt, est):
    '''
    total registration error between ground-truth and estimated
    '''
    labels = ['A', 'B', 'C', 'D']
    g = dict(zip(labels, gt))
    e = dict(zip(labels, est))
    err = 0
    for l in labels[1:]: # the first points' error will always be 0
        diff = dist(g[l].tl, e[l].tl) + \
            dist(g[l].tr, e[l].tr) + \
            dist(g[l].br, e[l].br) + \
            dist(g[l].bl, e[l].bl)
        err += diff
    return err # / 12 uncomment for average error.

def method_picker(name):
    from AKAZE import AKAZE
    from amd_alpha import amd_alpha
    from Direct import iterative_ssd, iterative_ncc, iterative_mi
    from Fourier import Frequency

    methods = [AKAZE,
        amd_alpha,
        Frequency,
        iterative_ssd, iterative_ncc, iterative_mi
    ]
    method_names = list(map(lambda x: x.__name__.lower(), methods))
    return methods[method_names.index(name.lower())]


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

    # other options
    parser.add_argument('-file', '-f',       help='image filename', type=str,
                        default='../data/T1_Img_002.00.tif')
    parser.add_argument('-method', '-m',     help='method to evaluate', type=str, default='akaze')
    parser.add_argument('-downsample', '-ds', help='downsample images', type=int, default='-1')
    parser.add_argument('-output', '-o', action='store_true', help='output results to csv')
    parser.add_argument('-debug', action='store_true', help='write the stitched image after each stitch')
    parser.add_argument('-tex', action='store_true', help='output results to LaTeX table')
    parser.add_argument('-viz', '-vis', action='store_true', help='create a heatmap of the results')

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

        # create output folder if needed
        folder_name = 'results'
        outname = os.path.join(folder_name, '%s_%s' % (method, k))
        if kw['viz'] or kw['tex'] or kw['output']:
            make_results_folder(folder_name)

        # output visualization
        if kw['viz']:
            plot_results(outname, results[k], k)

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

        if kw['output']:
            df = reindex(results[k], k)
            df.to_csv(outname + '.csv')

    print('done')

