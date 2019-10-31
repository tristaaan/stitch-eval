import math
import numpy as np

def dist(p1, p2):
    L2 = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
    return math.sqrt(L2)

def average_corner_error(gt, est):
    '''
    total registration error between ground-truth and estimated
    returns the average error of the moving images and the max error among them.
    '''
    labels = ['A', 'B', 'C', 'D']
    g = dict(zip(labels, gt))
    e = dict(zip(labels, est))
    max_err = -9e9
    min_err = 9e9
    err = 0
    for l in labels[1:]: # the first points' error will always be 0
        diff = dist(g[l].tl, e[l].tl) + \
               dist(g[l].tr, e[l].tr) + \
               dist(g[l].br, e[l].br) + \
               dist(g[l].bl, e[l].bl)
        if diff < min_err:
            min_err = diff
        if diff > max_err:
            max_err = diff
        err += diff
    # The first reference image never moves.
    return err / 12, min_err, max_err
