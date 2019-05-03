import math
import numpy as np

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
