import os
import sys

import numpy as np

from math import pi
from time import time
from imutils import rotate_bound

from util import crop_zeros, merge, tuple_sub, uint16_to_uint8

sys.path.insert(0, 'py_alpha_amd_release/')
import py_alpha_amd_release.filters as filters
import py_alpha_amd_release.transforms as transforms
from py_alpha_amd_release.register import Register as amd_alpha_register
from py_alpha_amd_release.transforms import Rigid2DTransform

# Registration Parameters
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

# The number of iterations
param_iterations = 100
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 0.01


def stitch(ref_im, flo_im, mask_ref=False):
    # start timer
    start = time()
    ref_im_orig = ref_im.copy()
    flo_im_orig = flo_im.copy()

    # masks
    if mask_ref:
        mask1 = ref_im != 0
    else:
        mask1 = np.ones(ref_im.shape, 'bool')
    mask2 = flo_im != 0

    ref_im = ref_im.astype('float32')
    flo_im = flo_im.astype('float32')

    # normalize
    ref_im = filters.normalize(ref_im, 0.1, mask1)
    flo_im = filters.normalize(flo_im, 0.1, mask2)

    diag = 0.5 * (transforms.image_diagonal(ref_im, spacing) +
                  transforms.image_diagonal(flo_im, spacing))

    # weights
    weights1 = np.ones(ref_im.shape)
    weights2 = np.ones(flo_im.shape)

    reg = amd_alpha_register(2)

    reg.set_report_freq(0)
    reg.set_alpha_levels(alpha_levels)

    reg.set_reference_image(ref_im)
    reg.set_reference_mask(mask1)
    reg.set_reference_weights(weights1)

    reg.set_floating_image(flo_im)
    reg.set_floating_mask(mask2)
    reg.set_floating_weights(weights2)

    # Setup the Gaussian pyramid resolution levels
    reg.add_pyramid_level(4, 5.0)
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...]
    step_lengths = np.array([[0.5, 0.5], [0.5, 0.5],[0.5, 0.01]])

    # Create the transform and add it to the registration framework
    reg.add_initial_transform(Rigid2DTransform(), np.array([0.15, 1.0, 1.0]))

    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(0.001)
    reg.set_sampling_fraction(param_sampling_fraction)
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer('adam')

    # Create output directory
    output_dir = '../data/'
    directory = os.path.dirname(output_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize(output_dir)

    # Start the registration
    reg.run()

    # transform in the format theta, y, x
    (transform, _) = reg.get_output(0)
    theta,y,x = transform.get_params()
    angle = theta * 180 / pi # convert to degrees
    # print(angle, y,x)
    (tx, ty) = (0,0)
    pre_size = flo_im_orig.shape
    flo_im_orig = rotate_bound(flo_im_orig, angle)
    post_size = flo_im_orig.shape
    flo_im_orig, (cy,_,cx,_) = crop_zeros(flo_im_orig, crop_vals=True)
    rot_diff = tuple_sub(post_size, pre_size)
    tx -= cx - rot_diff[1] // 2
    ty -= cy - rot_diff[0] // 2

    base = merge(ref_im_orig, flo_im_orig, -x, -y)
    return (base, [-x+tx,-y+ty,angle], time() - start)

def amd_alpha(blocks):
    np.random.seed(1234)
    A,B,C,D = blocks

    AB, M1, t1 = stitch(A, B)
    CD, M2, t2 = stitch(C, D, mask_ref=True)
    E,  M3, t3 = stitch(AB, CD)

    E = crop_zeros(E, zero=100)
    return (E, [M1, M2, M3], sum([t1,t2,t3]))
