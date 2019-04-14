import os
import sys

import numpy as np
import scipy as sp
import scipy.misc

from math import pi
from time import time
from imageio import imwrite
from util import crop_zeros, eq_paste

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
param_iterations = 500
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 0.1

def prepare_paste(im,x,y):
    '''
    pad an image for pasting
    '''
    x = -int(x) # coordinates amd_alpha gives are reversed?
    y = -int(y)
    if y < 0:
        vert = (0, abs(y))
    else:
        vert = (y, 0)

    if x < 0:
        horz = (0, abs(x))
    else:
        horz = (x, 0)

    return np.pad(im, (vert, horz), 'constant', constant_values=(0,0))

def stitch(ref_im, flo_im):
    # start timer
    start = time()
    ref_im_orig = ref_im.copy()
    flo_im_orig = flo_im.copy()

    # normalize
    ref_im = filters.normalize(ref_im, 0.0, None)
    flo_im = filters.normalize(flo_im, 0.0, None)

    diag = 0.5 * (transforms.image_diagonal(ref_im, spacing) +
                  transforms.image_diagonal(flo_im, spacing))

    weights1 = np.ones(ref_im.shape)
    mask1 = np.ones(ref_im.shape, 'bool')
    weights2 = np.ones(flo_im.shape)
    # mask2 = np.ones(flo_im.shape, 'bool')
    mask2 = flo_im != 0 # mask the 0 valued areas

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
    step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])

    # Create the transform and add it to the registration framework
    reg.add_initial_transform(Rigid2DTransform(), np.array([1.0/diag, 1.0, 1.0]))

    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(0.001)
    reg.set_sampling_fraction(param_sampling_fraction)
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer('adam')

    # Create output directory
    output_dir = '../data/tmp'
    directory = os.path.dirname(output_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize(output_dir)

    # Start the registration
    reg.run()

    # transform in the format theta, y, x
    (transform, value) = reg.get_output(0)
    theta,y,x = transform.get_params()
    print(theta, y,x)
    flo_im_orig = sp.ndimage.rotate(flo_im_orig, -(theta * 180) / pi)
    flo_im_orig = prepare_paste(crop_zeros(flo_im_orig, zero=100), x,y)
    return (eq_paste(ref_im_orig, flo_im_orig), time() - start)

def amd_alpha(blocks):
    A,B,C,D = blocks

    AB, t1 = stitch(A, B)
    CD, t2 = stitch(C, D)
    E,  t3 = stitch(AB, CD)

    E = crop_zeros(E, zero=100)
    imwrite('../data/stitched.tif', E)
    return (E, sum([t1,t2,t3]))
