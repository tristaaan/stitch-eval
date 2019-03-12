import argparse
import math
import random
import numpy as np
import pandas as pd

from imageio import imread, imwrite

from im_split import im_split
from AKAZE import AKAZE
from Fourier import Fourier
from util import clamp_uint16


def eval_method(image, original, method, **kwargs):
  blocks = im_split(image, **kwargs)
  stitched, duration = method(blocks)
  result = i_RMSE(stitched, original)
  return (duration, result)


def eval_param(image, method, param, data_range):
  cname = method.__name__
  table = pd.DataFrame(columns=[param, 'time', cname])
  orig = imread(image)
  print('%s evaluation with %s' % (param, method.__name__))
  for val in data_range:
    kw = { param: val }
    duration, result = eval_method(image, orig, method, **kw)
    print("param: %0.02f, t: %0.2f, a: %0.2f" % (val, duration, result*100))
    table = table.append(
      {param: val, 'time': duration, cname: result}, ignore_index=True)
  return table


def run_eval(image, method, noise=False, rotation=False, overlap=False, **kwargs):
  # if no arguments are provided all are run.
  if not noise and not rotation and not overlap:
    noise    = True
    rotation = True
    overlap  = True

  out = []
  if noise:
    out.append(eval_param(image, method, 'noise', [d / 100 for d in range(0,11,2)]))

  if rotation:
    out.append(eval_param(image, method, 'rotation', range(0,181,45)))

  if overlap:
    out.append(eval_param(image, method, 'overlap', [o/100 for o in range(5, 51, 5)]))

  return out


def pad_image(stitched, target_size):
  sw, sh = stitched.shape
  tw, th = target_size

  nw, nh = (0,0)
  if sw < tw:
      nw = tw - sw
  if sh < th:
      nh = th - sh

  if nw + nh > 0:
    return (np.pad(stitched, ((0, nw), (0, nh)), 'constant', constant_values=(0,0)), tw, th)
  return (stitched, tw, th)


def i_RMSE(stitched, original):
  stitched, ow, oh = pad_image(stitched, original.shape)
  total_px = ow * oh
  stitched = stitched.astype('float64') / (2**16 -1)
  original = original.astype('float64') / (2**16 -1)
  abs_err = abs(original - stitched) ** 2
  return math.sqrt((1/total_px) * abs_err.sum())


def method_picker(name):
  methods = [Fourier, AKAZE]
  method_names = list(map(lambda x: x.__name__.lower(), methods))
  return methods[method_names.index(name.lower())]


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run evaluations with a method')
  parser.add_argument('-noise', action='store_true', help='run noise evaluations')
  parser.add_argument('-rotation', action='store_true', help='run rotation evaluations')
  parser.add_argument('-overlap', action='store_true', help='run overlap evaluations')
  parser.add_argument('-file', '-f', help='image filename', type=str, default='../data/T1_Img_002.00.tif')
  parser.add_argument('-method', '-m', help='method to evaluate', type=str, default='akaze')

  args = parser.parse_args()
  kw = vars(args)
  kw['method'] = method_picker(kw['method'])
  x = run_eval(kw['file'], **kw)
  for t in x:
    print(t)