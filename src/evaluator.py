import argparse
import math
import random
import numpy as np
import pandas as pd

from imageio import imread, imwrite

from im_split import im_split
from feature_based.AKAZE import AKAZE

def clamp_uint16(val):
  return min(val, (2**16)-1)

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


def run_eval(image, method, noise=False, rotation=False, overlap=False):
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


def evaluation(stitched, original):
  '''
  Sum absolute difference
  '''
  stitched, ow, oh = pad_image(stitched, original.shape)
  total_px = ow * oh
  differents = np.count_nonzero(original - stitched[:oh,:ow])

  # percent the same
  return (total_px - differents) / total_px


def evaluation2(stitched, original):
  '''
  https://user.engineering.uiowa.edu/~n-morph/research/invert_and_trans/node3.html#eq:aic_error
  It is important to define a ROI because the amount of padding applied to the image data can have
  a significant effect on the average error calculation. The ROI restricts the error measurements
  to areas of interest preventing the situation where the largest error occurs in the background
  of the image.
  '''
  stitched, ow, oh = pad_image(stitched, original.shape)
  roi_H, roi_W = (int(oh * 0.95), int(ow * 0.95))
  total_px = roi_H * roi_W
  differents = stitched[:roi_H,:roi_W] - original[:roi_H,:roi_W]
  nom = differents.sum() / total_px
  return nom

def random_evaluation(stitched, original):
  '''
  Take random windowed samples between the two images. Avoid the upper-left quad
  '''
  stitched, max_y, max_x = pad_image(stitched, original.shape)
  iters = 2000;
  diffs = 0
  for i in range(0,iters):
    y_quad = int(random.random() > 0.5)
    x_quad = int(random.random() > 0.5)
    while x_quad + y_quad == 0:
      y_quad = int(random.random() > 0.5)
      x_quad = int(random.random() > 0.5)
    r_y = random.randint((max_y / 2) * y_quad, max_y-1)
    r_x = random.randint((max_x / 2) * x_quad, max_x-1)
    diffs += abs(int(stitched[r_y, r_x]) - int(original[r_y, r_x]))
    diffs = clamp_uint16(diffs)

  return diffs / (iters * ((2**16)-1))

def i_RMSE(stitched, original):
  stitched, ow, oh = pad_image(stitched, original.shape)
  total_px = ow * oh
  stitched = stitched.astype('float64') / (2**16 -1)
  original = original.astype('float64') / (2**16 -1)
  abs_err = abs(original - stitched) ** 2
  return math.sqrt((1/total_px) * abs_err.sum())

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run evaluations with a method')
  parser.add_argument('-noise', action='store_true', help='run noise evaluations')
  parser.add_argument('-rotation', action='store_true', help='run rotation evaluations')
  parser.add_argument('-overlap', action='store_true', help='run overlap evaluations')
  args = parser.parse_args()
  kw = vars(args)
  x = run_eval('../data/S2_Img_003.00.tif', AKAZE, **kw)
  for t in x:
    print(t)