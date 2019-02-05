import numpy as np
import pandas as pd

from time import time
from imageio import imread, imwrite

from im_split import im_split
from feature_based.AKAZE import AKAZE


def eval_method(image, original, method, **kwargs):
  blocks = im_split(image, **kwargs)
  start = time()
  stitched = method(blocks)
  duration = (time() - start) / (len(blocks) - 1)
  result = evaluation2(stitched, original)
  print("t: %0.2f, a: %0.2f" % (duration, result))
  return (duration, result)


def run_eval(image, method):
  cname = method.__name__
  orig = imread(image)
  noise_table = pd.DataFrame(columns=['time', cname])
  for db in [d / 100 for d in range(0,11,2)]:
    print('db: %f' % db)
    duration, result = eval_method(image, orig, method, noise=db)
    noise_table = noise_table.append({'time': duration, cname: result}, ignore_index=True)

  rot_table = pd.DataFrame(columns=['time', cname])
  for rot in range(0,181,45):
    print('rotation: %d' % rot)
    duration, result = eval_method(image, orig, method, rotation=rot)
    rot_table = noise_table.append({'time': duration, cname: result}, ignore_index=True)

  overlap_table = pd.DataFrame(columns=['time', cname])
  for p in [o/100 for o in range(0, 50, 5)]:
    print('overlap: %d' % p)
    duration, result = eval_method(image, orig, method, overlap=p)
    overlap_table = noise_table.append({'time': duration, cname: result}, ignore_index=True)

  return (noise_table, rot_table, overlap_table)


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

if __name__ == '__main__':
  x = run_eval('../data/S2_Img_003.00.tif', AKAZE)
  print(x[0])
  print(x[1])
  print(x[2])