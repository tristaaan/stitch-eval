import math
import argparse
import re
from os import path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns
import pandas as pd


def saveimfids(fname, im, fids, truthy=[]):
    '''
    plot fiducial markers on the stitched image
    '''
    fig, a = plt.subplots(1, 1, figsize=(4, 5))
    colors = ['r', 'tab:orange', 'c', 'm']
    a.imshow(im, cmap='gray')
    # offset fiducial markers for visualization
    offset_x, offset_y = (0,0)
    min_x = min(f.min_x() for f in fids)
    min_y = min(f.min_y() for f in fids)
    if min_x < 0:
        offset_x = -1 * min_x
    if min_y < 0:
        offset_y = -1 * min_y
    # draw them on the plot
    for i,fid in enumerate(fids):
        # fid.transform(offset_x, offset_y, 0)
        for c in fid.corners:
            x,y = c
            x += offset_x
            y += offset_y
            a.plot(x,y, 'o', color=colors[i])
    for i,fid in enumerate(truthy):
        for c in fid.corners:
            x,y = c
            x += offset_x
            y += offset_y
            a.plot(x,y, 'o', color=colors[i], mfc='none')
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    plt.savefig(fname)
    a.cla()
    fig.clf()
    plt.close(fig)


def reindex(table, param):
  '''
  change table layout for heatmap
  '''
  assert table.shape[0] > 1, 'Cannot reindex table with one row'

  ranges = table.columns
  num_ranges = list(map(lambda x: float(re.sub(r'[°(db)%]','', x)), ranges))
  flat = pd.DataFrame(columns=['overlap', param, 'error', 'time'])
  for index, row in table.iterrows():
    for i,v in enumerate(ranges):
      cell = row[i]
      err, duration = list(map(lambda c: float(re.sub(r'[\(\)s]', '', c)), \
                           cell.split(', ')))
      flat = flat.append({'overlap': index, param: num_ranges[i], \
                         'time': duration, 'error': err}, ignore_index=True)
  return flat


def plot_results(fname, results, param, image_size=512):
  '''
  visualize results in a seaborn annotated heatmap
  '''
  # if there is only the overlap param create a 1d plot.
  if len(results.columns) == 5:
    plot_1d_results(fname, results, param)
    return

  reformatted = results.pivot('overlap', param, 'success')
  errors      = results.pivot('overlap', param, 'error')
  vmax = max(results[['total']].values)[0]
  sns.set()
  f, ax = plt.subplots(figsize=(9, 6))
  sns.heatmap(reformatted, annot=errors, fmt='0.02f', \
              cmap=sns.color_palette("vlag_r", n_colors=vmax), \
              linewidths=.5, ax=ax, annot_kws={'rotation':40}, \
              vmin=0, vmax=vmax)
  plt.savefig('%s.png' % fname)


def plot_1d_results(fname, results, param):
  # import pdb
  # pdb.set_trace()
  vals    = results[['error']].values
  sucs    = results[['success']].values
  x_marks = list(map(lambda x: math.floor(x * 100), results[['overlap']].values))
  # plt.plot(x_marks, vals)
  plt.plot(x_marks, sucs)
  plt.savefig('%s.png' % fname)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Visualize csv results')
  parser.add_argument('-file', '-f', help='filename', type=str)
  parser.add_argument('-size', '-s', help='image size from the results to help' \
                                         +'determine the color scale', \
                                         type=int, default=512)
  args = parser.parse_args()
  kw = vars(args)

  fname = kw['file']
  table = pd.read_csv(fname)
  outfile = path.basename(fname).split('.')[0]

  plot_results(outfile, table, table.columns[2], image_size=kw['size'])
