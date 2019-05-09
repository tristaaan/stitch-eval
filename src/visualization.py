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
    colors = ['ro', 'co', 'bo', 'yo']
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
        fid.transform(offset_x, offset_y, 0)
        for c in fid.corners:
            x,y = c
            a.plot(x,y, colors[i])
    for fid in truthy:
        for c in fid.corners:
            x,y = c
            a.plot(x,y, 'w*')
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


def plot_results(fname, results, param, image_size=512, needs_reindex=True):
  '''
  visualize results in a seaborn annotated heatmap
  '''
  if results.shape[0] == 1:
    plot_1d_results(fname, results, param)
    return

  if needs_reindex:
    results = reindex(results, param)
  reformatted = results.pivot('overlap', param, 'error')

  max_e = np.nanmax(results.error)
  success_start = image_size / 10 / max_e
  failure_end = success_start + 0.00001
  tick_count = 5
  tick = int(max_e / tick_count)
  if success_start < max_e:
    ticks = [success_start*max_e] + list(range(int(success_start*max_e + tick),\
                                         int(max_e), tick-1)) + [max_e]
  else:
    ticks = [0,max_e]

  sns.set()
  f, ax = plt.subplots(figsize=(9, 6))
  if success_start < 1:
    spread = [(0,    '#7EF249'),
              (success_start, '#3FCD00'),
              (failure_end,   '#FFF2F2'),
              (1,    '#F2494C')]
  else:
    spread = [(0,    '#7EF249'),
              (1,    '#3FCD00')]

  cmap = clr.LinearSegmentedColormap.from_list('custom blue', spread, N=max_e)
  sns.heatmap(reformatted, annot=True, fmt='.02f', cmap=cmap,  \
              linewidths=.5, ax=ax, annot_kws={'rotation':40}, \
              vmin=0, vmax=max_e,                 \
              cbar_kws={'ticks': ticks})
  plt.savefig('%s.png' % fname)


def plot_1d_results(fname, results, param):
  # import pdb
  # pdb.set_trace()
  vals    = list(map(lambda x: float(x.split(', ')[0].replace('(','')), \
                     results.loc[0].values[1:]))
  x_marks = list(map(lambda x: float(x.replace('%', '')), \
                     results.columns.values[1:]))
  print(vals, x_marks)
  plt.plot(x_marks, vals)
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

  plot_results(outfile, table, table.columns[2], \
               image_size=kw['size'], needs_reindex=False)
