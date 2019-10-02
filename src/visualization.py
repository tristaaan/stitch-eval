import argparse
import glob
import math
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

def plot_results(fname, results, param, output_dir='.', image_size=512):
  '''
  visualize results in a seaborn annotated heatmap
  '''
  # if there is only the overlap param create a 1d plot.
  if param == 'overlap':
    plot_1d_results(fname, results, param, output_dir=output_dir)
    return

  reformatted = results.pivot('overlap', param, 'success')
  errors      = results.pivot('overlap', param, 'error')
  vmax = results[['total']].values[0,0] + 1
  sns.set()
  f, ax = plt.subplots(figsize=(9, 6))
  # define the color map, red to blue, no gray point
  # very similar to sns.color_palette("coolwarm_r", n_colors=vmax)
  spread = [
    (0, '#BE2F33'),
    (0.49, '#E6CEC2'),
    (0.5, '#C7D4E8'),
    (1, '#415DC9')
  ]
  cmap = clr.LinearSegmentedColormap.from_list('mmap', spread, N=vmax)
  sns.heatmap(reformatted, annot=errors, fmt='0.01f', \
              linewidths=.5, ax=ax, annot_kws={'rotation':40}, \
              cmap=cmap, cbar_kws={'label': 'success'}, \
              vmin=0, vmax=vmax)

  # center the ticks on each segment of the color bar
  # for some reason seaborn doesn't do this automatically
  cbar = ax.collections[0].colorbar
  tick_locs = np.arange(0,vmax,2) + 0.5
  cbar.set_ticks(tick_locs)
  cbar.set_ticklabels(np.arange(0,vmax,2))

  # save fig
  plt.savefig(path.join(output_dir, ('%s.png' % fname)))
  # close in case of later reuse.
  ax.cla()
  f.clf()
  plt.close(f)

def plot_1d_results(fname, results, param, output_dir='.'):
  # import pdb
  # pdb.set_trace()
  vals    = results[['error']].values
  sucs    = results[['success']].values
  x_marks = list(map(lambda x: math.floor(x * 100), results[['overlap']].values))
  # plt.plot(x_marks, vals)
  plt.plot(x_marks, sucs)
  plt.savefig(path.join(output_dir, ('%s.png' % fname)))
  plt.clf()
  plt.close()

def get_csvs_from_directory(directory):
    return glob.glob(path.join(directory, '*.csv'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Visualize csv results')
  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument('-file', '-f', help='input csv', type=str)
  input_group.add_argument('-dir', '-d', help='input directory', type=str)
  parser.add_argument('-o', help='output directory', type=str, default='.')
  parser.add_argument('-size', '-s', help='image size from the results to help' \
                                         +' determine the color scale', \
                                         type=int, default=512)
  args = parser.parse_args()
  kw = vars(args)

  if kw['file']:
    fname = kw['file']
    table = pd.read_csv(fname)
    outfile = path.basename(fname).split('.')[0]
    param = outfile.split('_')[-3]
    plot_results(outfile, table, param, image_size=kw['size'])
    print('visualization created')
  elif kw['dir']:
    directory = kw['dir']
    output_dir = kw['o']
    files = get_csvs_from_directory(directory)
    for f in files:
      table = pd.read_csv(f)
      outfile = path.basename(f).split('.')[0]
      param = outfile.split('_')[-3]
      print(f, param)
      plot_results(outfile, table, param,
                   image_size=kw['size'], output_dir=output_dir)
    print('%d visualization(s) created' % len(files))
