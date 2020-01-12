import argparse
import glob
import math
import re
import json
from os import path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd


def saveimfids(fname, im, fids, truthy=[]):
    '''
    plot fiducial markers on the stitched image
    '''
    fig, a = plt.subplots(1, 1, figsize=(4, 5))
    # should be very visible colors on any image
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
    lines = []
    # draw image markers
    for i,fid in enumerate(fids):
        for c in fid.corners:
            x,y = c
            x += offset_x
            y += offset_y
            a.plot(x,y, 'o', color=colors[i])
            lines.append([x,y])
    # draw outlines
    ls = ':'
    for i in range(0, len(lines), 4):
      pts = lines[i:i+4]
      color = colors[i // 4]
      # draw first to last point
      ax, ay = pts[0]
      bx, by = pts[-1]
      l = mlines.Line2D([ax,bx], [ay,by], color=color, ls=ls)
      a.add_line(l)
      # draw all other outlines
      for j in range(len(pts[:-1])):
        nextj = j + 1
        ax, ay = pts[j]
        bx, by = pts[nextj]
        l = mlines.Line2D([ax,bx], [ay,by], color=color, ls=ls)
        a.add_line(l)

    # draw ground truth markers
    for i,fid in enumerate(truthy):
        for c in fid.corners:
            x,y = c
            x += offset_x
            y += offset_y
            a.plot(x,y, 'o', color=colors[i], mfc='none', mew=1)
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    plt.savefig(fname)
    a.cla()
    fig.clf()
    plt.close(fig)


def prepare_table(df, threshold):
  '''
  read json results, append columns: 'success' and 'error'
  '''
  results = df[['results']].values
  error = []
  success = []
  vmax = 0
  for json_result in results:
    # correct json syntax so it can be loaded
    mstr = json_result[0].replace('\'', '"')
    result = json.loads(mstr)
    avg_err = 0
    avg_suc_err = 0
    suc = 0
    vmax = len(result)
    avg_err_size = vmax
    for record in result.values():
      max_err = float(record['max']) / 4 # average this
      # if the max error is under the threshold record the error
      err = float(record['err'])
      if max_err != 'inf' and max_err <= threshold:
        avg_suc_err += err
        suc += 1

      # record non-infinite error here if zero successes
      if err != np.inf:
        avg_err += err
      else:
        avg_err_size -= 1
    # record average successful error
    if suc > 0:
      error.append('%0.02f' % (avg_suc_err / suc))
    # otherwise record the minimal max-error
    else:
      # all errors were inf
      if avg_err_size == 0:
        error.append('inf')
      # record average failed error
      else:
        error.append('%0.02f' % (avg_err / avg_err_size))
    success.append(suc)
  # create new df with new columns
  cols = pd.DataFrame({'error': error, 'success': success})
  df = df.reset_index()
  # concatenate the csv and the new columns
  return pd.concat([df, cols], axis=1), vmax


def plot_results(fname, results, param, threshold, output_dir='.', savecsv=False):
  '''
  visualize results in a seaborn annotated heatmap
  '''
  # if there is only the overlap param create a 1d plot.
  if param == 'overlap':
    plot_1d_results(fname, results, param, threshold,
                    output_dir=output_dir, savecsv=savecsv)
    return

  parsed_results, vmax = prepare_table(results, threshold)
  if savecsv:
    parsed_results.to_csv(path.join(output_dir, fname + '_parsed.csv'))
  successes = parsed_results.pivot('overlap', param, 'success')
  successes = (successes / vmax) * 100
  errors    = parsed_results.pivot('overlap', param, 'error')

  sns.set()
  f, ax = plt.subplots(figsize=(9, 6))
  # define the color map, red to blue, no gray point
  # very similar to sns.color_palette("coolwarm_r", n_colors=vmax)
  spread = [
    (0, '#bfbfbf'), # gray, no successes
    (0.01, '#BE2F33'), # worst red
    (0.49, '#E6CEC2'), # best red (success lt 50%)
    (0.5, '#C7D4E8'), # worst blue (success gte 50%)
    (1, '#415DC9') # best blue
  ]
  vmax += 1
  pmax = 101

  # segmented colorbar with vmax bins
  cmap = clr.LinearSegmentedColormap.from_list('mmap', spread, N=vmax)

  # plot heatmap
  sns.heatmap(successes, annot=errors, fmt='s', \
              linewidths=.5, ax=ax, annot_kws={'rotation':40}, \
              cmap=cmap, cbar_kws={'label': 'success'}, \
              vmin=0, vmax=pmax)

  # center the ticks on each segment of the color bar
  # for some reason seaborn doesn't do this automatically
  cbar = ax.collections[0].colorbar
  tick_locs = np.arange(0,pmax,25) + 0.5
  cbar.set_ticks(tick_locs)
  cbar.set_ticklabels(list(map(lambda x: '%.0f%%' % x, np.arange(0,pmax,25))))

  # save fig
  plt.savefig(path.join(output_dir, ('%s.png' % fname)))
  # close in case of later reuse.
  ax.cla()
  f.clf()
  plt.close(f)

def plot_1d_results(fname, results, param, threshold, output_dir='.', savecsv=False):
  parsed_results, _ = prepare_table(results, threshold)
  if savecsv:
    parsed_results.to_csv(path.join(output_dir, fname + '_parsed.csv'))
  vals    = parsed_results[['error']].values
  sucs    = parsed_results[['success']].values
  names=['%d%%' % s for s in range(10,101,10)]
  x_marks = list(map(lambda x: math.floor(x * 100), parsed_results[['overlap']].values))

  fig, ax = plt.subplots()
  ax.plot(x_marks, sucs)
  ax.set_xticks(x_marks)
  ax.set_xticklabels(names)
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
  parser.add_argument('-threshold', '-t', help='error threshold', type=float, default=1.0)
  parser.add_argument('-save_csvs', '-s', help='save csv\'s with error and success columns',
                      action='store_true', default=False)

  args = parser.parse_args()
  kw = vars(args)
  params = re.compile(r'(overlap|noise|rotation)')

  if kw['file']:
    fname = kw['file']
    table = pd.read_csv(fname)
    outfile = path.basename(fname).split('.')[0]
    param = params.findall(outfile)[0]
    plot_results(outfile, table, param, kw['threshold'], savecsv=kw['save_csvs'])
    print('visualization created')
  elif kw['dir']:
    directory = kw['dir']
    output_dir = kw['o']
    files = get_csvs_from_directory(directory)
    for f in files:
      table = pd.read_csv(f)
      outfile = path.basename(f).split('.')[0]
      param = params.findall(outfile)[0]
      print(f, param)
      plot_results(outfile, table, param, kw['threshold'], output_dir=output_dir, savecsv=kw['save_csvs'])
    print('%d visualization(s) created' % len(files))
