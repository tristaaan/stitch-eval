import argparse
import glob
import math
import re
import json
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


def prepare_table(df, threshold):
  '''
  read json results, append columns: success and error
  '''
  results = df[['results']].values
  error = []
  success = []
  vmax = 0
  for json_result in results:
    mstr = json_result[0].replace('\'', '"')
    result = json.loads(mstr)
    avg_err = 0
    suc = 0
    vmax = len(result)
    for record in result.values():
      max_err = record['max']
      if max_err != 'inf' and max_err <= threshold:
        avg_err += record['err']
        suc += 1
    error.append(avg_err / len(result))
    success.append(suc)
  cols = pd.DataFrame({'error': error, 'success': success})
  return pd.concat([df, cols], axis=1), vmax


def plot_results(fname, results, param, threshold, output_dir='.'):
  '''
  visualize results in a seaborn annotated heatmap
  '''
  # if there is only the overlap param create a 1d plot.
  if param == 'overlap':
    plot_1d_results(fname, results, param, threshold, output_dir=output_dir)
    return

  parsed_results, vmax = prepare_table(results, threshold)
  successes = parsed_results.pivot('overlap', param, 'success')
  errors    = parsed_results.pivot('overlap', param, 'error')
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
  sns.heatmap(successes, annot=errors, fmt='0.02f', \
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

def plot_1d_results(fname, results, param, threshold, output_dir='.'):
  parsed_results, _ = prepare_table(results, threshold)
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

  args = parser.parse_args()
  kw = vars(args)
  params = re.compile(r'(overlap|noise|rotation)')

  if kw['file']:
    fname = kw['file']
    table = pd.read_csv(fname)
    outfile = path.basename(fname).split('.')[0]
    param = params.findall(outfile)[0]
    plot_results(outfile, table, param, kw['threshold'])
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
      plot_results(outfile, table, param, kw['threshold'], output_dir=output_dir)
    print('%d visualization(s) created' % len(files))
