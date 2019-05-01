import argparse
import re
from os import path

import matplotlib.pyplot as plt
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
  ranges = table.columns
  num_ranges = list(map(lambda x: float(re.sub(r'[°(db)]','', x)), ranges))
  flat = pd.DataFrame(columns=['overlap', param, 'error', 'time'])
  for index, row in table.iterrows():
    for i,v in enumerate(ranges):
      cell = row[i]
      err, duration = list(map(lambda x: float(re.sub(r'[\(\)s]', '', x)), \
                               cell.split(', ')))
      flat = flat.append({'overlap': index, param: num_ranges[i], \
                         'time': duration, 'error': err}, ignore_index=True)
  return flat

def plot_results(fname, results, param, needs_reindex=True):
  '''
  visualize results in a seaborn annotated heatmap
  '''
  if needs_reindex:
    results = reindex(results, param)
  reformatted = results.pivot('overlap', param, 'error')

  sns.set()
  f, ax = plt.subplots(figsize=(9, 6))
  cmap = sns.blend_palette(['green','red'], as_cmap=True)
  sns.heatmap(reformatted, annot=True, fmt='.02f', cmap=cmap, \
              linewidths=.5, ax=ax)
  plt.savefig('%s.png' % fname)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Visualize csv results')
  parser.add_argument('-file', '-f', help='filename', type=str)
  args = parser.parse_args()
  kw = vars(args)
  fname = kw['file']
  table = pd.read_csv(fname)
  outfile = path.basename(fname).split('.')[0]
  plot_results(outfile, table, table.columns[2], needs_reindex=False)

