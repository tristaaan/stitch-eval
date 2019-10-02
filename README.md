# Comparison of Image Stitching Methods

Comparison and evaluation tool for a variety of image stitching methods. These methods include:

- Area/spatial based
- Frequency based
- Feature based
- Learning based

## Cloning

This repository contains a submodule for the [Öfverstedt method repository](https://github.com/MIDA-group/py_alpha_amd_release). If you would like to pull the code for it when pulling this repo use:

```
git clone --recursive https://github.com/tristaaan/stitch-eval
```

If you have already pulled you can update the submodule with:

```
git submodule update --init
```

## Scripts

### `im_split.py`

```py
$ python im_split.py -h
usage: im_split.py [-h] [-noise NOISE] [-rotation ROTATION] [-overlap OVERLAP]
                   [-blocks BLOCKS] [-downsample DOWNSAMPLE] [-file FILE]

Split an image into blockswith some overlap, rotation,and/or noise

optional arguments:
  -h, --help            show this help message and exit
  -noise NOISE, -n NOISE
                        add noise
  -rotation ROTATION, -r ROTATION
                        add rotation
  -overlap OVERLAP, -o OVERLAP
                        overlap percentage [0...1]
  -blocks BLOCKS, -b BLOCKS
                        number of blocks, must be a square number
  -downsample DOWNSAMPLE, -ds DOWNSAMPLE
                        downsample output images
  -file FILE, -f FILE   image filename
```

### `evaluator.py`

```
$ python evaluator.py -h
usage: evaluator.py [-h] [-noise] [-rotation] [-overlap] [-o_range O_RANGE]
                    [-r_range R_RANGE] (-file FILE | -dir DIR)
                    [-method METHOD] [-downsample DOWNSAMPLE] [-output]
                    [-debug] [-record_all] [-tex] [-viz]

Run evaluations with a method

optional arguments:
  -h, --help            show this help message and exit
  -noise                run noise evaluations
  -rotation             run rotation evaluations
  -overlap              run overlap evaluations
  -o_range O_RANGE      range of overlap start:end:stride
  -r_range R_RANGE      range of rotation, start:end:stride, for negative
                        values use -r_range="..."
  -file FILE, -f FILE   image filename
  -dir DIR, -d DIR      run evaluations on all images in a directory
  -method METHOD, -m METHOD
                        method to evaluate
  -downsample DOWNSAMPLE, -ds DOWNSAMPLE
                        downsample images
  -output, -o           output results to csv
  -debug                write the stitched image after each stitch
  -record_all, -r       record all errors and average them regardless if they
                        are under the error threshold
  -tex                  output results to LaTeX table
  -viz, -vis            create a heatmap of the results
```

### `visualization.py`

```
$ python visualization.py  -h
usage: visualization.py [-h] (-file FILE | -dir DIR) [-o O] [-size SIZE]

Visualize csv results

optional arguments:
  -h, --help           show this help message and exit
  -file FILE, -f FILE  input filename
  -dir DIR, -d DIR     input directory
  -o O                 output directory
  -size SIZE, -s SIZE  image size from the results to help determine the color
                       scale
```