# Comparison of Image Stitching Methods

Evaluation tool for image stitching methods. Some methods included:

- SURF, SIFT, AKAZE
- An area based method by [Öfverstedt et al.](https://ieeexplore.ieee.org/document/8643403).
- Frequency based method
- Learning based methods (see [`deep-homography`](https://github.com/tristaaan/stitch-eval/tree/deep_homography) branch and notebooks within).

This is the corresponding repository for the thesis project "[Implementation and Evaluation of Image Stitching Methods](http://thesixsides.com/misc/thesis.pdf)" (2020) by Tristan Wright

## Cloning

This repository contains a submodule for the [Öfverstedt method repository](https://github.com/MIDA-group/py_alpha_amd_release). If you would like to pull the code for it when pulling this repo use:

```
git clone --recursive https://github.com/tristaaan/stitch-eval
```

If you have already pulled you can update the submodule with:

```
git submodule update --init
```

## Installing

With Anaconda: 
```
conda create --name stitch-eval --file requirements.txt
```

Separately you'll need to install `imutils` with pip: 

```
pip install imutils # not sure why anaconda cannot find this.
```

Can install these libraries with `pip` too but much easier to manage with anaconda.

Libraries for machine learning methods not included as they're very system dependent. Safe to say one will need Keras and TensorFlow.

## Scripts

### `im_split.py`

```py
$ python im_split.py -h
usage: im_split.py [-h] [-noise NOISE] [-rotation ROTATION] [-overlap OVERLAP] [-blocks BLOCKS] [-downsample DOWNSAMPLE] [-file FILE]

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
$ python visualization.py -h
usage: visualization.py [-h] (-file FILE | -dir DIR) [-o O] [-threshold THRESHOLD] [-save_csvs]

Visualize csv results

optional arguments:
  -h, --help            show this help message and exit
  -file FILE, -f FILE   input csv
  -dir DIR, -d DIR      input directory
  -o O                  output directory
  -threshold THRESHOLD, -t THRESHOLD
                        error threshold
  -save_csvs, -s        save csv's with error and success columns
```
