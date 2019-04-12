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

### im_split.py

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

### evaluator.py

```
$ python evaluator.py -h
usage: evaluator.py [-h] [-noise] [-rotation] [-overlap] [-file FILE]
                    [-method METHOD] [-downsample DOWNSAMPLE] [-output]

Run evaluations with a method

optional arguments:
  -h, --help            show this help message and exit
  -noise                run noise evaluations
  -rotation             run rotation evaluations
  -overlap              run overlap evaluations
  -file FILE, -f FILE   image filename
  -method METHOD, -m METHOD
                        method to evaluate
  -downsample DOWNSAMPLE, -ds DOWNSAMPLE
                        downsample images
  -output, -o           output results to LaTeX table
```