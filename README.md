# Comparison of Image Stitching Methods

Comparison and evaluation tool for a variety of image stitching methods. These methods include: 

- Area/spatial based
- Frequency based
- Feature based
- Learning based

## Scripts

### im_split.py

```py
$ python im_split.py -h
usage: im_split.py [-h] [-noise NOISE] [-rotation ROTATION] [-overlap OVERLAP] [-blocks BLOCKS] [-file FILE]

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
  -file FILE, -f FILE   image filename
```

### evaluator.py

```
$ python evaluator.py -h
usage: evaluator.py [-h] [-noise] [-rotation] [-overlap] [-file FILE]

Run evaluations with a method

optional arguments:
  -h, --help           show this help message and exit
  -noise               run noise evaluations
  -rotation            run rotation evaluations
  -overlap             run overlap evaluations
  -file FILE, -f FILE  image filename
```