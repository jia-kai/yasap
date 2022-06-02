# YASAP: Yet Anothe Star Alignment Program

This script uses SURF to align images and the iterative Lucas-Kanade method to
finetune the alignment. The code and help message should be self-explanatory. It
processes the images in a streaming manner, meaning that the memory usage does
not depend on the number of input images.

Dependencies: opencv, numpy

## About the stacking algorithms

Use `--stacker {mean,noext}` to choose the stacking algorithm.

* `mean`: Arithmetic mean
* `noext` (the default): Arithmetic mean, excluding min/max extremes. It is
  useful for removing airplane/satellite trails.

## An example workflow

1. Shoot the milky way using high ISO and fast shutter to get pinpoint stars.
2. Convert the raw files to TIFF images in Darktable.
3. Create star mask and foreground mask using GIMP.
4. Run YASAP to merge the images:

       ./yasap.py ../stack1/*  -o ../stack1.tif --mask ../stack1.mask.png
       ./yasap.py ../stack1/*  -o ../stack1-fg.tif --mask ../stack1.fgmask.png
5. Merge the two images in GIMP.

[Here is the result](https://photos.app.goo.gl/HXME6QkkJPLXqaEd7).
