# YASAP: Yet Anothe Star Alignment Program

This script uses SURF to align images and the iterative Lucas-Kanade sparse
optical flow to finetune the alignment. It assumes multiple shots of the same
target with similar views and stacks these shots to denoise.

Dependencies: opencv, numpy, cython

## An example workflow

1. Shoot the milky way using high ISO and fast shutter to get pinpoint stars.
2. Convert the raw files to TIFF images in Darktable. Enable lens correction if
   possible.
3. Create star mask (i.e., a large area that contains most of the sky for
   alignment) using GIMP.
4. Run YASAP to merge the stack:

       ./yasap.py ../stack1/*  -o ../stack1.tif --mask ../stack1.mask.png \
            --rm-min=1 --rm-max=1

       # assuming the camera is stable and thus the images are already aligned,
       # we compute a denoised foreground from the image stack
       ./yasap.py ../stack1/*  -o ../stack1-fg.tif --use-identity \
            --rm-min=1 --rm-max=1
5. Merge the two images in GIMP.

Note: use `--rm-max` and `--rm-min` to remove outliers including cloud and
airplane/satellite trails.


## Details & notes

* YASAP processes the images in a streaming manner and maintains a working
  memory bounded by constant size.
* YASAP uses a numerically stable algorithm to compute the mean image.
* The input images should be given in time order if no tracking equipment was
  used during shooting. Each image is first matched with the previous image
  assuming a small displacement. The transformation is then refined against the
  first image.
* Perhaps counterintuitively at first glance, the correct transformation is
  homography (eight degrees of freedom) instead of affine transformation (six
  degrees of freedom) even if we assume an ideal pinhole camera model and
  infinitely far pinpoint stars (unless you have a spherical sensor), especially
  for wide-angle photos.
* Sometimes using star point alignment (instead of optical flow) for refinement
  gives better results. This mode was added recently, and I am not sure if it is
  always better than optical flow. Enable it by `--refiner star`.
* Use `./remove_bg.py` to remove sky background.
