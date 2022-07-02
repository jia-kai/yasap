# YASAP: Yet Anothe Star Alignment Program

This script uses SURF to align images and the iterative Lucas-Kanade method to
finetune the alignment. The code and help message should be self-explanatory. It
processes the images in a streaming manner, meaning that the memory usage does
not depend on the number of input images.

Dependencies: opencv, numpy, cython

## An example workflow

1. Shoot the milky way using high ISO and fast shutter to get pinpoint stars.
2. Convert the raw files to TIFF images in Darktable.
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


## Notes

* Sometimes using star point alignment (instead of optical flow) for refinement
  gives better results. This mode was added recently, and I am not sure if it is
  always better than optical flow. Enable it by `--refiner star`.
* Use `./remove_bg.py` to remove sky background.
