# YASAP: Yet Anothe Star Alignment Program

This script uses SURF to align images and the iterative Lucas-Kanade method to
finetune the alignment. The code and help message should be self-explanatory. It
processes the images in a streaming manner, meaning that the memory usage does
not depend on the number of input images.

Dependencies: opencv, numpy
