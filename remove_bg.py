#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from libyasap.utils import read_img, save_img, disp_img

import cv2
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='remove background and linear rescale for deep sky images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', help='input image',
                        required=True)
    parser.add_argument('-o', '--output', help='output image',
                        required=True)
    parser.add_argument('--gaussian-frac', default=0.01, type=float,
                        help='Gaussian kernel size for row smoothing '
                        'relative to image height')
    parser.add_argument('--min-rank', default=0.05, type=float,
                        help='rank of minimal value to be selected')
    parser.add_argument('--disp-bg', action='store_true',
                        help='display inferred background image')
    args = parser.parse_args()

    img = read_img(args.input)
    assert img.ndim == 3 and img.shape[2] == 3
    print(f'image shape: {img.shape}')

    min_rank = int(round(args.min_rank * img.shape[1]))
    row_min = np.expand_dims(
        np.partition(img, min_rank, axis=1)[:, min_rank],
        1)
    ksize = int(args.gaussian_frac * img.shape[0])
    ksize += (ksize + 1) % 2
    row_min = cv2.GaussianBlur(row_min, (1, ksize), 0)
    if args.disp_bg:
        disp_img('input', img, wait=False)
        disp_img('bg', np.broadcast_to(row_min, img.shape))

    img -= row_min
    img = np.clip(img, 0, 1, out=img)
    vmin = img.min()
    vmax = img.max()
    img -= vmin
    img *= 1 / (vmax - vmin)
    save_img(img, args.output)
    print(f'bg mean: {np.mean(row_min):.3g}')
    print(f'rescale: {vmin:.3g} {vmax: .3g}')

if __name__ == '__main__':
    main()
