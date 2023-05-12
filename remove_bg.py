#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from libyasap.utils import read_img, save_img, disp_img

import cv2
import numpy as np

import argparse

def work(img: np.ndarray, args) -> np.ndarray:
    img_orig = img.copy()

    img *= args.contrast
    if args.brightness:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img = lab[:, :, 0]

    min_rank = int(round(args.min_rank * img.shape[1]))
    row_min = np.expand_dims(
        np.partition(img, min_rank, axis=1)[:, min_rank],
        1)
    ksize = int(args.gaussian_frac * img.shape[0])
    ksize += (ksize + 1) % 2
    row_min = cv2.GaussianBlur(row_min, (1, ksize), 0)

    img -= row_min

    if args.brightness:
        lab[:, :, 0] = img
        img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        row_min = row_min[:, :, np.newaxis] / 100   # for verbose

    img = np.clip(img, 0, 1, out=img)
    vmin = img.min()
    vmax = img.max()
    img -= vmin
    img *= 1 / (vmax - vmin)

    if args.mask:
        mask = read_img(args.mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255
        elif mask.dtype == np.uint16:
            mask = mask.astype(np.float32) / 65535
        else:
            assert mask.dtype == np.float32, (
                f'unhandled mask dtype {mask.dtype}'
            )
        if mask.ndim == 2:
            mask = np.expand_dims(mask, 2)
        img = img * mask + img_orig * (1 - mask)

    if args.verbose:
        disp_img('bg', np.broadcast_to(row_min, img.shape), wait=False)
        disp_img('input', img_orig, wait=False)
        disp_img('output', img, wait=True)

    print(f'bg mean: {np.mean(row_min):.3g}')
    print(f'rescale: {vmin:.3g} {vmax: .3g}')
    return img

def main():
    parser = argparse.ArgumentParser(
        description='remove background and linear rescale for deep sky images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', help='input image',
                        required=True)
    parser.add_argument('-o', '--output', help='output image',
                        required=True)
    parser.add_argument(
        '--brightness', action='store_true',
        help='only remove the background on the brightness channel')
    parser.add_argument('--skip', action='store_true',
                        help='skip bg removal; useful as format converter')
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='adjust input image contrast')
    parser.add_argument('--gaussian-frac', default=0.02, type=float,
                        help='Gaussian kernel size for row smoothing '
                        'relative to image height')
    parser.add_argument('--min-rank', default=0.02, type=float,
                        help='rank of minimal value to be selected')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='display internal results')
    parser.add_argument('--roi', help='clip ROI: (x,y,w,h) in pixels or ratio')
    parser.add_argument('--mask', help='use another image as a mask for '
                        'applying bg removal')
    args = parser.parse_args()

    img = read_img(args.input)
    assert img.ndim == 3 and img.shape[2] == 3
    print(f'image shape: {img.shape}')

    if args.roi:
        x, y, w, h = map(float, args.roi.split(','))
        if max(x, y, w, h) < 1:
            H, W, _ = img.shape
            x, y, w, h = [int(round(i * s))
                          for i, s in zip([x, y, w, h], [W, H, W, H])]
        else:
            x, y, w, h = map(int, args.roi.split(','))
        img = img[y:y+h, x:x+w]

    if not args.skip:
        img = work(img, args)

    save_img(img, args.output)

if __name__ == '__main__':
    main()
