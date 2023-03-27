#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from libyasap.alignment import ImageStackAlignment, OpticalFlowRefiner
from libyasap.star_point import StarPointRefiner
from libyasap.config import AlignmentConfig
from libyasap.stacker import StreamingStacker
from libyasap.utils import setup_logger, logger, save_img, disp_img

import numpy as np
import cv2

import argparse
import gc

REFINER_MAP = {
    'opt': OpticalFlowRefiner,
    'star': StarPointRefiner,
}

def visualize(aligned, cur_result):
    if aligned is None:
        aligned = (np.zeros_like(cur_result),
                   np.zeros_like(cur_result, dtype=bool))
    def imshow(x, title):
        img = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        disp_img(title, img, wait=False, max_size=500)

    r, m = aligned
    r = r.copy()
    r[~m] *= 0.8
    imshow(r, 'aligned')
    imshow(cur_result, 'result')
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('imgs', nargs='+')
    parser.add_argument('-o', '--output', required=True,
                        help='output file path that supports u16 (usually '
                        '.tif file); if .npy file is used, the original fp32 '
                        'mat will be saved')
    parser.add_argument('--mask',
                        help='mask on all images for the ROI of star region: '
                        'white for star, black for others')
    parser.add_argument('--rm-min', type=int, default=0,
                        help='number of extreme min values (i.e., bottom n '
                        'values) to be removed; require working memory '
                        'proportional to this value')
    parser.add_argument('--rm-max', type=int, default=0,
                        help='number of extreme max values to be removed; see '
                        'also --rm-min')
    parser.add_argument('--refiner', default='opt', choices=REFINER_MAP.keys(),
                        help='choose the refiner algorithm; `star` might be '
                        'better for deep sky imaging. See code for more info')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize internal results')
    parser.add_argument('--log', help='also write log to file')
    AlignmentConfig.add_to_parser(parser)

    args = parser.parse_args()

    setup_logger(args.log)
    np.set_printoptions(suppress=True)

    config = AlignmentConfig().update_from_args(args)

    align = ImageStackAlignment(config, REFINER_MAP[args.refiner]())
    if args.mask:
        align.set_mask_from_file(args.mask)
    stacker = StreamingStacker(args.rm_min, args.rm_max)
    discard_list = []
    for idx, path in enumerate(args.imgs):
        logger.info(f'working on {idx}/{len(args.imgs)}: {path}')
        aligned = align.feed_image_file(path)
        if aligned is None:
            discard_list.append(path)
        else:
            stacker.add_img(*aligned)
        if args.visualize:
            visualize(aligned, stacker.get_preview_result())
        gc.collect()
    save_img(stacker.get_result(), args.output)
    logger.info(f'discarded images: {len(discard_list)} {discard_list}')
    err = align.error_stat()
    if err.size:
        logger.info(f'err: mean={np.mean(err):.3g} max={np.max(err):.3g}')

if __name__ == '__main__':
    main()
