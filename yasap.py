#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from libyasap.alignment import ImageStackAlignment
from libyasap.refiner import SparseOpticalFlowRefiner, DenseOpticalFlowRefiner
from libyasap.star_point import StarPointRefiner
from libyasap.config import AlignmentConfig
from libyasap.stacker import StackerBase, STACKER_DICT
from libyasap.utils import (setup_logger, logger, save_img, disp_img,
                            set_use_rigid)
from libyasap.postprocess import run_postprocess

import numpy as np
import cv2

import argparse
import gc

REFINER_MAP = {
    'opts': SparseOpticalFlowRefiner,
    'optd': DenseOpticalFlowRefiner,
    'star': StarPointRefiner,
}

def visualize(aligned, cur_result, **others):
    if aligned is None:
        aligned = (np.zeros_like(cur_result),
                   np.zeros_like(cur_result, dtype=bool))
    def imshow(x, title):
        disp_img(title, x, wait=False, max_size=500)

    for k, v in others.items():
        imshow(v, k)

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
    parser.add_argument('imgs', nargs='+',
                        help='image files; use @fname to read filenames '
                        'from a list')
    parser.add_argument('--postprocess',
                        help='run a postprocess script; requires one input and '
                        'one output; the script one main(img)->img function')
    parser.add_argument('-o', '--output', required=True,
                        help='output file path that supports u16 (usually '
                        '.tif file); if .npy file is used, the original fp32 '
                        'mat will be saved')
    parser.add_argument('--mask',
                        help='mask on all images for the ROI of star region: '
                        'white for star, black for others')
    parser.add_argument('--refiner', default='opts',
                        choices=list(REFINER_MAP.keys()),
                        help='choose the refiner algorithm; `star` might be '
                        'better for deep sky imaging. See code for more info')
    parser.add_argument('--stacker', default='mean',
                        choices=list(STACKER_DICT.keys()),
                        help='choose the stacker algorithm')
    parser.add_argument('--only-stack', action='store_true',
                        help='only do the stack part, assuming images have '
                        'been aligned')
    parser.add_argument('--use-rigid-transform', action='store_true',
                        help='use 4-DOF rigid transform instead of 8-DOF '
                        'homography')
    parser.add_argument('--log', help='also write log to file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='visualize internal results')
    AlignmentConfig.add_to_parser(parser)
    StackerBase.Config.add_to_parser(parser)
    for i in STACKER_DICT.values():
        i.Config.add_to_parser(parser)

    args = parser.parse_args()
    setup_logger(args.log)

    if args.postprocess:
        assert len(args.imgs) == 1, 'only one input can be provided'
        return run_postprocess(args.postprocess, args.imgs[0], args.output)

    if args.use_rigid_transform:
        set_use_rigid(True)

    if len(args.imgs) == 1 and args.imgs[0].startswith('@'):
        with open(args.imgs[0][1:]) as fin:
            args.imgs = [i.strip() for i in fin]
    else:
        args.imgs = sorted(args.imgs)

    align = ImageStackAlignment(AlignmentConfig().update_from_args(args),
                                REFINER_MAP[args.refiner]())
    if args.mask:
        align.set_mask_from_file(args.mask)
    stacker_cls = STACKER_DICT[args.stacker]
    stacker: StackerBase = stacker_cls(
        stacker_cls.Config().update_from_args(args)
    )
    stacker.set_config(StackerBase.Config().update_from_args(args))
    discard_list = []
    for idx, path in enumerate(args.imgs):
        logger.info('working on '
                    f'{idx}({idx-len(discard_list)})/{len(args.imgs)}: {path}')
        gc.collect()
        if args.only_stack:
            img = align.read_img(path)
            stacker.add_img(img, np.ones_like(img[:, :, 0], dtype=bool))
            if args.verbose:
                disp_img('current', img, wait=False)
                disp_img('result', stacker.get_preview_result(), wait=False)
                cv2.waitKey(1)
            continue

        aligned = align.feed_image_file(path)
        roi_mask = align.get_roi_mask()
        if aligned is None:
            discard_list.append(path)
        else:
            if not stacker.add_img(*aligned, roi_mask=roi_mask):
                discard_list.append(path)
        if args.verbose:
            visualize(aligned, stacker.get_preview_result(),
                      preproc=align.prev_preproc_img)
    save_img(stacker.get_result(), args.output)
    logger.info(f'discarded images: {len(discard_list)} {discard_list}')
    err = align.error_stat()
    if err.size:
        logger.info(f'err: mean={np.mean(err):.3g} max={np.max(err):.3g}')

if __name__ == '__main__':
    main()
