#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import argparse
import collections
import itertools
import logging
import os
import sys

import cv2
import numpy as np

logger = logging.getLogger('yasap')

class StreamingStack(metaclass=abc.ABCMeta):
    """base class for stacking a stream of alined images"""

    @abc.abstractmethod
    def add_img(self, img, mask):
        pass

    @abc.abstractmethod
    def get_result(self):
        pass

class MeanStreamingStack(StreamingStack):
    """stack by mean"""

    _sum_img = None
    _cnt_img = None

    def add_img(self, img, mask):
        assert (img.dtype == np.float32 and mask.dtype == np.bool_ and
                mask.shape == img.shape[:2] and
                img.ndim == 3 and img.shape[2] == 3), (img.shape, mask.shape)
        if self._sum_img is None:
            assert np.all(mask)
            self._sum_img = img.copy()
            self._cnt_img = np.zeros_like(img[:, :, 0])
        else:
            self._sum_img += img
        self._cnt_img += mask

    def get_result(self):
        return self._sum_img / np.expand_dims(self._cnt_img, 2)


class NoExtreamStreamingStack(MeanStreamingStack):
    """stack by mean while removing extreme values"""

    _max = None
    _min = None

    def add_img(self, img, mask):
        super().add_img(img, mask)
        if self._max is None:
            self._max = img.copy()
            self._min = img.copy()
        else:
            max1 = np.maximum(self._max, img)
            mask = np.broadcast_to(np.expand_dims(mask, 2), img.shape)
            self._max[mask] = max1[mask]
            min1 = np.minimum(self._min, img)
            self._min[mask] = min1[mask]

    def get_result(self):
        c = np.broadcast_to(np.expand_dims(self._cnt_img, 2),
                            self._sum_img.shape)
        ret = (self._sum_img - self._max - self._min) / np.maximum(c - 2, 1)
        ret[c == 2] = self._sum_img[c == 2] / 2
        ret[c == 1] = self._sum_img[c == 1]
        return ret


class AlignmentConfig:
    """default configuration options"""

    ftr_match_coord_dist = 0.01
    """maximal distance of coordinates between matched feature points, relative
    to image width"""

    draw_matches = False
    """weather to draw match points"""

    save_refined_dir = ''
    """directory to save refined images"""

    skip_coarse_align = False
    """weather to skip corse alignment process; useful for aligning the
    foreground (which is at roughly the same location on every image)"""


def perspective_transform(H: np.ndarray, pts: np.ndarray):
    # compute perspective transform and compare results with opencv to check
    # that my understanding is correct (not sure why they need input shape (1, n,
    # 2)
    r_cv = cv2.perspectiveTransform(pts[np.newaxis], H)[0]
    t = (H @ np.concatenate([pts, np.ones_like(pts[:, :1])], axis=1).T).T
    r = t[:, :2] / t[:, 2:]
    from IPython import embed
    assert np.allclose(r, r_cv), embed()
    return r

def avg_l2_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((x - y)**2).sum(axis=1)).mean())

def find_homography(src, dst, method):
    """:return: H, avg dist"""
    assert len(src) == len(dst) and len(src) >= 10
    H, _ = cv2.findHomography(src, dst, method)
    dist = avg_l2_dist(perspective_transform(H, src), dst)
    return H, dist

class ImageStackAlignment:
    _config: AlignmentConfig

    _out_shape: tuple[int] = None
    # (w, h) of image shape

    _detector = None
    _matcher = None
    _prev_trans = np.eye(3, dtype=np.float32)
    # homography from previous image to first image

    _first_img_path: str = None
    # used for exif info

    _first_img_gray_8u: np.ndarray = None
    _first_ftr_points: np.ndarray
    # (n, 2) array of feature point locations on first image

    _prev_img: np.ndarray = None
    # only used for debug

    _prev_ftr: tuple = None

    _mask = None

    def __init__(self, config=AlignmentConfig()):
        self._config = config
        self._detector = cv2.xfeatures2d.SURF_create(hessianThreshold=800)
        self._matcher = cv2.BFMatcher_create(cv2.NORM_L2, True)

    def set_mask_from_file(self, fpath: str):
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f'failed to read {fpath}'
        mask = np.zeros_like(img)
        mask[img >= 127] = 255
        self._mask = mask

    def feed_image_file(self, fpath: str):
        """feed an image to be aligned
        :return: the aligned image, mask of valid values
        """
        logger.info(f'processing {fpath}')
        if self._first_img_path is None:
            self._first_img_path = fpath
        img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        assert img is not None, f'failed to read {fpath}'
        img = img.astype(np.float32) / np.float32(np.iinfo(img.dtype).max)
        h, w, _ = img.shape
        if self._out_shape is None:
            self._out_shape = (w, h)
        else:
            assert self._out_shape == (w, h)
        return self._handle_image(img)

    def _handle_image(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_8u = (img_gray * 255).astype(np.uint8)
        if not self._config.skip_coarse_align:
            prev_ftr = self._prev_ftr
            ftr = self._detector.detectAndCompute(img_gray_8u, self._mask)
            logger.debug(f'feature points: {len(ftr[0])}')
            self._prev_ftr = ftr

        if self._first_img_gray_8u is None:
            self._first_img_gray_8u = img_gray_8u
            self._first_ftr_points = cv2.goodFeaturesToTrack(
                img_gray, maxCorners=300, qualityLevel=0.3, minDistance=20,
                blockSize=7, mask=self._mask)
            return img, np.ones(img.shape[:2], dtype=np.bool_)

        if self._config.skip_coarse_align:
            H = np.eye(3, dtype=np.float32)
            H_err_c = 0.0
        else:
            matches, p0xy, p1xy = self._compute_matches(
                'coarse', prev_ftr, ftr)
            # note: due to the rotation of camera plane, we do need perspective
            # transforms
            H, H_err_c = find_homography(p1xy, p0xy, cv2.RANSAC)

            if self._config.draw_matches:
                print(H)
                if self._prev_img is None:
                    self._prev_img = self._first_img_gray_8u
                self._disp_img('matches', cv2.drawMatches(
                    self._prev_img, prev_ftr[0], img_gray_8u, ftr[0],
                    matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
                self._prev_img = img_gray_8u

        H = self._prev_trans @ H
        warp_as_dst = cv2.warpPerspective(img_gray, H, self._out_shape)
        p1_all, st, err = cv2.calcOpticalFlowPyrLK(
            self._first_img_gray_8u, (warp_as_dst * 255).astype(np.uint8),
            self._first_ftr_points, None, maxLevel=3)
        mask = st == 1
        p0 = self._first_ftr_points[mask]
        p1 = p1_all[mask]
        Href, H_err_r = find_homography(p1, p0, 0)    # least squares
        H = Href @ H

        logger.debug(f'coarse: dist_aligned={H_err_c:.2g}; '
                     f'refine: pts={len(p0)}/{len(self._first_ftr_points)} '
                     f'dist_before={avg_l2_dist(p0, p1):.3g} '
                     f'dist_aligned={H_err_r:.3g} ')

        if not self._config.skip_coarse_align:
            self._prev_ftr = ftr
        self._prev_trans = H
        newimg = cv2.warpPerspective(img, H, self._out_shape)
        mask = np.empty(img.shape[:2], dtype=np.uint8)
        mask[:] = 255
        mask = cv2.warpPerspective(mask, H, self._out_shape)
        mask = mask >= 127
        return newimg, mask

    @classmethod
    def _disp_img(cls, title: str, img: np.ndarray, wait=True):
        """display an image"""
        scale = 1000 / max(img.shape)
        if scale < 1:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(title, img)
        if wait:
            if cv2.waitKey(-1) & 0xff == ord('q'):
                print('exit')
                sys.exit()

    def _compute_matches(self, log_name, ftr0, ftr1):
        """compute opencv keypoint matches with distance filtering

        :return: matches, keypoint xy of ftr0, keypoint xy of ftr1
        """
        maxdist = self._config.ftr_match_coord_dist * self._out_shape[0]
        kp0, desc0 = ftr0
        kp1, desc1 = ftr1

        assert len(kp0) and len(kp1)
        matches = self._matcher.match(desc0, desc1)

        def select_kpxy():
            kp0_xy = np.array([kp0[i.queryIdx].pt for i in matches],
                              dtype=np.float32)
            kp1_xy = np.array([kp1[i.trainIdx].pt for i in matches],
                              dtype=np.float32)
            return kp0_xy, kp1_xy

        kp0_xy, kp1_xy = select_kpxy()
        dist = np.sqrt(((kp0_xy - kp1_xy)**2).sum(axis=1))
        mask = dist <= maxdist
        matches = np.array(matches)[mask]
        matches = sorted(matches, key=lambda x: x.distance)
        logger.info(f'{log_name} matching: num={len(matches)}/{len(mask)} '
                    f'avg_dist={dist.mean():.2g} '
                    f'filtered_dist={dist[mask].mean():.2g}')
        assert len(matches) >= 10
        return matches, *select_kpxy()

    @classmethod
    def save_img(cls, img: np.ndarray, fpath: str):
        """save image file in 16bit formats"""
        assert img.dtype == np.float32
        img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        succ = cv2.imwrite(fpath, img)
        assert succ, f'failed to write image {fpath}'


STACKER_MAP = {
    'mean': MeanStreamingStack,
    'noext': NoExtreamStreamingStack,
}

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('imgs', nargs='+')
    parser.add_argument('-o', '--output', required=True, help='output file path')
    parser.add_argument('--mask',
                        help='mask on all images for the ROI of star region: '
                        'white for star, black for others')
    parser.add_argument('--stacker', choices=STACKER_MAP.keys(), default='noext',
                        help='stacking algorithm for aligned images')
    for i in dir(AlignmentConfig):
        if i.startswith('_'):
            continue
        v = getattr(AlignmentConfig, i)
        i = i.replace('_', '-')
        kw = dict(help='see AlignmentConfig class')
        if type(v) is bool:
            if v:
                i = 'no-' + i
                kw['action'] = 'store_false'
                kw['dest'] = i
            else:
                kw['action'] = 'store_true'
        else:
            kw['type'] = type(v)
            kw['default'] =v
        parser.add_argument(f'--{i}', **kw)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(suppress=True)

    config = AlignmentConfig()
    for i in dir(AlignmentConfig):
        if not i.startswith('_'):
            setattr(config, i, getattr(args, i))
    align = ImageStackAlignment(config)
    stacker = STACKER_MAP[args.stacker]()
    if args.mask:
        align.set_mask_from_file(args.mask)
    for i in args.imgs:
        img, mask = align.feed_image_file(i)
        stacker.add_img(img, mask)
    align.save_img(stacker.get_result(), args.output)

if __name__ == '__main__':
    main()
