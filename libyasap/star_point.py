from .utils import (logger, disp_img, precise_quantile, get_mask_for_largest,
                    perspective_transform, find_homography, disp_match_pairs,
                    in_bounding_box)
from .config import AlignmentConfig

import pyximport
pyximport.install()
from .star_point_algo import find_star_centers, get_match_mask

import cv2
import numpy as np

class StarPointRefiner:
    """refine using a custom star point detector"""
    # the result seems to be very similar to OpticalFlowRefiner

    _parent = None
    _first_img_gray_8u = None
    _first_pt = None
    _knn = None

    def _draw_points(self, pt):
        img = np.zeros_like(self._first_img_gray_8u)
        pt = pt.astype(np.uint32)
        img[pt[:, 1], pt[:, 0]] = 255
        dilation = 9
        ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (dilation * 2 + 1, ) * 2)
        img = cv2.dilate(img, ele)
        return img

    def _get_star_coords(self, img_gray):
        config: AlignmentConfig = self._parent._config
        thresh = precise_quantile(img_gray, config.star_point_quantile)
        img_bin = img_gray >= thresh
        if (mask := self._parent._mask) is not None:
            assert mask.min() >= 0 and mask.max() == 1
            img_bin *= mask

        star_pt = find_star_centers(
            img_gray * img_bin, config.star_point_min_area,
            config.star_point_min_bbox_ratio)

        oring_star_pt = len(star_pt)

        if len(star_pt) > (n := config.star_point_max_num):
            mask = get_mask_for_largest(star_pt[:, 0], n)
            star_pt = star_pt[mask]
        star_pt = star_pt[:, 1:]

        logger.debug(f'{thresh=:.2f} num_star={oring_star_pt}->{len(star_pt)}')

        if config.star_point_disp:
            disp_img('img', img_gray, wait=False)
            disp_img('star_blob', img_bin, wait=False)
            disp_img('star_pt', self._draw_points(star_pt))

        return star_pt

    def set_first_image(self, img_gray, img_gray_8u, parent):
        """:param parent: the owner ImageStackAlignment instance"""
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        self._first_pt = pt = self._get_star_coords(img_gray)
        self._knn = cv2.ml.KNearest_create()
        self._knn.train(pt, cv2.ml.ROW_SAMPLE,
                        np.arange(pt.shape[0], dtype=np.int32))

    def get_trans(self, H, img_gray):
        config: AlignmentConfig = self._parent._config
        src_pt = self._get_star_coords(img_gray)
        dst_pt = self._first_pt

        # iterative closest points

        prev_H = H
        for iter_ in range(config.star_point_icp_max_iters):
            src_pt_H = perspective_transform(H, src_pt)
            src_pt_H_mask = in_bounding_box(
                src_pt_H, 0, 0,
                self._first_img_gray_8u.shape[1],
                self._first_img_gray_8u.shape[0],
            )
            src_pt_H = src_pt_H[src_pt_H_mask]

            _, match_idx, _, dist = self._knn.findNearest(src_pt_H, 1)
            assert match_idx.ndim == 2 and dist.ndim == 2
            match_idx = np.squeeze(match_idx, 1).astype(np.int32)
            dist = np.squeeze(dist, 1)
            mask = get_match_mask(len(dst_pt), config, match_idx, dist)

            H_src = src_pt[src_pt_H_mask][mask]
            H_dst = dst_pt[match_idx[mask]]

            if len(H_src) <= 5:
                logger.warning(f'discard due to too few matches: {dist[mask]}')
                return

            H, err = find_homography(H_src, H_dst, 0)

            if err < config.star_point_icp_stop_err or np.allclose(H, prev_H):
                break

            prev_H = H

            if err >= config.align_err_disp_threh:
                logger.info(
                    f'iter {iter_}: {err=:.3g} '
                    f'selected={mask.sum()}({mask.sum()/len(src_pt):.2f})'
                )
                img_src = self._draw_points(src_pt_H)
                img_dst = self._draw_points(dst_pt)
                disp_img('src_orig', self._first_img_gray_8u, wait=False)
                disp_img('dst_orig', img_gray, wait=False)
                disp_img('src', img_src, wait=False)
                disp_img('dst', img_dst, wait=False)
                disp_match_pairs(
                    np.zeros_like(img_dst), H_dst,
                    np.zeros_like(img_src), H_src)
                disp_match_pairs(
                    self._first_img_gray_8u, H_dst,
                    (img_gray * 255).astype(np.uint8), H_src)

        logger.info(f'star point ICP: iter={iter_+1} {err=:.3g} '
                    f'selected={mask.sum()}({mask.sum()/len(src_pt)*100:.1f}%)')

        return H, err
