from .utils import (logger, disp_img, precise_quantile, get_mask_for_largest,
                    perspective_transform, find_homography, disp_match_pairs,
                    in_bounding_box, format_relative_aa_bbox, avg_l2_dist)
from .refiner import RefinerBase

import typing

import pyximport
pyximport.install()
from .star_point_algo import find_star_centers, get_match_mask

import cv2
import numpy as np

if typing.TYPE_CHECKING:
    from .alignment import ImageStackAlignment
else:
    ImageStackAlignment = 'ImageStackAlignment'

class StarPointRefiner(RefinerBase):
    """refine using a custom star point detector"""
    # the result seems to be very similar to OpticalFlowRefiner

    _parent: ImageStackAlignment
    _first_img_gray_8u: np.ndarray
    _first_pt: np.ndarray   # star points on the first image
    _first_quality: float

    _knn: "cv2.ml.KNearest"

    def _draw_points(self, pt):
        img = np.zeros_like(self._first_img_gray_8u)
        pt = pt.astype(np.uint32)
        img[pt[:, 1], pt[:, 0]] = 255
        dilation = 9
        ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (dilation * 2 + 1, ) * 2)
        img = cv2.dilate(img, ele)
        return img

    def _get_star_coords_and_score(self, img_gray) -> tuple[np.ndarray, float]:
        """get coordinates of star points on this image, and a quality score of
        this image"""
        config = self._parent._config
        thresh = precise_quantile(img_gray, config.star_point_quantile)
        img_bin = img_gray >= thresh
        if (mask := self._parent._mask) is not None:
            assert mask.dtype == np.uint8
            img_bin *= (mask >= 1)

        star_pt, circleness_score = find_star_centers(
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

        return star_pt, circleness_score

    def set_first_image(self, img_gray, img_gray_8u, parent):
        """:param parent: the owner ImageStackAlignment instance"""
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        self._first_pt, self._first_quality = self._get_star_coords_and_score(
            img_gray)
        logger.info(f'Star point first image: num_pt={len(self._first_pt)}'
                    f' quality={self._first_quality:.2f}')
        self._knn = cv2.ml.KNearest_create()
        pt = self._first_pt.astype(np.float32)
        self._knn.train(pt, cv2.ml.ROW_SAMPLE,
                        np.arange(pt.shape[0], dtype=np.int32))

    def get_trans(self, H, img_gray):
        config = self._parent._config
        src_pt, quality_score = self._get_star_coords_and_score(img_gray)
        if (quality_score <
                self._first_quality - config.star_point_quality_thresh):
            logger.warning(f'discard due to low quality: {quality_score:.2f}'
                           f' (ref: {self._first_quality:.2f})')
            return

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

            if iter_ == 0:
                err_init = avg_l2_dist(src_pt_H[mask], H_dst)

            H, err, err_max = find_homography(H_src, H_dst, cv2.RANSAC)

            if err < config.star_point_icp_stop_err or np.allclose(H, prev_H):
                break

            prev_H = H

            if err >= config.align_err_disp_thresh or config.star_point_disp:
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

        logger.info(f'star point ICP: iter={iter_+1} '
                    f'selected={mask.sum()}({mask.sum()/len(src_pt)*100:.1f}%) '
                    f'bbox={format_relative_aa_bbox(H_dst, img_gray.shape)}\n'
                    f'  err:{err_init:.3g}->{err:.3g},{err_max:.3g};'
                    f' quality={quality_score:.2f}'
                    )

        return H, err
