from .utils import (logger, disp_img, precise_quantile, get_mask_for_largest,
                    perspective_transform, find_homography, disp_match_pairs,
                    in_bounding_box, format_relative_aa_bbox, avg_l2_dist,
                    F64Arr, U8Arr, F32Arr)
from .refiner import RefinerBase
from .config import AlignmentConfig

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

    _ref_img_gray_8u: np.ndarray
    _ref_pt: F32Arr   # star points on the first image
    _ref_quality: float
    _ref_to_first_trans: F64Arr = np.eye(3, dtype=np.float64)
    _ref_img_cnt: int = 0
    _img_cnt: int = 0

    _knn: "cv2.ml.KNearest"

    def _draw_points(self, pt):
        img = np.zeros_like(self._ref_img_gray_8u)
        pt = pt.astype(np.uint32)
        img[pt[:, 1], pt[:, 0]] = 255
        dilation = 9
        ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (dilation * 2 + 1, ) * 2)
        img = cv2.dilate(img, ele)
        return img

    @classmethod
    def find_star_centers(cls, img_gray: F32Arr, config: AlignmentConfig) -> F32Arr:
        """find the star centers
        :return: ``(N, 3)``-shaped array for quality, x, y"""
        return find_star_centers(img_gray, config.star_point_min_area,
                                 config.star_point_min_bbox_ratio)

    @classmethod
    def get_star_qual(cls, star_pt: F32Arr) -> float:
        """get quality score from :meth:`find_star_centers` results"""
        return float(np.quantile(star_pt[:, 0], 0.3))

    def _get_star_coords_and_score(
            self, H: F64Arr, img_gray: F32Arr) -> tuple[F32Arr, float]:
        """get coordinates of star points on this image, and a quality score of
        this image
        :param H: coarse transform from first image to ``img_gray``
        :return: (N, 2)-shaped star coordinates, overall star quality
        """
        config = self._parent._config

        if self._parent._mask is not None:
            cur_mask = cv2.warpPerspective(
                self._parent._mask, H, img_gray.shape[::-1],
                flags=cv2.INTER_LINEAR | cv2. WARP_INVERSE_MAP
            )
            img_gray = np.where(cur_mask > 127, img_gray, 0)

        thresh = precise_quantile(img_gray, config.star_point_quantile)
        img_bin = img_gray >= thresh
        img_gray = np.where(img_bin, img_gray, 0)

        star_pt = self.find_star_centers(img_gray, config)

        orig_nr_pt = len(star_pt)

        if len(star_pt) > (n := config.star_point_max_num):
            mask = get_mask_for_largest(star_pt[:, 0], n)
            star_pt = star_pt[mask]

        star_qual = self.get_star_qual(star_pt)
        star_pt = star_pt[:, 1:]

        logger.debug(f'pxl_thresh={thresh:.2f}'
                     f' num_star={orig_nr_pt}->{len(star_pt)}')

        if config.star_point_disp:
            disp_img('img', img_gray, wait=False)
            disp_img('star_thresh', img_bin, wait=False)
            disp_img('star_pt', self._draw_points(star_pt),
                     wait=not config.star_point_disp_no_wait)

        return star_pt, star_qual

    @typing.override
    def set_first_image(self, img_gray: F32Arr, img_gray_8u: U8Arr,
                        parent: 'ImageStackAlignment'):
        """:param parent: the owner ImageStackAlignment instance"""
        self._parent = parent
        self._ref_img_gray_8u = img_gray_8u
        self._ref_pt, self._ref_quality = self._get_star_coords_and_score(
            np.eye(3), img_gray)
        logger.info(f'Star point first image: num_pt={len(self._ref_pt)}'
                    f' quality={np.median(self._ref_quality):.2f}')
        self._init_knn()

    def _init_knn(self):
        self._knn = cv2.ml.KNearest_create()
        pt = self._ref_pt.astype(np.float32)
        self._knn.train(pt, cv2.ml.ROW_SAMPLE,
                        np.arange(pt.shape[0], dtype=np.int32))

    @typing.override
    def get_trans(self, H: F64Arr, img_gray: F32Arr
                  ) -> typing.Optional[tuple[F64Arr, float]]:
        self._img_cnt += 1
        config = self._parent._config
        src_pt, quality_score = self._get_star_coords_and_score(H, img_gray)
        dst_pt = self._ref_pt

        # iterative closest points

        next_break = False
        # compute H for this to ref
        H = np.linalg.inv(self._ref_to_first_trans) @ H
        prev_H = H
        for iter_ in range(config.star_point_icp_max_iters):
            src_pt_H = perspective_transform(H, src_pt)
            src_pt_H_mask = in_bounding_box(
                src_pt_H, 0, 0,
                self._ref_img_gray_8u.shape[1],
                self._ref_img_gray_8u.shape[0],
            )
            src_pt_H = src_pt_H[src_pt_H_mask]

            _, match_idx, _, dist = self._knn.findNearest(src_pt_H, 1)
            assert match_idx.ndim == 2 and dist.ndim == 2
            match_idx = np.squeeze(match_idx, 1).astype(np.int32)
            dist = np.squeeze(dist, 1)

            if next_break:
                max_dist_jump = 1.5
            else:
                max_dist_jump = config.star_point_icp_max_dist_jump

            mask = get_match_mask(len(dst_pt), max_dist_jump, match_idx, dist)

            H_src = src_pt[src_pt_H_mask][mask]
            H_dst = dst_pt[match_idx[mask]]

            if len(H_src) <= 5:
                logger.warning(f'discard due to too few matches: {dist[mask]}')
                return

            if iter_ == 0:
                err_init = avg_l2_dist(src_pt_H[mask], H_dst)

            H, err, err_max = find_homography(H_src, H_dst, cv2.RANSAC)

            if next_break:
                break

            if err < config.star_point_icp_stop_err or np.allclose(H, prev_H):
                next_break = True

            prev_H = H

            if err >= config.align_err_disp_thresh or config.star_point_disp:
                logger.info(
                    f'iter {iter_}: {err=:.3g} '
                    f'selected={mask.sum()}({mask.sum()/len(src_pt):.2f})'
                )
                img_src = self._draw_points(src_pt_H)
                img_dst = self._draw_points(dst_pt)
                disp_img('src_orig', self._ref_img_gray_8u, wait=False)
                disp_img('dst_orig', img_gray, wait=False)
                disp_img('src', img_src, wait=False)
                disp_img('dst', img_dst, wait=False)
                disp_match_pairs(
                    np.zeros_like(img_dst), H_dst,
                    np.zeros_like(img_src), H_src,
                    name='match_ptr', wait=False
                )
                disp_match_pairs(
                    self._ref_img_gray_8u, H_dst,
                    (img_gray * 255).astype(np.uint8), H_src,
                    name='match_img', wait=not config.star_point_disp_no_wait
                )

        logger.info(f'star point ICP: iter={iter_+1} '
                    f'selected={mask.sum()}({mask.sum()/len(src_pt)*100:.1f}%) '
                    f'bbox={format_relative_aa_bbox(H_dst, img_gray.shape)}\n'
                    f'  err:{err_init:.3g}->{err:.3g},{err_max:.3g};'
                    f' quality={quality_score:.2f}'
                    )

        H = self._ref_to_first_trans @ H

        if (quality_score <
                self._ref_quality - config.star_point_quality_max_drop):
            logger.warning(f'discard due to low quality: {quality_score:.2f}'
                           f' (ref: {self._ref_quality:.2f})')
            err = np.inf
        elif (ref_itval := config.star_point_refresh_ref_img) > 0:
            if (ref_itval + self._ref_img_cnt <= self._img_cnt and
                    err < config.star_point_refresh_ref_err_thresh):
                logger.info(f'reference image refreshed at {self._img_cnt}')
                self._ref_to_first_trans = H
                self._ref_img_cnt = self._img_cnt
                self._ref_img_gray_8u = (
                    np.clip(img_gray * 255, 0, 255).astype(np.uint8))
                self._ref_pt = src_pt
                self._init_knn()

        return H, err
