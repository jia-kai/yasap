from .utils import (perspective_transform, in_bounding_box, avg_l2_dist,
                    find_homography, logger, disp_match_pairs,
                    format_relative_aa_bbox, precise_quantile, disp_img)
from .config import AlignmentConfig

import numpy as np
import cv2

import typing

class RefinerBase:
    def set_first_image(self, img_gray, img_gray_8u, parent):
        """:param parent: the owner ImageStackAlignment instance"""
        raise NotImplementedError()

    def get_trans(self, H, img_gray) -> typing.Optional[
            tuple[np.ndarray, float]]:
        """Given ``H`` which is the transform from ``img_gray`` to first image,
        compute the refined transform.

        :return: new H, err or None if match failed
        """
        raise NotImplementedError()


class OpticalFlowRefinerBase(RefinerBase):
    _first_img_gray_8u: np.ndarray
    _parent: typing.Any
    _homography_method: int

    def _run_opt_flow(self, img_gray) -> tuple[int, np.ndarray, np.ndarray]:
        """:return: number of pointer before quality filtering,
            points on first image, points on ``img_gray``"""
        raise NotImplementedError()

    def _get_matched_points(self, H, img_gray):
        """
        :param H: from ``img_gray`` to first image
        :param img_gray: current image in uint8, with ``H`` applied
        :return: (num orig points, p0 on first image, p1 on current image)"""
        nr_orig, p0_nx2, p1_nx2 = self._run_opt_flow(img_gray)
        assert (p0_nx2.shape == p1_nx2.shape and p0_nx2.ndim == 2
                and p0_nx2.shape[1] == 2)

        # remove matches that are not on current image
        p0_on_p1 = perspective_transform(np.linalg.inv(H), p0_nx2)
        mask = in_bounding_box(
            p0_on_p1, 0, 0, img_gray.shape[1], img_gray.shape[0])

        p0 = p0_nx2[mask]
        p1 = p1_nx2[mask]
        return nr_orig, p0, p1

    def get_trans(self, H, img_gray):
        warp_as_dst = cv2.warpPerspective(img_gray, H, self._parent._out_shape)
        assert warp_as_dst.dtype == np.float32
        warp_as_dst = (warp_as_dst * 255).astype(np.uint8)
        nr_orig, p0, p1 = self._get_matched_points(H, warp_as_dst)
        Href, err, err_max = find_homography(p1, p0, self._homography_method)

        logger.info(
            f'refine: filter_pts={nr_orig}->{len(p0)} '
            f'bbox={format_relative_aa_bbox(p0, img_gray.shape)}\n'
            f'  err:{avg_l2_dist(p0, p1):.3g}->{err:.3g} {err_max=:.3g} '
        )

        config = self._parent._config

        if err >= config.align_err_disp_thresh:
            print(p1 - p0)
            print(Href)
            disp_match_pairs(
                self._first_img_gray_8u, p0, warp_as_dst, p1)

        if err >= config.refine_abort_thresh:
            logger.warning(f'image discarded due to high error {err:.3g}')
            return

        return (Href @ H), err


class SparseOpticalFlowRefiner(OpticalFlowRefinerBase):
    """refine alignment using off-the-shelf sparse optical flow"""

    _homography_method = 0  # least square

    _first_ftr_points: np.ndarray
    # (n, 2) array of feature point locations on first image

    def set_first_image(self, img_gray, img_gray_8u, parent):
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        config: AlignmentConfig = parent._config
        self._first_ftr_points = cv2.goodFeaturesToTrack(
            img_gray, maxCorners=300,
            qualityLevel=config.sparse_opt_quality_level,
            minDistance=20,
            blockSize=config.sparse_opt_block_size,
            mask=parent._mask)

    def _run_opt_flow(self, img_gray) -> tuple[int, np.ndarray, np.ndarray]:
        config: AlignmentConfig = self._parent._config
        w = config.sparse_opt_win_size
        p1_all, st, err = cv2.calcOpticalFlowPyrLK(
            self._first_img_gray_8u, img_gray,
            self._first_ftr_points, None,
            maxLevel=config.sparse_opt_levels,
            winSize=(w, w))

        p0_nx2 = np.squeeze(self._first_ftr_points, 1)
        p1_nx2 = np.squeeze(p1_all, 1)

        mask = np.squeeze(st == 1, 1)
        assert mask.ndim == 1 and mask.shape[0] == p0_nx2.shape[0]
        return p0_nx2.shape[0], p0_nx2[mask], p1_nx2[mask]


class DenseOpticalFlowRefiner(OpticalFlowRefinerBase):
    """refine alignment using dense optical flow in the masked region"""

    _homography_method = cv2.LMEDS

    _mask_xy: np.ndarray

    def set_first_image(self, img_gray, img_gray_8u, parent):
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        assert parent._mask is not None, 'mask must be provided'
        self._mask_xy = np.ascontiguousarray(np.argwhere(parent._mask > 127)[
            :, ::-1])

    def _run_opt_flow(self, img_gray) -> tuple[int, np.ndarray, np.ndarray]:
        """:return: (num orig points, p0 on first image, p1 on current image)"""
        flow = cv2.calcOpticalFlowFarneback(
            self._first_img_gray_8u, img_gray, None,
            pyr_scale=.5, levels=1, winsize=7,
            iterations=5, poly_n=5, poly_sigma=1.1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        px0_xy = self._mask_xy
        N = px0_xy.shape[0]
        x0, y0 = px0_xy.T
        subf = flow[y0, x0].reshape(N, 2)
        px0_xy = px0_xy.astype(np.float32)
        return N, px0_xy, px0_xy + subf
