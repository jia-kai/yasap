from .utils import (perspective_transform, in_bounding_box, med_l2_dist,
                    find_homography, logger, disp_match_pairs,
                    format_relative_aa_bbox, F64Arr, F32Arr, U8Arr)
from .config import AlignmentConfig

import numpy as np
import cv2

import typing

if typing.TYPE_CHECKING:
    from .alignment import ImageStackAlignment

class RefinerBase:
    def set_first_image(self, img_gray: F32Arr, img_gray_8u: U8Arr,
                        parent: 'ImageStackAlignment'):
        _ = img_gray
        _ = img_gray_8u
        _ = parent
        raise NotImplementedError()

    def get_trans(self, H: F64Arr, img_gray: F32Arr
                  ) -> typing.Optional[tuple[F64Arr, float]]:
        """compute the refined transform

        :param H: ``(3, 3)`` array for the coarse transform from ``img_gray``
            (the current image) to first image
        :param img_mask: optional star-region mask on the first image
        :return: new H, err or None if match failed
        """
        _ = H
        _ = img_gray
        raise NotImplementedError()


class OpticalFlowRefinerBase(RefinerBase):
    _first_img_gray_8u: np.ndarray
    _parent: typing.Any
    _homography_method: int

    def _run_opt_flow(self, img_gray) -> tuple[int, np.ndarray, np.ndarray]:
        """:return: number of pointer before quality filtering,
            points on first image, points on ``img_gray``"""
        _ = img_gray
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

    @typing.override
    def get_trans(self, H: F64Arr, img_gray: F32Arr):
        warp_as_dst = cv2.warpPerspective(img_gray, H, self._parent._out_shape)
        assert warp_as_dst.dtype == np.float32
        warp_as_dst = (warp_as_dst * 255).astype(np.uint8)
        nr_orig, p0, p1 = self._get_matched_points(H, warp_as_dst)
        Href, err, err_max = find_homography(p1, p0, self._homography_method)

        logger.info(
            f'refine: filter_pts={nr_orig}->{len(p0)} '
            f'bbox={format_relative_aa_bbox(p0, tuple(img_gray.shape))}\n'
            f'  err:{med_l2_dist(p0, p1):.3g}->{err:.3g} {err_max=:.3g} '
        )

        config = self._parent._config

        if err >= config.align_err_disp_thresh:
            print(p1 - p0)
            print(Href)
            disp_match_pairs(
                self._first_img_gray_8u, p0, warp_as_dst, p1)

        return (Href @ H), err


class SparseOpticalFlowRefiner(OpticalFlowRefinerBase):
    """refine alignment using off-the-shelf sparse optical flow"""

    _homography_method = cv2.RANSAC

    _first_ftr_points: np.ndarray
    # (n, 2) array of feature point locations on first image

    @typing.override
    def set_first_image(self, img_gray: F32Arr, img_gray_8u: U8Arr,
                        parent: 'ImageStackAlignment'):
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        config: AlignmentConfig = parent._config
        self._first_ftr_points = cv2.goodFeaturesToTrack(
            img_gray, maxCorners=300,
            qualityLevel=config.sparse_opt_quality_level,
            minDistance=20,
            blockSize=config.sparse_opt_block_size,
            mask=parent._mask)

        nr_ftr = self._first_ftr_points.shape[0]
        logger.info(f'sparse opt init: {nr_ftr=}')
        assert nr_ftr >= 10

    def _run_opt_flow(self, img_gray) -> tuple[int, np.ndarray, np.ndarray]:
        config: AlignmentConfig = self._parent._config
        w = config.sparse_opt_win_size
        p1_all, st, _ = cv2.calcOpticalFlowPyrLK(
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
    _bin_mask: U8Arr

    @typing.override
    def set_first_image(self, img_gray: F32Arr, img_gray_8u: U8Arr,
                        parent: 'ImageStackAlignment'):
        _ = img_gray
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        assert parent._mask is not None, 'mask must be provided'
        self._mask_xy = np.ascontiguousarray(np.argwhere(parent._mask > 127)[
            :, ::-1])
        self._bin_mask = (parent._mask > 127).astype(np.uint8)

    def _run_opt_flow(self, img_gray) -> tuple[int, np.ndarray, np.ndarray]:
        """:return: (num orig points, p0 on first image, p1 on current image)"""
        mask_l = self._mask_xy.min(axis=0)
        mask_h = (self._mask_xy.max(axis=0) + 1)
        mask_ext = (mask_h - mask_l) * 0.1
        mx0, my0 = (mask_l - mask_ext).astype(np.int32).tolist()
        mx1, my1 = (mask_h + mask_ext).astype(np.int32).tolist()

        of_img0 = self._first_img_gray_8u
        of_img1 = img_gray
        assert of_img0.dtype == of_img1.dtype == np.uint8

        of_img0 = of_img0[my0:my1, mx0:mx1]
        of_img1 = of_img1[my0:my1, mx0:mx1]

        flow = cv2.calcOpticalFlowFarneback(
            of_img0, of_img1, None,
            pyr_scale=.5, levels=1, winsize=7,
            iterations=5, poly_n=5, poly_sigma=1.1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        px0_xy = self._mask_xy
        N = px0_xy.shape[0]
        x0, y0 = px0_xy.T
        subf = flow[y0 - my0, x0 - mx0].reshape(N, 2)
        px0_xy = px0_xy.astype(np.float32)
        return N, px0_xy, px0_xy + subf
