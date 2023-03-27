from .config import AlignmentConfig
from .star_point import StarPointRefiner
from .utils import (perspective_transform, in_bounding_box, avg_l2_dist,
                    find_homography, disp_img, logger, get_mask_for_largest,
                    disp_match_pairs, read_img, format_relative_aa_bbox)

import cv2
import numpy as np

class OpticalFlowRefiner:
    """refine alignment using off-the-shelf sparse optical flow"""
    _first_img_gray_8u: np.ndarray = None
    _first_ftr_points: np.ndarray
    # (n, 2) array of feature point locations on first image

    _parent = None

    def set_first_image(self, img_gray, img_gray_8u, parent):
        """:param parent: the owner ImageStackAlignment instance"""
        self._parent = parent
        self._first_img_gray_8u = img_gray_8u
        self._first_ftr_points = cv2.goodFeaturesToTrack(
            img_gray, maxCorners=300, qualityLevel=0.3, minDistance=20,
            blockSize=7, mask=parent._mask)

    def _get_matched_points(self, H, img_gray):
        """:return: (num orig points, p0 on first image, p1 on current image)"""
        assert img_gray.dtype == np.float32
        img_gray = (img_gray * 255).astype(np.uint8)

        p1_all, st, err = cv2.calcOpticalFlowPyrLK(
            self._first_img_gray_8u, img_gray,
            self._first_ftr_points, None, maxLevel=1, winSize=(12, 12))

        p0_nx2 = np.squeeze(self._first_ftr_points, 1)
        p1_nx2 = np.squeeze(p1_all, 1)

        mask = np.squeeze(st == 1, 1)
        assert mask.ndim == 1 and mask.shape[0] == p0_nx2.shape[0]

        # remove matches that are not on current image
        p0_on_p1 = perspective_transform(np.linalg.inv(H), p0_nx2)
        mask &= in_bounding_box(
            p0_on_p1, 0, 0, img_gray.shape[1], img_gray.shape[0])

        p0 = p0_nx2[mask]
        p1 = p1_nx2[mask]

        return p0_nx2.shape[0], p0, p1

    def get_trans(self, H, img_gray):
        """Given ``H`` which is the transform from ``img_gray`` to first image,
        compute the refined transform.

        :return: new H, err
        """
        warp_as_dst = cv2.warpPerspective(img_gray, H, self._parent._out_shape)
        nr_orig, p0, p1 = self._get_matched_points(H, warp_as_dst)
        Href, err, err_max = find_homography(p1, p0, 0)    # least squares

        logger.info(
            f'refine: filter_pts={nr_orig}->{len(p0)} '
            f'bbox={format_relative_aa_bbox(p0, img_gray.shape)}\n'
            f'  err:{avg_l2_dist(p0, p1):.3g}->{err:.3g} max={err_max:.3g} '
        )

        config = self._parent._config
        if err >= config.refine_abort_thresh:
            logger.warning(f'image discarded due to high error {err:.3g}')
            return

        if err >= config.align_err_disp_thresh:
            disp_match_pairs(
                self._first_img_gray_8u, p0,
                (warp_as_dst * 255).astype(np.uint8), p1)

        return (Href @ H), err


class ImageStackAlignment:
    _config: AlignmentConfig
    _refiner: OpticalFlowRefiner

    _out_shape: tuple[int] = None
    # (w, h) of image shape

    _detector = None
    _matcher = None
    _prev_trans = np.eye(3, dtype=np.float32)
    # homography from previous image to first image

    _prev_img: np.ndarray = None
    # only used for debug

    _prev_ftr: tuple = None

    _mask = None

    _is_first = True
    _first_img_gray_8u: np.ndarray

    _error_stat = None

    def __init__(self, config, refiner):
        self._config = config
        self._refiner = refiner
        self._detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=self._config.hessian_thresh)
        self._matcher = cv2.BFMatcher_create(cv2.NORM_L2, True)
        self._error_stat = []

    def set_mask_from_file(self, fpath: str):
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f'failed to read {fpath}'
        mask = np.zeros_like(img)
        mask[img >= 127] = 255
        self._mask = mask

    def feed_image_file(self, fpath: str):
        """feed an image to be aligned
        :return: (the aligned image, mask of valid values) or None if error too
            large
        """
        logger.info(f'processing {fpath}')
        img = read_img(fpath)
        h, w, _ = img.shape
        if self._out_shape is None:
            self._out_shape = (w, h)
        else:
            assert self._out_shape == (w, h)
        return self._handle_image(img)

    def _img_preproc(self, src: np.ndarray) -> np.ndarray:
        """preprocess an input gray image"""
        if not self._config.remove_bg:
            return src
        flat = src
        if self._mask is not None:
            flat = flat[self._mask > 127]
        thresh = np.quantile(flat.flatten(), self._config.remove_bg_thresh)
        src = np.maximum(src - thresh, 0)
        src /= src.max()
        return src

    def _handle_image(self, img):
        if self._config.use_identity_trans:
            return img, np.ones_like(img[:, :, 0], dtype=np.bool_)
        img_gray = self._img_preproc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        img_gray_8u = (img_gray * 255).astype(np.uint8)
        if not self._config.skip_coarse_align:
            prev_ftr = self._prev_ftr
            ftr = self._get_ftr_points(img_gray_8u)
            logger.debug(f'img size: {img.shape}; '
                         f'feature points: {len(ftr[0])}')
            self._prev_ftr = ftr

        if self._is_first:
            self._is_first = False
            self._refiner.set_first_image(img_gray, img_gray_8u, self)
            self._first_img_gray_8u = img_gray_8u
            return img, np.ones(img.shape[:2], dtype=np.bool_)

        # coarse align
        if self._config.skip_coarse_align:
            H = np.eye(3, dtype=np.float32)
        else:
            matches, p0xy, p1xy = self._compute_matches(
                'coarse', prev_ftr, ftr)
            # note: due to the rotation of camera plane, we do need perspective
            # transforms
            H, H_err_c, _ = find_homography(p1xy, p0xy, cv2.RANSAC)

            if self._config.draw_matches:
                print(H)
                if self._prev_img is None:
                    self._prev_img = self._first_img_gray_8u
                disp_img('matches', cv2.drawMatches(
                    self._prev_img, prev_ftr[0], img_gray_8u, ftr[0],
                    matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
                self._prev_img = img_gray_8u

            logger.debug(f'coarse: dist_aligned={H_err_c:.2g}')

        H = self._refiner.get_trans(self._prev_trans @ H, img_gray)

        if H is None:
            return

        H, err = H

        if err >= self._config.refine_abort_thresh:
            logger.warning(f'image discarded due to high error: {err:.3g}')
            return

        self._error_stat.append(err)

        if not self._config.skip_coarse_align:
            self._prev_ftr = ftr

        self._prev_trans = H
        newimg = cv2.warpPerspective(img, H, self._out_shape)
        mask = np.empty(img.shape[:2], dtype=np.uint8)
        mask[:] = 255
        mask = cv2.warpPerspective(mask, H, self._out_shape)
        mask = mask >= 127
        return newimg, mask

    def error_stat(self):
        """get error stats"""
        return np.array(self._error_stat, dtype=np.float32)

    def _get_ftr_points(self, img):
        """:return: ftr points, desc array as by opencv"""
        ftr, desc = self._detector.detectAndCompute(img, self._mask)
        k = self._config.max_ftr_points
        assert len(ftr), 'no feature points detected'
        if len(ftr) <= k:
            return ftr, desc

        mask = get_mask_for_largest(np.array([i.response for i in ftr]), k)
        assert type(ftr) is tuple
        ftr = tuple(np.array(ftr)[mask])
        desc = desc[mask]
        return ftr, desc

    def _compute_matches(self, log_name, ftr0, ftr1):
        """compute opencv keypoint matches with distance filtering

        :return: matches, keypoint xy of ftr0, keypoint xy of ftr1
        """
        maxdist = self._config.ftr_match_coord_dist * self._out_shape[0]
        kp0, desc0 = ftr0
        kp1, desc1 = ftr1

        assert len(kp0) and len(kp1), 'no key points detected'
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
        if np.count_nonzero(mask) < 4:
            raise ValueError(f'not enough matches: '
                             f'dist={dist/self._out_shape[0]} {mask=};'
                             ' consider increasing --ftr-match-coord-dist')
        matches = np.array(matches)[mask]
        matches = sorted(matches, key=lambda x: x.distance)
        logger.info(f'{log_name} matching: num={len(matches)}/{len(mask)} '
                    f'avg_dist={dist.mean():.2g} '
                    f'filtered_dist={dist[mask].mean():.2g}')
        assert len(matches) >= 4
        return matches, *select_kpxy()
