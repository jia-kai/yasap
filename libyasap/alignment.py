from .config import AlignmentConfig
from .star_point import StarPointRefiner
from .refiner import RefinerBase
from .utils import (find_homography, disp_img, logger, get_mask_for_largest,
                    read_img, img_as_u8, F64Arr, F32Arr, U8Arr)
from .rot_homo import FocalRotationHomographyEstimator
from . import utils

import cv2
import numpy as np
import numpy.typing as npt

from abc import abstractmethod, ABCMeta
from datetime import datetime, timedelta
import typing
import weakref

type KeyPointList = tuple['cv2.KeyPoint', ...]
type ImgFtrDesc = tuple[KeyPointList, F64Arr]
"""feature points and descriptors"""

class FeatureMatcher(metaclass=ABCMeta):
    @abstractmethod
    def set_first_image(self, img: F32Arr, img_gray_8u: U8Arr):
        pass

    @abstractmethod
    def get_matches(self, new_img: F32Arr) -> typing.Optional[
            tuple[F32Arr, F32Arr]]:
        """get ``(N, 2)`` for x, y of key points on first image, and key points
        on current image"""


class FeatureMatcherCV2(FeatureMatcher):
    _parent: 'ImageStackAlignment'

    def __init__(self, parent: 'ImageStackAlignment'):
        self._parent = weakref.proxy(parent)
        self._detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=self._parent._config.hessian_thresh,
            extended=True,
            nOctaves=5,
        )
        self._matcher = cv2.BFMatcher_create(cv2.NORM_L2, True)

    def set_first_image(self, img: F32Arr, img_gray_8u: U8Arr):
        _ = img
        self._first_ftr = self._get_ftr_points(img_gray_8u)

    def get_matches(self, new_img: F32Arr) -> typing.Optional[
            tuple[F32Arr, F32Arr]]:
        _ = new_img
        new_img_gray_8u = img_as_u8(new_img)
        new_ftr = self._get_ftr_points(new_img_gray_8u)
        return self._compute_matches(self._first_ftr, new_ftr)

    def _get_ftr_points(self, img: U8Arr) -> ImgFtrDesc:
        """:return: ftr points, desc array as by opencv"""
        ftr, desc = self._detector.detectAndCompute(img, self._parent._mask)
        k = self._parent._config.max_ftr_points
        logger.debug(f'img size: {img.shape}; '
                     f'feature points: {len(ftr)}')
        if len(ftr) <= k:
            return ftr, desc

        mask = get_mask_for_largest(np.array([i.response for i in ftr]), k)
        assert type(ftr) is tuple
        ftr = tuple(np.array(ftr)[mask])
        desc = desc[mask]
        return ftr, desc

    def _compute_matches(self, ftr0, ftr1) -> typing.Optional[tuple[
            F32Arr, F32Arr]]:
        par = self._parent
        assert par._out_shape is not None
        maxdist = par._config.ftr_match_coord_dist * par._out_shape[0]
        kp0, desc0 = ftr0
        kp1, desc1 = ftr1

        if not (len(kp0) and len(kp1)):
            return

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
            logger.error(f'not enough matches: '
                         f'dist={dist/par._out_shape[0]} {mask=};'
                         ' consider increasing --ftr-match-coord-dist')
            return
        matches = np.array(matches)[mask]
        matches = sorted(matches, key=lambda x: x.distance)
        logger.info(f'coarse matching: num={len(matches)}/{len(mask)} '
                    f'avg_dist={dist.mean():.2g} '
                    f'filtered_dist={dist[mask].mean():.2g}')
        assert len(matches) >= 4
        return select_kpxy()

class FeatureMatcherXfeat(FeatureMatcher):
    _parent: 'ImageStackAlignment'

    def __init__(self, parent: 'ImageStackAlignment'):
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent /
                            'accelerated_features'))

        from modules.xfeat import XFeat

        self._parent = weakref.proxy(parent)
        self._xfeat = XFeat()

    def _get_ftr(self, img: F32Arr) -> dict:
        import torch
        if (m := self._parent._mask) is not None:
            img = img.copy()
            img[m < 127] = 0
        img_pt = torch.from_numpy(img)[torch.newaxis, torch.newaxis]
        return self._xfeat.detectAndComputeDense(
            img_pt, top_k=self._parent._config.max_ftr_points)

    def set_first_image(self, img: F32Arr, img_gray_8u: U8Arr):
        _ = img_gray_8u
        self._first_ftr = self._get_ftr(img)

    def get_matches(self, new_img: F32Arr) -> typing.Optional[
            tuple[F32Arr, F32Arr]]:
        import torch
        with torch.no_grad():
            new_ftr = self._get_ftr(new_img)
            idx_list = self._xfeat.batch_match(
                self._first_ftr['descriptors'], new_ftr['descriptors'])
            matches = self._xfeat.refine_matches(
                self._first_ftr, new_ftr, matches=idx_list, batch_idx=0)
            def as_npy(x):
                return x.cpu().numpy()
            return as_npy(matches[:, :2]), as_npy(matches[:, 2:])


class HomographyInterpolator:
    _parent: 'ImageStackAlignment'
    _homo_est: FocalRotationHomographyEstimator
    _enabled: bool

    _first_time: typing.Optional[datetime] = None

    _cur_time: float
    """relative current time in seconds"""
    _cur_H: F64Arr = np.eye(3, dtype=np.float64)
    _cur_H_good: bool = False

    _pred_cache: typing.Optional[F64Arr] = None

    _prev_H: F64Arr
    """previous H for prediction, if not _enabled"""

    _hist: list[tuple[float, F64Arr]]
    """history of (relative time, rotation vec) for good H if _enabled"""

    def __init__(self, parent: 'ImageStackAlignment'):
        self._parent = weakref.proxy(parent)
        self._hist = []
        if parent._config.homography_rotation_interpolation:
            assert utils.g_homography_estimator is not None, (
                'must use --use-restricted-transform for rotation interpolation')
            self._homo_est = utils.g_homography_estimator
            self._enabled = True
        else:
            self._enabled = False

    @classmethod
    def get_image_timestamp(cls, fpath: str) -> datetime:
        import exifread
        with open(fpath, 'rb') as fin:
            tags = exifread.process_file(fin)
        time_str = (str(tags['EXIF DateTimeOriginal']) + '.' +
                    str(tags['EXIF SubSecTimeOriginal']))
        return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S.%f')

    def on_new_image(self, fpath: str):
        """called when a new image is to be processed"""

        self._pred_cache = None
        if not self._enabled:
            self._prev_H = self._cur_H
            return

        time = self.get_image_timestamp(fpath)
        if self._first_time is None:
            self._first_time = time
        elif self._cur_H_good:
            rvec = self._homo_est.get_rvec_from_homography(self._cur_H)
            self._hist.append((self._cur_time, rvec))
            del self._cur_H

        self._cur_time = ((time - self._first_time) /
                          timedelta(microseconds=1)) * 1e-6
        self._cur_H_good = False

    def predict(self) -> F64Arr:
        """predict the initial homography for the current image"""
        if not self._enabled:
            return self._prev_H

        if len(self._hist) < 2:
            return np.eye(3, dtype=np.float64)

        if self._pred_cache is not None:
            return self._pred_cache.copy()

        all_x = np.array([i[0] for i in self._hist], dtype=np.float64)
        all_x = np.vstack([all_x, np.ones_like(all_x)]).T
        assert all_x.ndim == 2 and all_x.shape[1] == 2

        all_y = np.array([i[1] for i in self._hist], dtype=np.float64)
        assert (all_y.ndim == 2 and all_y.shape[1] == 3
                and all_y.shape[0] == all_x.shape[0])

        (rot_k, rot_b), res, _, _ = np.linalg.lstsq(all_x, all_y)

        rot_speed_hr = np.rad2deg(np.linalg.norm(rot_k)) * 3600
        res = np.linalg.norm(res)

        logger.info(f'homography interpolation: time={self._cur_time:.3f}s'
                    f' nr={len(self._hist)} residual={res:.2g}'
                    f' angular_velocity={rot_speed_hr:.3f}deg/hr')
        rvec = self._cur_time * rot_k + rot_b
        pred = self._homo_est.get_homography_from_rvec(rvec)
        self._pred_cache = pred
        return pred

    def update_homo(self, H: F64Arr, err: float):
        """update the homography of the current image; can be called multiple
        times"""
        assert H.shape == (3, 3)
        good = err < self._parent._config.refine_abort_thresh
        if not self._cur_H_good or good:
            if err < 5.0:
                self._cur_H = H
            self._cur_H_good = good


class ImageStackAlignment:
    _config: AlignmentConfig
    _coarse_mather: FeatureMatcher
    _refiner: RefinerBase
    _homo_interp: HomographyInterpolator

    _out_shape: typing.Optional[tuple[int, int]] = None
    # (w, h) of image shape

    _mask: typing.Optional[U8Arr] = None
    """ROI mask in ``{0, 255}``"""

    _is_first: bool = True
    _first_img_gray_8u: U8Arr
    _first_img_gray: F32Arr
    _first_str: ImgFtrDesc
    """corase align features on the first image"""

    _error_stat: list[float]

    prev_preproc_img: np.ndarray
    """previous preprocessed image for internal use"""

    def __init__(self, config: AlignmentConfig, refiner: RefinerBase):
        self._config = config
        self._refiner = refiner
        self._homo_interp = HomographyInterpolator(self)
        self._error_stat = []
        if config.use_xfeat:
            self._coarse_mather = FeatureMatcherXfeat(self)
        else:
            self._coarse_mather = FeatureMatcherCV2(self)

    def set_mask_from_file(self, fpath: str):
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f'failed to read {fpath}'
        img = self._handle_crop(img)
        mask = np.zeros_like(img)
        mask[img >= 127] = 255
        self._mask = mask

    def get_roi_mask(self) -> typing.Optional[npt.NDArray[np.bool_]]:
        """get the ROI mask"""
        if self._mask is None:
            return None
        return self._mask > 127

    def _apply_lens_correction(self, fpath: str, img: F32Arr) -> F32Arr:
        import lensfunpy
        import exifread
        with open(fpath, 'rb') as fin:
            tags = exifread.process_file(fin)
        cam_maker = str(tags['Image Make'])
        cam_model = str(tags['Image Model'])
        lens_maker = str(tags['EXIF LensMake'])
        lens_model = str(tags['EXIF LensModel'])

        db = lensfunpy.Database()
        cam = db.find_cameras(cam_maker, cam_model)
        assert cam, f'camera {cam_maker} {cam_model} not found'
        cam = cam[0]
        lens = db.find_lenses(cam, lens_maker, lens_model)
        assert lens, f'lens {lens_maker} {lens_model} not found'

        lens = lens[0]

        focal_length = (float(tags['EXIF FocalLength'].values[0].num) /
                        float(tags['EXIF FocalLength'].values[0].den))
        aperture = (float(tags['EXIF FNumber'].values[0].num) /
                    float(tags['EXIF FNumber'].values[0].den))
        mod = lensfunpy.Modifier(lens, cam.crop_factor, img.shape[1],
                                 img.shape[0])
        mod.initialize(
            focal_length, aperture, self._config.lens_correction_dist,
            pixel_format=np.float32)

        #cv2.imwrite('/tmp/img-0.jpg', (img * 255).astype(np.uint8))

        vignetting = mod.apply_color_modification(img)

        #cv2.imwrite('/tmp/img-1.jpg', (img * 255).astype(np.uint8))

        if ((coordmap := mod.apply_subpixel_geometry_distortion())
                is not None):
            geo_type = 2
            undist = np.empty_like(img)
            for i in range(3):
                undist[:, :, i] = cv2.remap(
                    img[:, :, i], coordmap[:, :, i], None,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT
                )
        else:
            coordmap = mod.apply_geometry_distortion()
            if coordmap is None:
                geo_type = 0
                undist = img
            else:
                geo_type = 1
                undist = cv2.remap(img, coordmap, None,
                                   interpolation=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT)
        #cv2.imwrite('/tmp/img-2.jpg', (undist * 255).astype(np.uint8))
        if not vignetting:
            logger.warning('vignetting correction failed')
        if geo_type != 2:
            logger.warning('geometry correction type is {geo_type}')
        logger.info('Lens correction:\n'
                    f' Camera: {cam}\n'
                    f' Lens: {lens}\n'
                    f' len={focal_length:.1f}; F/{aperture:.1f}')
        return undist

    def read_img(self, fpath: str) -> F32Arr:
        """read an image with basic corrections applied"""
        logger.info(f'loading {fpath}')
        img = read_img(fpath)
        if self._config.apply_lens_correction:
            img = self._apply_lens_correction(fpath, img)
        img = self._handle_crop(img)
        return img

    def feed_image_file(self, fpath: str) -> typing.Optional[
            tuple[F32Arr, npt.NDArray[np.bool_]]]:
        """feed an image to be aligned
        :return: (the aligned image, mask of valid values) or None if error too
            large
        """
        img = self.read_img(fpath)
        self._homo_interp.on_new_image(fpath)
        h, w, _ = img.shape
        if self._out_shape is None:
            self._out_shape = (w, h)
        else:
            assert self._out_shape == (w, h)
        return self._handle_image(img)

    def _handle_crop(self, img):
        c = self._config.crop_all_imgs
        if c:
            x, y, w, h = map(int, c.split(':'))
            img = img[y:y+h, x:x+w]
        else:
            x = y = 0
            h, w = img.shape[:2]

        if utils.g_homography_estimator is not None:
            utils.g_homography_estimator.set_image_size(w, h, x, y)
        return img

    def _img_preproc(self, src: F32Arr) -> F32Arr:
        """preprocess an input gray image"""
        src *= self._config.preproc_contrast
        src += self._config.preproc_brightness
        if self._config.preproc_gaussian > 0:
            src = cv2.GaussianBlur(src, (0, 0), self._config.preproc_gaussian)

        if self._config.mask_moving:
            assert self._mask is not None
            self._cur_mask = cv2.warpPerspective(
                self._mask, self._homo_interp.predict(), src.shape[::-1],
                flags=cv2.INTER_LINEAR | cv2. WARP_INVERSE_MAP
            )
        else:
            self._cur_mask = self._mask

        if self._config.remove_bg:
            flat = src
            if self._mask is not None:
                assert self._cur_mask is not None
                flat = flat[self._cur_mask > 127]

            thresh = np.quantile(flat.flatten(), self._config.remove_bg_thresh)
            src = np.maximum(src - thresh, 0)
            src /= src.max()
        src = np.clip(src, 0, 1, out=src)
        return src

    def _handle_image(self, img: F32Arr) -> typing.Optional[
            tuple[F32Arr, npt.NDArray[np.bool_]]]:
        if self._config.use_identity_trans:
            return img, np.ones_like(img[:, :, 0], dtype=np.bool_)
        img_gray = self._img_preproc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.prev_preproc_img = img_gray
        img_gray_8u = img_as_u8(img_gray).astype(np.uint8)
        if self._config.preproc_show:
            disp_img('preproc', img_gray_8u)

        if self._is_first:
            self._is_first = False
            self._refiner.set_first_image(img_gray, img_gray_8u, self)
            self._first_img_gray = img_gray
            self._first_img_gray_8u = img_gray_8u
            if not self._config.skip_coarse_align:
                self._coarse_mather.set_first_image(img_gray, img_gray_8u)
            return img, np.ones(img.shape[:2], dtype=np.bool_)

        # coarse align
        err = np.inf
        if self._config.skip_coarse_align:
            H = self._homo_interp.predict()
        else:
            Hinit = self._homo_interp.predict()
            warp_as_first = cv2.warpPerspective(img_gray, Hinit, self._out_shape)

            if self._config.linear_trans_before_ftr_match:
                xi = warp_as_first
                xdst = self._first_img_gray
                if self._mask is not None:
                    xi = xi[self._mask > 127]
                    xdst = xdst[self._mask > 127]
                else:
                    xi = xi.flatten()
                xi = np.vstack([xi, np.ones_like(xi)]).T
                tr_k, tr_b = np.linalg.lstsq(xi, xdst, rcond=None)[0]
                logger.info(f'for ftr match: {tr_k=:.3f} {tr_b=:.3f}')
                warp_as_first *= tr_k
                warp_as_first += tr_b
                warp_as_first = np.clip(warp_as_first, 0, 1, out=warp_as_first)

            match_info = self._coarse_mather.get_matches(warp_as_first)

            if match_info is None:
                logger.warning('image discarded since match failed')
                return

            p0xy, p1xy = match_info
            H, err, _, mask = find_homography(  # type: ignore
                p1xy, p0xy, cv2.RANSAC, need_mask=True)
            p0xy = p0xy[mask]
            p1xy = p1xy[mask]

            if self._config.draw_matches:
                img0 = self._first_img_gray_8u
                img1 = img_as_u8(warp_as_first)
                cv_kpt0 = [cv2.KeyPoint(p[0], p[1], 5) for p in p0xy]
                cv_kpt1 = [cv2.KeyPoint(p[0], p[1], 5) for p in p1xy]
                cv_match = [cv2.DMatch(i,i,0) for i in range(len(p0xy))]
                disp_img('matches',
                         cv2.drawMatches(
                             img0, cv_kpt0, img1, cv_kpt1,
                             cv_match, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS),
                         wait=not self._config.draw_matches_no_wait)

            H = H @ Hinit

            logger.debug(f'coarse align: err={err:.2g} pt={len(p0xy)}')

        self._homo_interp.update_homo(H, err)

        for _ in range(self._config.refine_iters):
            Htp = self._refiner.get_trans(H, img_gray)

            if Htp is None:
                return

            H, err = Htp
            self._homo_interp.update_homo(H, err)

        if err >= self._config.refine_abort_thresh:
            logger.warning(f'image discarded due to high error: {err:.3g}')
            return

        if self._config.star_quality_filter > 0:
            qual_img = img_gray
            if self._cur_mask is not None:
                qual_img = np.where(self._cur_mask < 127, qual_img, 0)
            star_pt = StarPointRefiner.find_star_centers(qual_img, self._config)
            qual = StarPointRefiner.get_star_qual(star_pt)
            logger.info(f'star quality: {qual:.3f}; nr_star={len(star_pt)}')
            if qual < self._config.star_quality_filter:
                logger.warning(f'image discarded due to low star quality')
            return

        self._error_stat.append(err)

        # use nearest neighbor interpolation to avoid blurring; there should not
        # be too much artifacts because we will average multiple images
        newimg = cv2.warpPerspective(img, H, self._out_shape,
                                     flags=cv2.INTER_NEAREST)
        mask = np.empty(img.shape[:2], dtype=np.uint8)
        mask[:] = 255
        mask = cv2.warpPerspective(mask, H, self._out_shape)
        mask = mask >= 127
        return newimg, mask

    def error_stat(self):
        """get error stats"""
        return np.array(self._error_stat, dtype=np.float32)
