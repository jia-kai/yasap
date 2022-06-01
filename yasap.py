#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import argparse
import collections
import itertools
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger('yasap')

class ImageCache:
    """in-memory cache of image arrays of the same size"""

    fpaths: list[str]

    shape: tuple[int] = None
    # (w, h)

    width: int
    height: int

    def __init__(self, fpaths: list[str]):
        self.fpaths = fpaths

    def get_img(self, idx: int) -> np.ndarray:
        """get the content of the image at given index; return f32 img"""
        # note: since we process the images sequentially, cache swapping does
        # not help unless all images are loaded in memory. So we do not
        # implement cache for now
        ret = cv2.imread(
            self.fpaths[idx], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        assert ret is not None, f'failed to read {self.fpaths[idx]}'
        ret = ret.astype(np.float32) / np.float32(np.iinfo(ret.dtype).max)
        h, w, _ = ret.shape
        if self.shape is None:
            self.shape = (w, h)
            self.width = w
            self.height = h
        else:
            assert self.shape == (w, h)
        return ret

    def get_gray_img_8bit(self, idx: int) -> np.ndarray:
        ret = cv2.cvtColor(self.get_img(idx), cv2.COLOR_BGR2GRAY) * 255
        return ret.astype(np.uint8)

    def save_img(self, img: np.ndarray, fpath: str):
        """save image with appropriate exif info"""
        assert img.dtype == np.float32
        img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(fpath, img)

    def __len__(self):
        return len(self.fpaths)


class StreamingStack(metaclass=abc.ABCMeta):
    """base class for stacking a stream of alined images"""

    @abc.abstractmethod
    def add_img(self, img):
        pass

    @abc.abstractmethod
    def get_result(self):
        pass

class MeanStreamingStack:
    """stack by mean"""

    _nr_img = 0
    _sum_img = None

    def add_img(self, img: np.ndarray):
        if self._sum_img is None:
            self._sum_img = img.copy()
        else:
            self._sum_img += img
        self._nr_img += 1

    def get_result(self):
        return self._sum_img / self._nr_img


class AlignmentConfig:
    """default configuration options"""

    ftr_match_coord_dist = 0.01
    """maximal distance of coordinates between matched feature points, relative
    to image width"""

    draw_matches = False
    """weather to draw match points"""

    disp_coarse_warped = False
    """weather to display warpped coare images"""

    disp_refine = False
    """display patches used in refinement"""

    refine_patch_size = 50
    """size of each patch used in refinement"""

    refine_iters = 20
    """number of iterations"""

    save_refined_dir = ''
    """directory to save refined images"""


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

def find_homography(src, dst, method):
    """:return: H, avg dist"""
    H, _ = cv2.findHomography(src, dst, method)
    dist = np.sqrt(
        ((perspective_transform(H, src) - dst)**2).sum(axis=1)).mean()
    return H, dist

class AABox(collections.namedtuple('AABoxBase', ['x0', 'y0', 'x1', 'y1'])):
    def scaled(self, fx, fy) -> 'AABox':
        return AABox(self.x0 * fx, self.y0 * fy, self.x1 * fx, self.y1 * fy)

    def as_int(self) -> 'AABox':
        return AABox(int(np.floor(self.x0)), int(np.floor(self.y0)),
                     int(np.ceil(self.x1)), int(np.ceil(self.y1)))

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0


PatchInfo = collections.namedtuple(
    'PatchInfo',
    ['patch',       # (h, w, c) image
     'grid_rela',   # (h, w, 2) for relative coordinates on the source image
     ]
)

class ImageStackAlignment:
    _config: AlignmentConfig
    _img_cache: ImageCache

    _coarse_transforms: list[np.ndarray]
    """list of perspective transforms from image[i] to image[0]"""

    _out_shape: tuple[int]

    _roi_box: AABox = AABox(0.2, 0.2, 0.8, 0.8)
    # x0, y0, x1, y1 of the max rect that fits within ROI

    _detector = None
    _matcher = None

    _rng: np.random.RandomState

    def __init__(self, img_paths: list[str], config=AlignmentConfig()):
        self._config = config
        self._img_cache = ImageCache(img_paths)
        self._detector = cv2.xfeatures2d.SURF_create(hessianThreshold=800)
        self._matcher = cv2.BFMatcher_create(cv2.NORM_L2, True)
        self._compute_coarse_transforms()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._rng = np.random.RandomState(42)

    def refine_and_stack(self, stacker: StreamingStack):
        """refine the transforms and feed the aligned images into the stacker"""
        img0 = self._img_cache.get_img(0)
        img0_pyr = self._build_refine_pyr(img0)
        stacker.add_img(img0)
        for idx in range(1, len(self._img_cache)):
            img1, loss = self._refine_and_apply(
                self._img_cache.get_img(idx), img0, img0_pyr,
                self._coarse_transforms[idx]
            )
            logger.debug(f'refine image {idx}: {loss=:.2g}')
            if (d := self._config.save_refined_dir):
                self.save_img(
                    img1, os.path.join(d, f'refine{idx}.tif'))
            stacker.add_img(img1)

    def save_img(self, img, fpath):
        return self._img_cache.save_img(img, fpath)

    def _scale_homography_to_relative(self, H: np.ndarray) -> np.ndarray:
        """scale the homography matrix to use relative ([0, 1]) coordinates"""
        H = H.copy()
        w, h = map(float, self._img_cache.shape)
        x = w - 1
        y = h - 1
        H[0] *= [1.0, y/x, 1/x]
        H[1] *= [x/y, 1.0, 1/y]
        H[2] *= [x, y, 1.0]
        return H

    def _build_refine_pyr(self, img: np.ndarray) -> list[np.ndarray]:
        """build an image pyramid for refinement"""
        pyr = []
        min_h = int(
            np.ceil(self._config.refine_patch_size / self._roi_box.height * 2))
        min_w = int(
            np.ceil(self._config.refine_patch_size / self._roi_box.width * 2))
        while img.shape[0] > min_h and img.shape[1] > min_w:
            if not pyr:
                pyr.append(cv2.GaussianBlur(img, (7, 7), 0))
            else:
                pyr.append(img)
            img = cv2.pyrDown(img)
        assert pyr
        return pyr

    def _make_grid_rela(self, x0, y0, x1, y1, w, h):
        """create a (y1 - y0, x1 - x0, 2) array containing relative coordinates
        in the given rectangle"""
        assert x0 < x1 <= w and y0 < y1 <= h
        gx = np.arange(x0, x1) / (w - 1)
        gy = np.arange(y0, y1) / (h - 1)
        gxv, gyv = np.meshgrid(gx, gy)
        grid = np.concatenate(
            [np.expand_dims(gxv, 2), np.expand_dims(gyv, 2)],
            axis=2
        ).astype(np.float32)
        assert grid.shape == (y1 - y0, x1 - x0, 2)
        return grid

    def _extract_refine_patches(self, img) -> list[PatchInfo]:
        """extract patches in the ROI from a single image"""
        patch_size = self._config.refine_patch_size
        roi = self._roi_box.scaled(img.shape[1], img.shape[0]).as_int()

        ret = []
        tw = roi.width // 2
        th = roi.height // 2
        h, w = img.shape[:2]
        for dx, dy in itertools.product(range(2), range(2)):
            x0 = roi.x0 + tw * dx
            y0 = roi.y0 + th * dy
            if tw > patch_size:
                x0 += self._rng.randint(tw - patch_size + 1)
            if th > patch_size:
                y0 += self._rng.randint(th - patch_size + 1)

            ret.append(PatchInfo(
                img[y0:y0+patch_size, x0:x0+patch_size],
                self._make_grid_rela(
                    x0, y0, x0 + patch_size, y0 + patch_size, w, h)
            ))
        return ret

    def _refine_and_apply(self, img_src, img_dst, dst_pyr, H):
        """refine and apply the perspective transform from ``img_src`` to
        ``img_dst``
        :return: transformed image, loss
        """
        Hinv = np.linalg.inv(H)
        Hinv = self._scale_homography_to_relative(Hinv)
        Hinv /= np.sqrt((Hinv**2).sum())
        src_pyr = self._build_refine_pyr(img_src)
        assert img_src.shape == img_dst.shape and len(src_pyr) == len(dst_pyr)
        patch_size = self._config.refine_patch_size
        for src_i, dst_i in zip(src_pyr, dst_pyr):
            assert src_i.shape == dst_i.shape
            dst_patches = self._extract_refine_patches(dst_i)
            Hinv, loss = self._refine_for_patches(Hinv, dst_patches, src_i)

        return self._apply_pytorch_ptrans(Hinv, img_src), loss

    def _as_torch(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32)).to(self._device)

    @classmethod
    def _as_npy(cls, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    @classmethod
    def _cvimg2nchw(cls, img: np.ndarray) -> np.ndarray:
        assert (img.ndim == 3 and img.shape[2] in [1, 3] and
                img.dtype == np.float32)
        return np.transpose(img, (2, 0, 1))[np.newaxis]

    def _refine_for_patches(self, Hinv: np.ndarray,
                            dst_patches: list[PatchInfo],
                            img_src: np.ndarray) -> np.ndarray:
        """use pytorch to refine the transform matrix for a set of patches

        :return: updated Hinv, loss
        """
        patch_size = self._config.refine_patch_size
        assert img_src.dtype == np.float32
        as_torch = self._as_torch
        img_src = as_torch(self._cvimg2nchw(img_src)).expand(
            len(dst_patches), -1, -1, -1)
        HinvT = nn.Parameter(as_torch(Hinv.T))
        opt = optim.AdamW([HinvT], lr=1e-4)
        all_patches = as_torch(np.concatenate(
            [self._cvimg2nchw(i.patch) for i in dst_patches],
            axis=0))
        all_grid = np.concatenate(
            [i.grid_rela[np.newaxis] for i in dst_patches],
            axis=0
        )
        all_grid = np.concatenate(
            [all_grid, np.ones_like(all_grid[:, :, :, : 1])],
            axis=3).reshape(-1, 3)
        all_grid = as_torch(all_grid)

        best_loss = float('inf')
        best_HinvT = None
        for it in range(self._config.refine_iters):
            opt.zero_grad()
            grid_on_src = all_grid @ (HinvT / torch.linalg.matrix_norm(HinvT))
            # pytorch grid_sample needs values in [-1, 1]
            grid_on_src = (
                (grid_on_src[:, :2] / grid_on_src[:, 2:] * 2 - 1).reshape(
                    len(dst_patches), patch_size, patch_size, 2)
            )
            sampled = nn.functional.grid_sample(
                img_src, grid_on_src, align_corners=False)

            v0 = all_patches.reshape(-1)
            v1 = sampled.reshape(-1)
            loss = 1 - torch.dot(v0, v1) / (
                torch.linalg.vector_norm(v0) *
                torch.linalg.vector_norm(v1))
            if loss < best_loss:
                best_loss = loss.item()
                best_HinvT = HinvT.clone()

            if self._config.disp_refine:
                print(f'iter {it}: {loss=:.2g}')
                self._disp_refine(all_patches, sampled)

            loss.backward()
            opt.step()
            with torch.no_grad():
                HinvT.requires_grad_(False)
                HinvT /= torch.linalg.matrix_norm(HinvT)
                HinvT.requires_grad_(True)

        return np.ascontiguousarray(self._as_npy(best_HinvT).T), best_loss

    def _apply_pytorch_ptrans(self, Hinv: np.ndarray,
                              img_src: np.ndarray) -> np.ndarray:
        """apply perspective transform using pytorch implementation to maintain
        compatiblility with :meth:`_refine_for_patches`"""
        w, h = self._img_cache.shape
        grid = self._make_grid_rela(0, 0, w, h, w, h).reshape(-1, 2)
        grid = np.concatenate([grid, np.ones_like(grid[:, :1])], axis=1)
        grid = self._as_torch(grid) @ self._as_torch(Hinv.T)
        grid = (grid[:, :2] / grid[:, 2:] * 2 - 1).reshape(1, h, w, 2)
        img_src = self._as_torch(self._cvimg2nchw(img_src))
        img_dst = nn.functional.grid_sample(img_src, grid, align_corners=False)
        return np.squeeze(self._torch_nchw_to_npy_nhwc(img_dst), 0)

    @classmethod
    def _torch_nchw_to_npy_nhwc(cls, x: torch.Tensor) -> np.ndarray:
        return cls._as_npy(torch.permute(x, (0, 2, 3, 1)))

    def _disp_refine(self, pdst: torch.Tensor, pget: torch.Tensor):
        """display refinement internal patches"""
        patch_size = self._config.refine_patch_size
        sep = 2
        pdst = self._torch_nchw_to_npy_nhwc(pdst)
        pget = self._torch_nchw_to_npy_nhwc(pget)
        img = np.zeros((patch_size * 2 + sep,
                        len(pdst) * (patch_size + sep) - sep,
                        pdst.shape[3]),
                       dtype=np.float32)
        for i in range(len(pdst)):
            x0 = i * (patch_size + sep)
            x1 = x0 + patch_size
            img[:patch_size, x0:x1] = pdst[i]
            img[-patch_size:, x0:x1] = pget[i]

        cv2.imshow('refine_patches', img)
        cv2.waitKey(-1)

    def _compute_coarse_transforms(self):
        """initialize :attr:`_coarse_transforms` using keypoint matching"""
        cur_ftr = None
        cur_img = None
        first_ftr = None
        trans = [np.eye(3, dtype=np.float64)]

        for idx in range(len(self._img_cache)):
            logger.debug(f'======= coarse match: '
                         f'processing image {idx}/{len(self._img_cache)}')
            prev_img = cur_img
            prev_ftr = cur_ftr
            cur_img = self._img_cache.get_gray_img_8bit(idx)
            cur_ftr = self._detector.detectAndCompute(cur_img, None)
            logger.debug(f'num of feature kpts: {len(cur_ftr[0])}')
            if idx == 0:
                first_ftr = cur_ftr
                if self._config.disp_coarse_warped:
                    self._disp_img('first', cur_img, wait=False)
                continue

            matches, p0xy, p1xy = self._compute_matches(
                'coarse', prev_ftr, cur_ftr)

            H, H_err = find_homography(p1xy, p0xy, cv2.RANSAC)
            trans.append(trans[idx - 1] @ H)  # trans to first image
            logger.debug(f'homography error: {H_err:.2g}')

            if self._config.draw_matches:
                print(H)
                img = cv2.drawMatches(
                    prev_img, prev_ftr[0], cur_img, cur_ftr[0],
                    matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self._disp_img('matches', img)

            if self._config.disp_coarse_warped:
                self._disp_img('this', cur_img, wait=False)
                self._disp_img(
                    'warp',
                    cv2.warpPerspective(cur_img, H, self._img_cache.shape),
                    wait=True)

        self._coarse_transforms = trans

    @classmethod
    def _disp_img(cls, title: str, img: np.ndarray, wait=True):
        """display an image"""
        scale = 1000 / max(img.shape)
        if scale < 1:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(title, img)
        if wait:
            cv2.waitKey(-1)

    def _compute_matches(self, log_name, ftr0, ftr1, maxdist=None):
        """compute opencv keypoint matches with distance filtering

        :return: matches, keypoint xy of ftr0, keypoint xy of ftr1
        """
        if maxdist is None:
            maxdist = self._config.ftr_match_coord_dist * self._img_cache.width
        kp0, desc0 = ftr0
        kp1, desc1 = ftr1

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
        logger.info(f'{log_name} matching: num={len(matches)} '
                    f'avg_dist={dist.mean():.2g} '
                    f'filtered_dist={dist[mask].mean():.2g}')
        assert len(matches) >= 10
        return matches, *select_kpxy()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('imgs', nargs='+')
    parser.add_argument('-o', '--output', required=True, help='output file path')
    for i in dir(AlignmentConfig):
        if not i.startswith('_'):
            v = getattr(AlignmentConfig, i)
            parser.add_argument(f'--{i}', type=type(v), default=v,
                                help='see source')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(suppress=True)

    config = AlignmentConfig()
    for i in dir(AlignmentConfig):
        if not i.startswith('_'):
            setattr(config, i, getattr(args, i))
    align = ImageStackAlignment(args.imgs, config)
    stacker = MeanStreamingStack()
    align.refine_and_stack(stacker)
    align.save_img(stacker.get_result(), args.output)

if __name__ == '__main__':
    main()
