from .config import ConfigWithArgparse
from .utils import logger, precise_quantile, read_img

import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
import typing

class StackerBase(metaclass=ABCMeta):
    class Config(ConfigWithArgparse):
        linear_rgb_match: bool = False
        """adjust contrast/brightness of RGB channels independently before
        stacking; useful for filtering uniform clouds / different exposures"""

        linear_rgb_match_dist_thresh: float = 0.005
        """distance threshold for linear RGB matching; images would be discarded
        if L2 distance of quantiles is larger than this"""

    _first_image: typing.Optional[np.ndarray] = None

    _linear_match_check_quants = np.concatenate([
        np.linspace(0, 0.9, 50),
        np.linspace(0.9, 1, 50),
    ])

    _config: Config

    def set_config(self, config: Config):
        """set whether to match mean color"""
        self._config = config

    def add_img(self, img: np.ndarray, mask: np.ndarray) -> bool:
        """add a new image to only the masked area; first mask is guaranteed to
        be full frame"""
        assert (img.dtype == np.float32 and mask.dtype == np.bool_ and
                mask.shape == img.shape[:2] and
                img.ndim == 3 and img.shape[2] == 3), (img.shape, mask.shape)
        if self._first_image is None:
            assert np.all(mask), 'first image must be all valid'
        elif self._config.linear_rgb_match:
            mfull = np.broadcast_to(mask[:, :, np.newaxis], img.shape)
            def get_mean_std(img):
                sub = np.ascontiguousarray(img[mfull].reshape(-1, 3).T)
                qs = np.quantile(sub, self._linear_match_check_quants, axis=1)
                return np.mean(sub, axis=1), np.std(sub, axis=1), qs

            first_mean, first_std, first_q = get_mean_std(self._first_image)
            this_mean, this_std, this_q = get_mean_std(img)
            b = first_mean - this_mean
            k = first_std / np.maximum(this_std, 1e-6)
            cmp_q = (this_q + b)
            cmp_q *= k

            dist = float(np.sqrt(np.square((cmp_q - first_q).flatten()).mean()))
            if dist > (thresh := self._config.linear_rgb_match_dist_thresh):
                logger.warning(f'discard due to large dist after linear '
                               f'RGB match: {dist=:.3g} {thresh=:.3g}')
                return False

            img = img + b
            img *= k
            img = np.clip(img, 0, 1, out=img)
            ks = ', '.join(map('{:.3f}'.format, k))
            bs = ', '.join(map('{:.3f}'.format, b))
            logger.info(f'Linear RGB match: k=[{ks}] b=[{bs}] {dist=:.3g}')

        self._do_add_img(img, mask)

        if self._first_image is None:
            self._first_image = img
        return True

    @property
    def _is_first_img(self) -> bool:
        return self._first_image is None

    @abstractmethod
    def _do_add_img(self, img, mask):
        raise NotImplementedError()

    @abstractmethod
    def get_preview_result(self) -> np.ndarray:
        """get inaccurate result for fast preview"""
        raise NotImplementedError()

    @abstractmethod
    def get_result(self) -> np.ndarray:
        raise NotImplementedError()


class SoftMaxStacker(StackerBase):
    """stack by soft max, for star trails"""

    class Config(ConfigWithArgparse):
        n: int
        """number of input images"""

        temp_rank: float = 0.9999
        """a parameter to control temperature, which is the balance between mean
        and max value for SoftMaxStacker; this parameter defines a threshold as
        the relative rank of values on the first input image, so that values
        above this threshold are reduced by no more than ``temp_delta``"""

        temp_delta: float = 0.05
        """see  ``temp_rank``"""

        def __init__(self, n: int, **kwargs):
            super().__init__(**kwargs)
            self.n = n

        def update_from_args(self, args):
            super().update_from_args(args)
            self.n = len(args.imgs)
            return self

    config: Config
    temp: float

    _cnt: np.ndarray
    """number of images at each pixel location; ndim=2"""

    _sum_img: np.ndarray

    def __init__(self, config: "SoftMaxStacker.Config"):
        self.config = config

    def _do_add_img(self, img: np.ndarray, mask: np.ndarray):
        if self._is_first_img:
            # let k be the temp, M be the thresh
            # assuming only one M and others are zeros
            # softmax / M = 1 - (log n) / (kM)
            thresh = precise_quantile(img, self.config.temp_rank)
            assert thresh > 0
            self.temp = np.log(self.config.n) / (thresh * self.config.temp_delta)
            logger.info(f'SoftMaxStacker: {thresh=:.3f} temp={self.temp:.3f}')

            self._cnt = np.ones_like(mask, dtype=np.float64)
            self._sum_img = img.astype(np.float64).copy()
            self._sum_img *= self.temp
            return

        mask3d = np.broadcast_to(np.expand_dims(mask, 2), img.shape)

        self._cnt[mask] += 1

        # (x, 1)
        new_cnt_sel = np.expand_dims(self._cnt[mask], 1)
        # (x, 3)
        val_sel = self._sum_img[mask3d].reshape(new_cnt_sel.size, 3)
        valu_sel = img[mask3d].reshape(new_cnt_sel.size, 3) * self.temp

        ka = np.log1p((-1) / new_cnt_sel)
        kb = np.log(new_cnt_sel)
        new_val_sel = np.logaddexp(val_sel + ka, valu_sel - kb)
        self._sum_img[mask3d] = new_val_sel.flatten()

    def get_preview_result(self) -> np.ndarray:
        return self.get_result()

    def get_result(self) -> np.ndarray:
        return (self._sum_img / self.temp).astype(np.float32)


class MeanStacker(StackerBase):
    """stack by mean value that allows removal of min/max outliers"""

    class Config(ConfigWithArgparse):
        rm_min: int = 0
        """number of extreme min values to be removed for mean stacker"""

        rm_max: int = 0
        """number of extreme max values to be removed for mean stacker"""


    INF_VAL = 2.3

    _mean_img: np.ndarray

    _min_list: list[np.ndarray]
    """lowest N values"""

    _max_list: list[np.ndarray]
    """lowest N values of negative images"""

    def __init__(self, config: "MeanStacker.Config"):
        self.rm_min = config.rm_min
        self.rm_max = config.rm_max

    def _update_min_list(self, dst: list[np.ndarray], img: np.ndarray):
        if not dst:
            return
        for i in range(len(dst)):
            new = np.minimum(dst[i], img)
            if i == len(dst) - 1:
                del img
            else:
                # write to dst[i] to save memory (not img because we do not want
                # to modify input param)
                np.maximum(dst[i], img, out=dst[i])
                img = dst[i]
            dst[i] = new
            del new

    def _do_add_img(self, img: np.ndarray, mask: np.ndarray):
        if self._is_first_img:
            self._mean_img = img.copy()
            self._cnt_img = np.ones_like(img[:, :, 0], dtype=np.int16)
            fill = np.empty_like(self._mean_img)
            fill[:] = self.INF_VAL
            self._min_list = [fill.copy() for _ in range(self.rm_min)]
            self._max_list = [fill.copy() for _ in range(self.rm_max)]
            del fill
        else:
            # let u[n] = sum(x[1:n+1])/n
            # then u[n] = u[n - 1] + (x[n] - u[n - 1]) / n
            self._cnt_img += mask
            d = img - self._mean_img
            d *= np.expand_dims(
                mask.astype(np.float32) / self._cnt_img.astype(np.float32),
                2)
            self._mean_img += d
            del d

        if self.rm_min or self.rm_max:
            img = img.copy()
            bad_mask = np.broadcast_to(
                np.expand_dims(np.logical_not(mask), 2), img.shape)
            img[bad_mask] = self.INF_VAL

            self._update_min_list(self._min_list, img)
            if self.rm_max:
                np.negative(img, out=img)
                img[bad_mask] = self.INF_VAL
                self._update_min_list(self._max_list, img)

    def _sum_list_non_inf(self, imgs: list[np.ndarray], cnt) -> np.ndarray:
        """sum list, treating values of INF_VAL as zero

        :param cnt: number of non-zeros, modified inplace, single channel
        :return: sum
        """
        r = None
        for i in imgs:
            mask = i < self.INF_VAL - 1e-4
            cnt[mask[:, :, 0]] += 1
            i = i * mask
            if r is None:
                r = i
            else:
                r += i
        return r

    def get_preview_result(self) -> np.ndarray:
        """get result for preview, which is fast"""
        assert self._mean_img is not None
        return self._mean_img

    def get_result(self) -> np.ndarray:
        """get the final result"""
        assert self._mean_img is not None
        if not (self._max_list or self._min_list):
            return self._mean_img

        minmax_cnt = np.zeros_like(self._cnt_img)
        sub = None

        if self._min_list:
            sub = self._sum_list_non_inf(self._min_list, minmax_cnt)
        if self._max_list:
            # note that max list has been negated
            c = self._sum_list_non_inf(self._max_list, minmax_cnt)
            if sub is None:
                sub = c
                sub = np.negative(sub, out=sub)
            else:
                sub -= c
            del c

        # (mean * n - sub) / (n - k) = mean - (sub - mean * k) / (n - k)

        tot_cnt = np.expand_dims(self._cnt_img, 2)
        minmax_cnt = np.expand_dims(minmax_cnt, 2)
        sub -= self._mean_img * minmax_cnt
        sub /= np.maximum(tot_cnt - minmax_cnt, 1)
        edge = np.broadcast_to(tot_cnt <= self.rm_min + self.rm_max, sub.shape)
        sub[edge] = 0

        return self._mean_img - sub


class CRgbCombineStacker(StackerBase):
    """combine four filtered images into RGB; images given in the order gray
    clear, R, G, B"""

    class Config(ConfigWithArgparse):
        crgb_use_pca: int = 0
        """whether to use PCA to compute H and S components"""

        crgb_pca_mask: str = ''
        """mask file for PCA computing"""


    _imgs: list[np.ndarray]
    _config: Config

    def __init__(self, config: "CRgbCombineStacker.Config"):
        self._imgs = []
        self._config = config

    def _do_add_img(self, img: np.ndarray, mask):
        self._imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[
            :, :, np.newaxis])

    def get_preview_result(self) -> np.ndarray:
        return self._imgs[0]

    def get_result(self) -> np.ndarray:
        assert len(self._imgs) == 4, f'4 images expected; got {len(self._imgs)}'
        def rescale(x: np.ndarray, maxv=1.0):
            x -= x.min()
            x *= maxv / x.max()
            return x

        gray, r, g, b = self._imgs
        bgr = np.concatenate([b, g, r], axis=2)
        V = rescale(np.clip(gray, 0, 1))

        if self._config.crgb_use_pca:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            h, w, _ = bgr.shape
            flat = bgr.reshape(h*w, 3)

            if self._config.crgb_pca_mask:
                mask = cv2.cvtColor(read_img(self._config.crgb_pca_mask),
                                    cv2.COLOR_BGR2GRAY) > .5
                pca.fit(flat[mask.reshape(h*w), :])
                logger.info(f'PCA mask: {np.count_nonzero(mask)}')
            else:
                pca.fit(flat)
            logger.info(f'PCA comp: {pca.components_}')
            H, S = pca.components_ @ flat.T
            if self._config.crgb_pca_mask:
                Hroi = H[mask.flatten()]
                vmin = Hroi.min()
                vmax = Hroi.max()
                tmin = 0.05
                tmax = 0.95
                H = (H - vmin) / (vmax - vmin) * (tmax - tmin) + tmin
                H = np.clip(H, 0, 1)
            else:
                H = rescale(H, maxv=.99)
            S = rescale(S)
            logger.info('PCA explained variance:'
                        f' {pca.explained_variance_ratio_};'
                        f' H std: {np.std(H)}; S std: {np.std(S)} ')
            H = H.reshape((h, w, 1)) * 360
            S = S.reshape((h, w, 1))
            hsv = np.concatenate([H, S, V], axis=2)
        else:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.squeeze(V, axis=2)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        logger.info(f'bgr range: {bgr.min()} {bgr.max()}')
        bgr -= bgr.min()
        bgr /= bgr.max()
        return bgr


def test_mean_stacker():
    import sys
    rng = np.random.RandomState(42)

    def run(img_w, img_h, nr_img, rm_min, rm_max):
        print(f'test {img_h}x{img_w} n={nr_img} rm=({rm_min},{rm_max})')
        imgs = (rng.uniform(0, 1, size=(nr_img, img_h, img_w, 3))
                .astype(np.float32))
        mask_cnt = rng.randint(1, nr_img + 1, size=img_h*img_w)
        mask_cnt[0] = 1 # ensure some edge cases
        mask_cnt[1] = 2

        masks = np.empty((nr_img, img_h, img_w), dtype=np.bool_)
        for i in range(img_h * img_w):
            m = np.zeros(nr_img, dtype=np.bool_)
            m[:mask_cnt[i]] = 1
            rng.shuffle(m[1:])  # first image has all mask on
            masks[:, i // img_w, i % img_w] = m

        stack = MeanStacker(MeanStacker.Config(rm_min=rm_min, rm_max=rm_max))
        for i, j in zip(imgs, masks):
            stack.add_img(i, j)
        got = stack.get_result()

        for i in range(img_h):
            for j in range(img_w):
                values = imgs[masks[:, i, j], i, j]
                assert values.ndim == 2 and values.shape[1] == 3, values.shape
                if values.shape[0] > rm_min + rm_max:
                    values = np.sort(values, axis=0)
                    values = values[rm_min:]
                    if rm_max:
                        values = values[:-rm_max]

                expect = np.mean(values, axis=0)
                goti = got[i, j]

                if not np.allclose(expect, goti):
                    print(f'fail at {i},{j}:')
                    print('expect:')
                    print(expect)
                    print('got:')
                    print(goti)
                    print('mask:', masks[:, i, j])
                    print('imgs:')
                    print(imgs[:, i, j])
                    sys.exit(1)

    run(3, 3, 2, 0, 0)
    run(3, 3, 3, 0, 1)
    run(3, 3, 3, 1, 0)
    run(3, 3, 8, 0, 1)
    run(3, 3, 8, 1, 1)

    run(20, 20, 10, 3, 5)
    run(21, 35, 25, 5, 3)

def test_softmax_stacker():
    from scipy.special import logsumexp
    num = 5
    h = 64
    w = 64
    rng = np.random.default_rng(1234)
    stacker = SoftMaxStacker(SoftMaxStacker.Config(n=num))

    imgs = rng.uniform(0, 1, size=(num, h, w, 3)).astype(np.float32)
    masks = rng.uniform(0, 1, size=(num, h, w)) < 0.7
    masks[0] = True

    for i in range(num):
        stacker.add_img(imgs[i], masks[i])

    expect = np.empty((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            inp = imgs[:, i, j][np.broadcast_to(
                np.expand_dims(masks[:, i, j], 1),
                (num, 3)
            )]
            inp = inp.reshape(-1, 3)
            assert 1 <= inp.shape[0] <= num
            expect[i, j] = (
                logsumexp(inp * stacker.temp, axis=0) -
                np.log(inp.shape[0])
            ) / stacker.temp

    np.testing.assert_allclose(stacker.get_result(), expect,
                               rtol=1e-4, atol=1e-4)


STACKER_DICT = {
    'mean': MeanStacker,
    'softmax': SoftMaxStacker,
    'crgb': CRgbCombineStacker,
}

if __name__ == '__main__':
    test_softmax_stacker()
    test_mean_stacker()
