import numpy as np

class StreamingStacker:
    """stacking a stream of aligned images"""

    INF_VAL = 2.3

    _mean_img = None

    _min_list = None
    """lowest N values"""

    _max_list = None
    """lowest N values of negative images"""

    def __init__(self, rm_min: int, rm_max: int):
        self.rm_min = rm_min
        self.rm_max = rm_max

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

    def add_img(self, img: np.ndarray, mask: np.ndarray):
        assert (img.dtype == np.float32 and mask.dtype == np.bool_ and
                mask.shape == img.shape[:2] and
                img.ndim == 3 and img.shape[2] == 3), (img.shape, mask.shape)
        if self._mean_img is None:
            assert np.all(mask), 'first image must be all valid'
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

    def get_result(self):
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

def run_random_test():
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

        stack = StreamingStacker(rm_min, rm_max)
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

if __name__ == '__main__':
    run_random_test()
