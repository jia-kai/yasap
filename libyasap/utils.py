import cv2
import numpy as np

import logging
import sys

logger = logging.getLogger('yasap')

# see https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"
    format_str = ('%(relativeCreated).2fs - %(name)s - %(levelname)s - '
              '%(message)s (%(filename)s:%(lineno)d)')

    FORMATS = {
        logging.DEBUG: cyan + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        record.relativeCreated = record.relativeCreated / 1000
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(file_path=None):
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    if file_path:
        chf = logging.FileHandler(file_path, 'w')
        chf.setLevel(logging.DEBUG)
        chf.setFormatter(logging.Formatter(CustomFormatter.format_str))
        logger.addHandler(chf)

def perspective_transform(H: np.ndarray, pts: np.ndarray):
    # compute perspective transform and compare results with opencv to check
    # that my understanding is correct (not sure why they need input shape (1, n,
    # 2)
    assert pts.ndim == 2 and pts.shape[1] == 2, pts.shape
    r_cv = cv2.perspectiveTransform(pts[np.newaxis], H)[0]
    t = (H @ np.concatenate([pts, np.ones_like(pts[:, :1])], axis=1).T).T
    r = np.ascontiguousarray(t[:, :2] / t[:, 2:], dtype=np.float32)
    assert np.allclose(r, r_cv)
    return r

def in_bounding_box(pts: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    """return a mask of weather each point lies in a bounding box (x1 and y1
    excluded)

    :param pts: (n, 2) array of the (x, y) coordinates
    """
    assert pts.ndim == 2 and pts.shape[1] == 2

    m0 = np.all(pts >= np.array([[x0, y0]], dtype=pts.dtype), axis=1)
    m1 = np.all(pts < np.array([[x1, y1]], dtype=pts.dtype), axis=1)
    return np.logical_and(m0, m1)

def pairwise_l2_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """input: (n, 2) coordinates; output: (n, ) distances"""
    assert x.shape == y.shape and x.ndim == 2 and x.shape[1] == 2, (
        x.shape, y.shape)
    return np.sqrt(((x - y)**2).sum(axis=1))

def avg_l2_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(pairwise_l2_dist(x, y).mean())

def max_l2_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(pairwise_l2_dist(x, y).max())

def find_homography(src, dst, method):
    """:return: H, avg dist, max dist"""
    assert src.shape == dst.shape and src.shape[0] >= 4 and src.ndim == 2, (
        src.shape, dst.shape)
    H, _ = cv2.findHomography(src, dst, method)
    t = perspective_transform(H, src)
    return H, avg_l2_dist(t, dst), max_l2_dist(t, dst)

def disp_img(title: str, img: np.ndarray, wait=True):
    """display an image while handling the keys"""
    if img.dtype == np.bool_:
        img = (img * 255).astype(np.uint8)

    scale = 1000 / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    if wait:
        while True:
            key = chr(cv2.waitKey(-1) & 0xff)
            if key == 'q':
                print('exit')
                sys.exit()
            if key == 'w':
                return
            print(f'press q to exit, w to close this window; got {key=}')

def disp_match_pairs(img0, p0, img1, p1):
    """display a pair of imgs with their matched point pairs"""
    assert (img0.shape == img1.shape and img0.ndim == 2 and
            img0.dtype == img1.dtype)
    if img0.dtype == np.float32:
        img0 = np.clip(img0 * 255, 0, 255).astype(np.uint8)
        img1 = np.clip(img1 * 255, 0, 255).astype(np.uint8)
    assert img0.dtype == np.uint8
    h, w = img0.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = img0
    img[:, :, 1] = img1
    color = np.random.randint(0, 255, (len(p0), 3))
    for x, i, j in zip(color, p0, p1):
        a, b = map(int, i.ravel())
        c, d = map(int, j.ravel())
        cv2.line(img, (a, b), (c, d), x.tolist(), 5)
        # cv2.circle(img, (c, d), 5, x.tolist(), -1)
    disp_img('match', img)

def precise_quantile(x: np.ndarray, q: float):
    """compute precise value of quantile"""
    # note: np.quantile is an estimation
    x = x.flatten()
    cut = min(int(len(x) * q), len(x) - 1)
    return np.partition(x, cut)[cut]

def get_mask_for_largest(x: np.ndarray, n: int):
    """get a mask for keeping the ``n`` largest values in ``x``"""
    assert x.ndim == 1
    k = x.shape[0] - n
    thresh = np.partition(x, k)[k]
    mask = x >= thresh
    assert (g := np.sum(mask)) == n, (g, n)
    return mask

def save_img(img: np.ndarray, fpath: str):
    """save image to file"""
    assert img.dtype == np.float32
    if fpath.endswith('.npy'):
        np.save(fpath, img)
        return
    img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    succ = cv2.imwrite(fpath, img)
    assert succ, f'failed to write image {fpath}'

def read_img(fpath: str) -> np.ndarray:
    """read an image as float32 format"""
    if fpath.endswith('.npy'):
        img = np.load(fpath)
        assert img.ndim in [2, 3] and img.dtype == np.float32
        return img
    img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    assert img is not None, f'failed to read {fpath}'
    img = img.astype(np.float32) / np.float32(np.iinfo(img.dtype).max)
    return img

def format_relative_aa_bbox(pts: np.ndarray, img_shape: tuple[int]) -> str:
    """compute the axis-aligned bounding box for a set of points relative to an
    image shape and format it as a string

    :param img_shape: (h, w, ch) or (h, w)
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    h, w = img_shape[:2]
    x0, y0 = np.min(pts, axis=0) / [w, h]
    x1, y1 = np.max(pts, axis=0) / [w, h]
    return f'({x0:.2f},{y0:.2f};w={x1-x0:.2f},h={y1-y0:.2f})'
