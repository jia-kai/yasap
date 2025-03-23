import cv2
import numpy as np
import numpy.typing as npt

import logging
import sys
import typing

if typing.TYPE_CHECKING:
    from .rot_homo import FocalRotationHomographyEstimator

type F64Arr = npt.NDArray[np.float64]
type F32Arr = npt.NDArray[np.float32]
type U8Arr = npt.NDArray[np.uint8]

logger = logging.getLogger('yasap')

g_homography_estimator: typing.Optional[
    'FocalRotationHomographyEstimator'] = None
"""if not None, will be used for homography"""

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

def set_use_restricted_transform(sensor_w: float, focal_len: float):
    from .rot_homo import FocalRotationHomographyEstimator
    global g_homography_estimator
    g_homography_estimator = FocalRotationHomographyEstimator(
        sensor_w, focal_len)
    logger.info('use focal rotation homography estimator:'
                f' {sensor_w=:.1f} {focal_len=:.1f}')

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

def med_l2_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(pairwise_l2_dist(x, y)))

def max_l2_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(pairwise_l2_dist(x, y).max())

def find_homography(src, dst, method, need_mask: bool = False):
    """:return: H, err, max dist, [optinal mask]"""
    assert src.shape == dst.shape and src.shape[0] >= 4 and src.ndim == 2, (
        src.shape, dst.shape)
    if g_homography_estimator is not None:
        if method == 0: # least square
            method = cv2.LMEDS
        assert method in (cv2.RANSAC, cv2.LMEDS)
        H, mask = g_homography_estimator.find_homography(src, dst, method)
        assert H is not None
    else:
        H, mask = cv2.findHomography(src, dst, method)
        mask = mask.ravel().astype(bool)
    t = perspective_transform(H, src)
    ret = H, avg_l2_dist(t[mask], dst[mask]), max_l2_dist(t, dst)
    if need_mask:
        return *ret, mask
    return ret

def disp_img(title: str, img: np.ndarray, wait=True, max_size: int=1000):
    """display an image while handling the keys"""
    print(f'display {title}: shape={img.shape}, dtype={img.dtype}')
    if img.dtype == np.bool_:
        img = (img * 255).astype(np.uint8)

    if img.dtype.kind == 'f':
        img = img_as_u8(img)

    scale = max_size / max(img.shape)
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
    else:
        cv2.waitKey(1)

def img_as_u8(img: F32Arr) -> U8Arr:
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def disp_match_pairs(img0, p0, img1, p1, *, name: str='match', wait: bool=True):
    """display a pair of imgs with their matched point pairs"""
    assert (img0.shape == img1.shape and img0.ndim == 2 and
            img0.dtype == img1.dtype)
    if img0.dtype == np.float32:
        img0 = img_as_u8(img0)
        img1 = img_as_u8(img1)
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
    disp_img(name, img, wait=wait)

def precise_quantile(x: np.ndarray, q: float):
    """compute precise value of quantile"""
    # note: np.quantile is an estimation
    x = x.flatten()
    cut = min(int(len(x) * q), len(x) - 1)
    return np.partition(x, cut)[cut]

def get_mask_for_largest(x: np.ndarray, n: int):
    """get a mask for keeping the ``n`` largest values in ``x``"""
    assert x.ndim == 1
    x = x + np.random.normal(scale=1e-8, size=x.shape)  # random tie breaking
    k = x.shape[0] - n
    thresh = np.partition(x, k)[k]
    mask = x >= thresh
    assert (g := np.sum(mask)) == n, (g, n)
    return mask

def write_exr_f32(img: np.ndarray, fpath: str):
    import OpenEXR
    import Imath
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.float32
    header = OpenEXR.Header(img.shape[1], img.shape[0])
    f32_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, f32_chan) for c in 'RGB'])
    out = OpenEXR.OutputFile(fpath, header)
    B = (img[:,:,0]).tobytes()
    G = (img[:,:,1]).tobytes()
    R = (img[:,:,2]).tobytes()
    out.writePixels({'R' : R, 'G' : G, 'B' : B})
    out.close()

def write_tiff_f32(img: np.ndarray, fpath: str):
    import tifffile
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.float32
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tifffile.imwrite(fpath, img, photometric='rgb')

def save_img(img: np.ndarray, fpath: str):
    """save image to file"""
    assert img.dtype == np.float32
    if fpath.endswith('.npy'):
        np.save(fpath, img)
        return
    if fpath.endswith('.exr'):
        write_exr_f32(img, fpath)
        return
    if fpath.endswith('.tiff'):
        write_tiff_f32(img, fpath)
        return

    img = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    succ = cv2.imwrite(fpath, img)
    assert succ, f'failed to write image {fpath}'

def read_img(fpath: str, *, user_wb=[]) -> F32Arr:
    """read an image as 3 channel float32 format
    :param user_wb: white balance coefficients; if empty, it will be modified
        inplace (so all images use the same coefficients if left as the default
        list object)
    """
    fpath_ext = fpath.lower()[fpath.rfind('.')+1:]
    if fpath_ext == 'npy':
        img = np.load(fpath)
        assert img.ndim in [2, 3] and img.dtype == np.float32
        return img
    if fpath_ext in ['fit', 'fits']:
        from astropy.io import fits
        with fits.open(fpath) as fin:
            data = fin['PRIMARY'].data
            if data.dtype == np.uint16:
                data = data.astype(np.float32) / (2**16-1)
            else:
                raise RuntimeError(f'unhandled data type {data.dtype}')
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        assert data.ndim == 3, f'invalid shape: {data.shape}'
        return data
    if fpath_ext in ['nef']:
        import rawpy
        with rawpy.imread(fpath) as raw:
            if not user_wb:
                user_wb[:] = raw.camera_whitebalance
                logger.info(f'use white balance: {user_wb}')
            assert len(user_wb) == 4
            rgb = raw.postprocess(
                # half_size=True,
                # use linear since noise/artifacts are dealt with by stacking
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                median_filter_passes=0, user_wb=user_wb,
                output_color=rawpy.ColorSpace.Rec2020,
                output_bps=16, user_flip=0,
                no_auto_scale=True, no_auto_bright=True,
            )
            white_level = raw.white_level
        assert rgb.dtype == np.uint16
        bgr = rgb[:, :, ::-1].astype(np.float32)
        bgr /= white_level
        bgr = np.clip(bgr, 0, 1, out=bgr)
        return bgr

    img = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    assert img is not None, f'failed to read {fpath}'
    if img.dtype != np.float32:
        assert np.issubdtype(img.dtype, np.integer), (
            f'unhandled dtype {img.dtype}')
        img = img.astype(np.float32) / np.float32(np.iinfo(img.dtype).max)
    return img

def format_relative_aa_bbox(pts: np.ndarray, img_shape: tuple[int, ...]) -> str:
    """compute the axis-aligned bounding box for a set of points relative to an
    image shape and format it as a string

    :param img_shape: (h, w, ch) or (h, w)
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    h, w = img_shape[:2]
    x0, y0 = np.min(pts, axis=0) / [w, h]
    x1, y1 = np.max(pts, axis=0) / [w, h]
    return f'({x0:.2f},{y0:.2f};w={x1-x0:.2f},h={y1-y0:.2f})'
