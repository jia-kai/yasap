from .utils import read_img, disp_img, logger, save_img, F32Arr, U8Arr

import cv2
import numpy as np
import numpy.typing as npt
import typing

import importlib.util
import sys

def auto_white_balance(img: F32Arr, *,
                       saturation_thresh=0.98,
                       quantile=0.05,
                       max_drop=0.03,
                       verbose: bool=False,
                       mask: typing.Optional[U8Arr] = None,
                       ) -> F32Arr:
    """set automatic balance for astro images by assuming stars are white

    :param saturation_thresh: only work with pixels whose all channels are below
        this threshold
    :param quantile: top quantile of pixels to be selected
    :param max_drop: max drop of chnannel values for selected pixels
    """

    masked_img = cv2.GaussianBlur(img, (5, 5), 0)
    if mask is not None:
        masked_img = np.where(mask[:, :, np.newaxis] > 127, masked_img, 0)

    chl_max = masked_img.reshape((-1, 3)).max(axis=0)
    logger.info(f'channel max: {chl_max}')
    sel_mask = np.zeros(img.shape[:2], dtype=bool)
    sel_mask[img.max(axis=2) < saturation_thresh] = True
    chl_mask = np.ones_like(sel_mask)
    for i in range(3):
        chl = masked_img[:, :, i]
        chl_th = max(float(np.quantile(chl[sel_mask], quantile)),
                     chl_max[i] - max_drop)
        chl_mask &= chl > chl_th

    sel_mask &= chl_mask
    nr_sel = np.count_nonzero(sel_mask)
    logger.info(f'num selected pixels: {nr_sel}')
    if not nr_sel:
        raise ValueError('no pixel selected')
    sub = img[np.broadcast_to(sel_mask[:, :, np.newaxis], img.shape)].reshape(
        (-1, 3)
    )
    mean = np.mean(sub, axis=0)
    coeff = np.array([1, mean[0] / mean[1], mean[0] / mean[2]])
    coeff /= coeff.max()
    logger.info(f'wb coeff: {coeff}')
    out = img * coeff[np.newaxis, np.newaxis, :]
    if verbose:
        disp_img('input', img, wait=False)
        disp_img('output', out)
    return out

def equalize_histogram(
    img: npt.NDArray[np.float32],
    quantile_scales: tuple[list[float], list[float]], *,
    bins=16,
    clip_limit=2.5,
    verbose: bool=False) -> npt.NDArray[np.float32]:
    """equalize partial histogram of an image

    :param bins: number of bins for histogram
    :param quantile_scales: quantile endpoints, new scale of values in this
        interval
    :param clip_limit: PDF clip limit compared to uniform distribution
    """

    img = img.astype(np.float32)

    def modify_channel(chl, low, high, rescale_size):
        """
        modify a channel to equalize [low, high) to [low, low + rescale_size)

        :param chl: the channel to be modified inplace
        :param val_range: range of values in the channel
        :return: mask, shift of future highs
        """
        mask = (chl >= low) & (chl < high)
        cnt = np.count_nonzero(mask)
        logger.info(f'interval {low:.3f}-{high:.3f}: {cnt} pixels')
        if cnt < bins:
            mask[:] = 0
            return mask, 0
        sub = chl[mask]
        bin_edges = np.linspace(low, high, bins + 1, endpoint=True)
        pdf, _ = np.histogram(sub, bins=bin_edges, range=(low, high))
        pdf = pdf.astype(np.float64)
        pdf /= pdf.sum()
        top_pdf = np.maximum(pdf - clip_limit / bins, 0).sum()
        pdf = np.minimum(pdf, clip_limit / bins, out=pdf)
        pdf += top_pdf / bins
        cdf = np.cumsum(pdf)
        cdf = np.concatenate(([0], cdf), dtype=np.float32)
        dst_low, dst_high = low, high
        dst_high = dst_low + (high - low) * rescale_size
        chl[chl >= high] += dst_high - high

        remap = cdf * (dst_high - dst_low)
        remap += dst_low

        if verbose >= 2:
            import matplotlib.pyplot as plt
            plt.plot(bin_edges, remap)
            plt.show()

        chl[mask] = np.interp(sub, bin_edges, remap)

        return mask, dst_high - high

    L_TOT_SCALE = 100
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    new_l = lab[:, :, 0].copy()
    quantile_edges, rescale_sizes = quantile_scales
    assert len(quantile_edges) == len(rescale_sizes) + 1
    qv = np.quantile(new_l, quantile_edges)

    if verbose:
        disp_img('input', img, wait=False)

    for i in range(len(rescale_sizes)):
        mask, delta = modify_channel(new_l, qv[i], qv[i + 1], rescale_sizes[i])
        qv[i+1:] += delta
        if verbose:
            disp_img(f'mask-{i}', mask, wait=False)

    new_l *= L_TOT_SCALE / new_l.max()
    lab[:, :, 0] = new_l
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    if verbose:
        disp_img('output', out)
    return out


def remove_bg(img: npt.NDArray[np.float32], *,
              brightness_channel: bool = False,
              min_rank=0.02, gaussian_frac=0.02,
              verbose: bool=False) -> npt.NDArray[np.float32]:
    """remove background of an image

    :param brightness_channel: only remove the background on the brightness
        channel
    :param min_rank: rank of minimal value to be selected
    :param gaussian_frac: Gaussian kernel size for row smoothing relative to
        image height; negative for global min
    :param mask_path: path to mask image
    """
    img_orig = img.copy()

    if brightness_channel:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img = lab[:, :, 0]

    min_rank = int(round(min_rank * img.shape[1]))
    row_min: npt.NDArray[np.float32] = np.expand_dims(
        np.partition(img, min_rank, axis=1)[:, min_rank],
        1)
    if gaussian_frac < 0:
        row_min[:] = np.mean(row_min, axis=0, keepdims=True)
    else:
        ksize = int(gaussian_frac * img.shape[0])
        ksize += (ksize + 1) % 2
        row_min = cv2.GaussianBlur(row_min, (1, ksize), 0)

    img -= row_min

    if brightness_channel:
        lab[:, :, 0] = img
        img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        row_min = row_min[:, :, np.newaxis] / 100   # for verbose

    img = np.clip(img, 0, 1, out=img)

    logger.info(f'bg mean: {np.mean(row_min):.3g}')
    if verbose:
        disp_img('bg', np.broadcast_to(row_min, img.shape), wait=False)
        disp_img('input', img_orig, wait=False)
        disp_img('output', img, wait=True)

    return img

def run_postprocess(script_path, input_path, output_path):
    img = read_img(input_path)
    assert img.dtype == np.float32 and img.ndim == 3 and img.shape[2] == 3
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    module_name = 'yasap_user_postprocess'
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for k, v in globals().items():
        if not k.startswith('_') and k != 'run_postprocess':
            setattr(module, k, v)
    out = module.main(img).astype(np.float32)
    assert out.ndim == 3 and out.shape[2] == 3
    save_img(out, output_path)

def blend_with_mask(img: npt.NDArray[np.float32],
                    other_img_path: str, mask_path: str,
                    *, sigma=30, verbose: bool=False) -> npt.NDArray[np.float32]:
    """blend an image with a mask
    :param img: the image to be blended
    :param other_img_path: the other image to be blended
    :param mask_path: the mask image; 0 for first image, 1 for second image
    :param sigma: sigma for Gaussian blur of the mask
    """
    other_img = np.load(other_img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    mask /= mask.max()
    mask = np.expand_dims(mask, 2)
    out = img * (1 - mask) + other_img * mask
    if verbose:
        disp_img('input', img, wait=False)
        disp_img('other', other_img, wait=False)
        disp_img('mask', mask, wait=False)
        disp_img('output', out)
    return out
