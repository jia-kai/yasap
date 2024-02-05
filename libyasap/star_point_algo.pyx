# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport M_PI

def display_queue(np.ndarray[np.int32_t, ndim=2] queue):
    import cv2
    queue = queue - queue.min(axis=0, keepdims=True)
    h, w = queue.max(axis=0) + 1
    img = np.zeros((h, w), dtype=np.uint8)
    cdef int i
    for i in range(queue.shape[0]):
        img[queue[i, 0], queue[i, 1]] = 255
    cv2.imshow('queue', img)
    cv2.waitKey(0)


@cython.boundscheck(False)
@cython.wraparound(False)
def find_star_centers(
    np.ndarray[np.float32_t, ndim=2] img,
    int min_area, float min_bbox_ratio) -> tuple[np.ndarray, float]:
    """find robust star centers
    :param img: the image, where dark pixels have been thresholded to 0

    :return:
        * (n, 3) array for total brightness and xy coordinates in opencv
        frame
        * a score indicating how circular the stars are
    """
    cdef np.ndarray[np.npy_bool, ndim=2] visited
    cdef np.ndarray[np.int32_t, ndim=2] queue
    cdef int i0, j0, i, j, di, dj, i1, j1, imin, imax, jmin, jmax
    cdef int bbox_low, bbox
    cdef double sum_val, wsum_i, wsum_j
    cdef list result = [], quality_scores = []
    cdef unsigned qh, qsize
    dij = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = np.zeros_like(img, dtype=np.bool_)
    queue = np.empty((img.shape[0] * img.shape[1], 2), dtype=np.int32)

    result = []
    for i0 in range(img.shape[0]):
        for j0 in range(img.shape[1]):
            if visited[i0, j0] or img[i0, j0] == 0:
                continue

            visited[i0, j0] = True

            qh = 0
            qsize = 1
            queue[0] = [i0, j0]

            imin = imax = i0
            jmin = jmax = j0
            sum_val = wsum_i = wsum_j = 0
            while qh < qsize:
                i, j = queue[qh]
                qh += 1
                sum_val += img[i, j]
                wsum_i += img[i, j] * i
                wsum_j += img[i, j] * j
                imin = min(i, imin)
                imax = max(i, imax)
                jmin = min(j, jmin)
                jmax = max(j, jmax)
                for di, dj in dij:
                    i1 = i + di
                    j1 = j + dj
                    if (0 <= i1 < img.shape[0] and 0 <= j1 < img.shape[1]
                            and not visited[i1, j1]
                            and img[i1, j1]):
                        visited[i1, j1] = True
                        queue[qsize] = [i1, j1]
                        qsize += 1

            bbox_low = min(imax - imin + 1, jmax - jmin + 1)**2
            bbox = max(imax - imin + 1, jmax - jmin + 1)**2
            if bbox_low >= 6 * 6:
                # a first-order estimate of the circularity
                quality_scores.append(1 - abs(qsize / (M_PI * bbox / 4) - 1))

                if False:
                    print(quality_scores[len(quality_scores)-1])
                    display_queue(queue[:qsize])

            if qsize >= max(min_area, min_bbox_ratio * bbox):
                result.append((sum_val, wsum_j / sum_val, wsum_i / sum_val))

    return np.array(result, dtype=np.float32), np.median(quality_scores)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_match_mask(int nr_dst, config,
                   np.ndarray[np.int32_t, ndim=1] match_idx,
                   np.ndarray[np.float32_t, ndim=1] dist):
    cdef np.ndarray[np.npy_bool, ndim=1] mask, dst_used
    cdef np.ndarray[np.npy_long, ndim=1] sorted_idx
    cdef int i
    cdef float prev_dist, max_dist_jump

    max_dist_jump = config.star_point_icp_max_dist_jump

    mask = np.zeros_like(dist, dtype=np.bool_)
    dst_used = np.zeros(nr_dst, dtype=np.bool_)
    sorted_idx = np.argsort(dist)

    prev_dist = -1
    for i in sorted_idx:
        didx = match_idx[i]
        assert 0 <= didx < nr_dst

        if not dst_used[didx]:
            if prev_dist == -1:
                prev_dist = dist[i]
            else:
                if dist[i] > max(prev_dist, 1) * max_dist_jump:
                    break
                prev_dist = dist[i]

            dst_used[didx] = True
            mask[i] = True

    return mask
