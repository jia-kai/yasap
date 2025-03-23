from .utils import F64Arr, logger

from scipy.optimize import least_squares
from jax.scipy.spatial.transform import Rotation
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import shapely

import typing

class FocalRotationHomographyEstimator:
    sensor_w: float
    focal_len: float
    img_w: int | None
    img_h: int | None
    K: F64Arr | None
    K_inv: F64Arr

    def __init__(self, sensor_w: float, focal_len: float):
        """
        Initialize the estimator with sensor parameters.

        Parameters:
        - sensor_w: Sensor width (e.g., in mm)
        - focal_len: Focal length (e.g., in mm)
        """
        self.sensor_w = sensor_w
        self.focal_len = focal_len
        self.img_w = None
        self.img_h = None
        self.K = None  # Intrinsic matrix will be computed once image size is set.

    def set_image_size(self, img_w: int, img_h: int, off_w: int, off_h: int):
        """
        Set the image size and compute the intrinsic matrix.
        If called multiple times, the new size must match the previously set size.

        Parameters:
        - img_w: Image width in pixels.
        - img_h: Image height in pixels.
        - off_w: Image cropping offset on the x axis
        - off_h: Image cropping offset on the y axis
        """
        if self.img_w is not None and self.img_h is not None:
            if self.img_w != img_w or self.img_h != img_h:
                raise ValueError("All image sizes must be the same.")
            # Otherwise, the image size is consistent; nothing to do.
        else:
            self.img_w = img_w
            self.img_h = img_h
            # Compute focal length in pixel units.
            f_pixel = self.focal_len * (img_w / self.sensor_w)
            cx = img_w / 2.0 - off_w
            cy = img_h / 2.0 - off_h
            self.K = np.array([[f_pixel,      0, cx],
                               [     0, f_pixel, cy],
                               [     0,      0,  1]], dtype=np.float64)
            self.K_inv = np.linalg.inv(self.K).astype(np.float64)

    def get_rvec_from_homography(self, H: F64Arr) -> F64Arr:
        """compute the rotation vector from a homography
        :param H: ``(3, 3)` homography
        :return: ``(3, )`` rotation vector
        """
        assert self.K is not None
        assert H.shape == (3, 3)
        K_inv = self.K_inv
        R_est = K_inv @ H @ self.K
        U, _, Vt = np.linalg.svd(R_est)
        R_init = U @ Vt
        assert np.linalg.det(R_init) > 0
        rvec, _ = cv2.Rodrigues(R_init)
        return rvec.ravel()

    def get_homography_from_rvec(self, rvec: F64Arr) -> F64Arr:
        assert self.K is not None
        assert rvec.shape == (3, )
        R, _ = cv2.Rodrigues(rvec)
        return self.K @ R @ self.K_inv

    def find_homography(self, pts1, pts2, method) -> typing.Optional[tuple[
            F64Arr, npt.NDArray[np.bool]]]:
        """
        Estimate the homography corresponding to a pure rotation
        (H = K * R * K⁻¹) by first obtaining an initial guess via RANSAC and then
        refining the rotation using SciPy’s Levenberg–Marquardt optimizer together with
        JAX for analytical Jacobian computation.

        Parameters:
        - pts1: Nx2 numpy array of source points (from image 1).
        - pts2: Nx2 numpy array of corresponding destination points (in image 2).

        Returns:
        - H_final: The refined homography matrix (3x3 numpy array) or None if
          failed
        - mask
        """
        if self.K is None:
            raise ValueError(
                "Image size has not been set. Please call set_image_size first.")

        pts1 = np.asarray(pts1, dtype=np.float64)
        pts2 = np.asarray(pts2, dtype=np.float64)

        # Estimate an initial homography
        H, mask = cv2.findHomography(pts1, pts2, method)
        if H is None:
            return

        # Use only the inlier correspondences.
        mask = mask.ravel().astype(bool)
        pts1_inliers = pts1[mask]
        pts2_inliers = pts2[mask]
        if len(pts1_inliers) < 4:
            logger.warning(
                f'too few inliers for homography: {len(pts1_inliers)}')
            return

        rvec, residual = self._refine_rvec(self.get_rvec_from_homography(H),
                                           pts1_inliers, pts2_inliers)

        # select lowest-error points and recompute
        pts_sel = []
        for i in np.argsort(residual):
            pts_sel.append(i)
            if len(pts_sel) < 5:
                continue
            area = shapely.MultiPoint(pts1_inliers[pts_sel]).convex_hull.area
            if area >= 100 * 100:
                break

        if len(pts_sel) < len(pts1_inliers):
            rvec, _ = self._refine_rvec(
                rvec, pts1_inliers[pts_sel], pts2_inliers[pts_sel])

            idx = np.argwhere(mask)
            mask[:] = False
            mask[idx[pts_sel]] = True

        H = self.get_homography_from_rvec(rvec)
        return H, mask

    def _refine_rvec(self, rvec_init: F64Arr,
                     pts1: F64Arr, pts2: F64Arr) -> tuple[F64Arr, F64Arr]:
        """refine the rotation vector

        :return: refined rotation vec, per-point residual
        """

        pts1_jax = jnp.array(pts1)
        pts2_jax = jnp.array(pts2)
        K_jax = jnp.array(self.K)
        K_inv_jax = jnp.array(self.K_inv)

        # Define the reprojection error function in JAX.
        # It takes a rotation vector (rvec) and returns a flattened residual vector.
        @jax.jit
        def reprojection_error(rvec):
            # Compute the rotation matrix from the rotation vector.
            R_mat = Rotation.from_rotvec(rvec).as_matrix()  # shape (3,3)
            # Construct the homography H = K * R * K⁻¹.
            H_refined = K_jax @ R_mat @ K_inv_jax
            # Convert pts1 to homogeneous coordinates.
            ones = jnp.ones((pts1_jax.shape[0], 1))
            pts1_h = jnp.concatenate([pts1_jax, ones], axis=1)  # shape (N, 3)
            pts1_h = pts1_h.T  # shape (3, N)
            # Apply the homography.
            pts_transformed = H_refined @ pts1_h  # shape (3, N)
            # Normalize (perspective division).
            pts_transformed = pts_transformed[:2] / pts_transformed[2:]
            pts_transformed = pts_transformed.T  # shape (N, 2)
            # Return the residual vector.
            return jnp.ravel(pts_transformed - pts2_jax)

        # Wrap the JAX reprojection error for SciPy's least_squares.
        def fun_np(rvec):
            return np.array(reprojection_error(jnp.array(rvec)))

        # Compute the Jacobian of the reprojection error using jax.jacrev.
        jac_fun = jax.jit(jax.jacrev(reprojection_error))
        def jac_np(rvec):
            return np.array(jac_fun(jnp.array(rvec)))

        # Use SciPy's least_squares with the LM algorithm.
        res = least_squares(
            fun_np, rvec_init, jac=jac_np,  # type: ignore
            method='lm')
        assert res.success
        residual = np.linalg.norm(fun_np(res.x).reshape(pts1.shape),
                                  axis=1)
        return res.x, residual
