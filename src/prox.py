
"""
prox.py

Proximal primitives for the computational-statistics sparse
regression pipeline.

Conventions
-----------
- Inputs are one-dimensional NumPy arrays unless stated otherwise.
- threshold >= 0.
- The hard sum-to-zero constraint is

      C = {x in R^p : 1^T x = 0}.

- No routine silently rescales or standardises its input.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "soft_threshold",
    "project_sum_zero",
    "prox_l1_sum_zero",
]


def _as_1d_float_array(x) -> np.ndarray:
    """Convert input to a finite one-dimensional float array."""
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError("Input must be a one-dimensional array.")

    if arr.size == 0:
        raise ValueError("Input array must be non-empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values.")

    return arr


def soft_threshold(x, threshold):
    r"""
    Componentwise soft-thresholding.

    Computes

        S_t(x)_j = sign(x_j) * max(|x_j| - t_j, 0),

    where `threshold` may be a nonnegative scalar or any
    nonnegative array broadcastable to the shape of `x`.

    Parameters
    ----------
    x : array_like
        Input array.
    threshold : float or array_like
        Nonnegative threshold(s).

    Returns
    -------
    ndarray
        Soft-thresholded array.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(threshold, dtype=float)

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")

    if not np.all(np.isfinite(t)):
        raise ValueError("threshold contains non-finite values.")

    if np.any(t < 0):
        raise ValueError("threshold must be nonnegative.")

    try:
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
    except ValueError as exc:
        raise ValueError(
            "threshold must be scalar or broadcastable to x."
        ) from exc


def project_sum_zero(x):
    r"""
    Orthogonal projection onto

        C = {u in R^p : 1^T u = 0}.

    For x in R^p,

        P_C(x) = x - mean(x) * 1.

    Parameters
    ----------
    x : array_like, shape (p,)
        Input vector.

    Returns
    -------
    ndarray, shape (p,)
        Euclidean projection of x onto C.
    """
    x = _as_1d_float_array(x)
    return x - np.mean(x)


def prox_l1_sum_zero(
    x,
    threshold,
    *,
    tol=1e-12,
    max_iter=200,
    return_dual=False,
):
    r"""
    Proximal map of threshold * ||u||_1 under a hard sum-zero constraint.

    Computes

        argmin_u  0.5 * ||u - x||_2^2 + threshold * ||u||_1
        subject to
                  1^T u = 0.

    The KKT conditions imply

        u_j = S_threshold(x_j - nu),

    where the scalar dual variable nu satisfies

        sum_j S_threshold(x_j - nu) = 0.

    The latter is a continuous monotone equation and is solved
    robustly by bisection.

    Parameters
    ----------
    x : array_like, shape (p,)
        Input vector.
    threshold : float
        Nonnegative l1 proximal threshold.
    tol : float, optional
        Numerical tolerance for the scalar root problem.
    max_iter : int, optional
        Maximum number of bisection iterations.
    return_dual : bool, optional
        If True, return (u, nu).

    Returns
    -------
    u : ndarray, shape (p,)
        Constrained proximal point.

    nu : float, optional
        KKT multiplier associated with the equality constraint.

    Notes
    -----
    For threshold = 0, this reduces exactly to the Euclidean
    projection onto the sum-zero hyperplane:

        prox_{0 ||.||_1 + iota_C}(x) = P_C(x).
    """
    x = _as_1d_float_array(x)

    threshold = float(threshold)
    tol = float(tol)

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number.")

    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    # Special case: pure Euclidean projection.
    if threshold == 0.0:
        nu = float(np.mean(x))
        u = x - nu

        if return_dual:
            return u, nu
        return u

    def root_function(nu):
        return float(np.sum(soft_threshold(x - nu, threshold)))

    # A guaranteed bracket:
    #
    # nu_lo = min(x) - threshold  => all soft-thresholded terms >= 0
    # nu_hi = max(x) + threshold  => all soft-thresholded terms <= 0
    nu_lo = float(np.min(x) - threshold)
    nu_hi = float(np.max(x) + threshold)

    f_lo = root_function(nu_lo)
    f_hi = root_function(nu_hi)

    # These should follow analytically; retain defensive checks.
    if f_lo < -tol or f_hi > tol:
        raise RuntimeError(
            "Failed to bracket the dual root in prox_l1_sum_zero."
        )

    # If an endpoint is already a root.
    if abs(f_lo) <= tol:
        nu = nu_lo
        u = soft_threshold(x - nu, threshold)

        if return_dual:
            return u, nu
        return u

    if abs(f_hi) <= tol:
        nu = nu_hi
        u = soft_threshold(x - nu, threshold)

        if return_dual:
            return u, nu
        return u

    # Monotone bisection.
    nu = 0.5 * (nu_lo + nu_hi)

    for _ in range(max_iter):
        nu = 0.5 * (nu_lo + nu_hi)
        f_mid = root_function(nu)

        if abs(f_mid) <= tol:
            break

        # The root function is non-increasing.
        if f_mid > 0.0:
            nu_lo = nu
        else:
            nu_hi = nu

        # Also terminate if the dual bracket is tiny.
        bracket_scale = max(
            1.0,
            abs(nu_lo),
            abs(nu_hi),
        )

        if (nu_hi - nu_lo) <= tol * bracket_scale:
            nu = 0.5 * (nu_lo + nu_hi)
            break

    u = soft_threshold(x - nu, threshold)

    # Remove only floating-point-level violation of the equality
    # constraint. We do NOT apply an additional projection here,
    # because that would generally change the proximal solution.
    residual = float(np.sum(u))

    if abs(residual) > 100.0 * tol * max(1.0, x.size):
        raise RuntimeError(
            "Bisection did not solve the sum-zero constraint "
            f"to the requested accuracy; residual={residual:.3e}."
        )

    if return_dual:
        return u, float(nu)

    return u
