
"""
model.py

Core model algebra for the computational-statistics sparse
regression pipeline.

Statistical model
-----------------
    y = X beta + epsilon,
    epsilon ~ N(0, sigma2 I).

Optional soft affine condition
------------------------------
    a^T beta = b

is represented through the Gaussian pseudo-observation term

    (a^T beta - b)^2 / (2 tau_c^2).

Laplace penalty
---------------
    h_theta(beta) = theta * ||beta||_1.

Moreau convention
-----------------
The whole nonsmooth term h_theta is smoothed:

    h_theta^lambda(beta)
      = min_u {
          theta ||u||_1
          + ||u-beta||^2 / (2 lambda)
        }.

Hence

    grad h_theta^lambda(beta)
      = [beta - prox_{lambda theta ||.||_1}(beta)] / lambda.

No routine in this module silently centers, scales, or otherwise
modifies the supplied statistical model.
"""

from __future__ import annotations

import numpy as np

from prox import soft_threshold


__all__ = [
    "gaussian_nll",
    "gaussian_grad",
    "soft_affine_nll",
    "soft_affine_grad",
    "smooth_hessian",
    "smooth_lipschitz",
    "sum_zero_lipschitz",
    "laplace_penalty",
    "laplace_moreau_value",
    "laplace_moreau_grad",
    "posterior_objective",
    "smoothed_posterior_objective",
]


# ============================================================
# Validation helpers
# ============================================================

def _as_1d_finite(x, name):
    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def _as_2d_finite(X, name="X"):
    arr = np.asarray(X, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have non-zero dimensions.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")

    return arr


def _validate_regression(beta, y, X, sigma2):
    beta = _as_1d_finite(beta, "beta")
    y = _as_1d_finite(y, "y")
    X = _as_2d_finite(X, "X")

    n, p = X.shape

    if y.shape[0] != n:
        raise ValueError(
            f"y has length {y.shape[0]}, but X has {n} rows."
        )

    if beta.shape[0] != p:
        raise ValueError(
            f"beta has length {beta.shape[0]}, but X has {p} columns."
        )

    sigma2 = float(sigma2)

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("sigma2 must be finite and strictly positive.")

    return beta, y, X, sigma2


def _validate_affine(a, b, tau_c, p):
    a = _as_1d_finite(a, "a")

    if a.shape[0] != p:
        raise ValueError(
            f"a has length {a.shape[0]}, but beta has length {p}."
        )

    b = float(b)
    tau_c = float(tau_c)

    if not np.isfinite(b):
        raise ValueError("b must be finite.")

    if not np.isfinite(tau_c) or tau_c <= 0.0:
        raise ValueError("tau_c must be finite and strictly positive.")

    return a, b, tau_c


# ============================================================
# Gaussian likelihood
# ============================================================

def gaussian_nll(beta, y, X, sigma2):
    r"""
    Gaussian negative log-likelihood up to an additive constant:

        f(beta)
          = ||y - X beta||_2^2 / (2 sigma2).
    """
    beta, y, X, sigma2 = _validate_regression(
        beta, y, X, sigma2
    )

    residual = X @ beta - y

    return 0.5 * float(residual @ residual) / sigma2


def gaussian_grad(beta, y, X, sigma2):
    r"""
    Gradient of the Gaussian negative log-likelihood:

        grad f(beta)
          = X^T (X beta - y) / sigma2.
    """
    beta, y, X, sigma2 = _validate_regression(
        beta, y, X, sigma2
    )

    return X.T @ (X @ beta - y) / sigma2


# ============================================================
# Soft affine likelihood
# ============================================================

def soft_affine_nll(beta, y, X, sigma2, a, b, tau_c):
    r"""
    Gaussian negative log-likelihood with soft affine condition:

        f(beta)
          = ||y-X beta||^2 / (2 sigma2)
            + (a^T beta - b)^2 / (2 tau_c^2).
    """
    beta, y, X, sigma2 = _validate_regression(
        beta, y, X, sigma2
    )

    a, b, tau_c = _validate_affine(
        a, b, tau_c, beta.size
    )

    residual = X @ beta - y
    affine_residual = float(a @ beta - b)

    return (
        0.5 * float(residual @ residual) / sigma2
        + 0.5 * affine_residual**2 / tau_c**2
    )


def soft_affine_grad(beta, y, X, sigma2, a, b, tau_c):
    r"""
    Gradient of the soft-affine negative log-likelihood:

        grad f(beta)
          = X^T(X beta-y)/sigma2
            + a(a^T beta-b)/tau_c^2.
    """
    beta, y, X, sigma2 = _validate_regression(
        beta, y, X, sigma2
    )

    a, b, tau_c = _validate_affine(
        a, b, tau_c, beta.size
    )

    return (
        X.T @ (X @ beta - y) / sigma2
        + a * (a @ beta - b) / tau_c**2
    )


# ============================================================
# Smooth Hessian and Lipschitz constants
# ============================================================

def smooth_hessian(X, sigma2, *, a=None, tau_c=None):
    r"""
    Hessian of the smooth likelihood term.

    Unconstrained:
        H = X^T X / sigma2.

    Soft affine:
        H = X^T X / sigma2 + a a^T / tau_c^2.

    Parameters
    ----------
    X : array_like, shape (n,p)
    sigma2 : float
    a : array_like, optional
        Affine normal vector.
    tau_c : float, optional
        Soft-constraint standard deviation.

    Returns
    -------
    H : ndarray, shape (p,p)
    """
    X = _as_2d_finite(X, "X")
    sigma2 = float(sigma2)

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("sigma2 must be finite and strictly positive.")

    H = (X.T @ X) / sigma2

    if a is None and tau_c is None:
        return H

    if (a is None) != (tau_c is None):
        raise ValueError(
            "a and tau_c must either both be supplied or both be None."
        )

    a = _as_1d_finite(a, "a")

    if a.size != X.shape[1]:
        raise ValueError(
            "a must have one entry for each column of X."
        )

    tau_c = float(tau_c)

    if not np.isfinite(tau_c) or tau_c <= 0.0:
        raise ValueError("tau_c must be finite and strictly positive.")

    H = H + np.outer(a, a) / tau_c**2

    return H


def smooth_lipschitz(X, sigma2, *, a=None, tau_c=None):
    r"""
    Exact Lipschitz constant of the gradient of the smooth
    quadratic likelihood term.

    It is computed through an augmented design matrix rather
    than explicitly forming X^T X:

        B = X / sqrt(sigma2)

    in the unconstrained case, and

        B = [ X / sqrt(sigma2) ]
            [ a^T / tau_c    ]

    in the soft-affine case.

    Since H = B^T B,

        L_f = lambda_max(H) = ||B||_2^2.
    """
    X = _as_2d_finite(X, "X")

    sigma2 = float(sigma2)

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("sigma2 must be finite and strictly positive.")

    B = X / np.sqrt(sigma2)

    if a is not None or tau_c is not None:
        if (a is None) != (tau_c is None):
            raise ValueError(
                "a and tau_c must either both be supplied or both be None."
            )

        a = _as_1d_finite(a, "a")

        if a.size != X.shape[1]:
            raise ValueError(
                "a must have one entry for each column of X."
            )

        tau_c = float(tau_c)

        if not np.isfinite(tau_c) or tau_c <= 0.0:
            raise ValueError(
                "tau_c must be finite and strictly positive."
            )

        B = np.vstack((B, a[None, :] / tau_c))

    # Largest singular value squared.
    smax = np.linalg.svd(
        B,
        compute_uv=False,
        full_matrices=False,
    )[0]

    return float(smax**2)


def sum_zero_lipschitz(X, sigma2):
    r"""
    Exact smooth-part Lipschitz constant after restriction to

        C = {beta : 1^T beta = 0}.

    Let P_C = I - 11^T/p. Then

        L_{f,C}
          = ||X P_C||_2^2 / sigma2.

    Rather than forming P_C explicitly,

        X P_C

    is obtained by subtracting each row mean from the
    corresponding entries of that row.
    """
    X = _as_2d_finite(X, "X")

    sigma2 = float(sigma2)

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("sigma2 must be finite and strictly positive.")

    X_projected = X - np.mean(X, axis=1, keepdims=True)

    smax = np.linalg.svd(
        X_projected,
        compute_uv=False,
        full_matrices=False,
    )[0]

    return float(smax**2 / sigma2)


# ============================================================
# Laplace penalty
# ============================================================

def laplace_penalty(beta, theta):
    r"""
    Unsmoothed Laplace penalty

        h_theta(beta) = theta ||beta||_1.
    """
    beta = _as_1d_finite(beta, "beta")

    theta = float(theta)

    if not np.isfinite(theta) or theta < 0.0:
        raise ValueError("theta must be finite and nonnegative.")

    return float(theta * np.sum(np.abs(beta)))


# ============================================================
# Moreau envelope of the whole Laplace penalty
# ============================================================

def laplace_moreau_value(beta, theta, lambda_my):
    r"""
    Moreau-Yosida envelope of

        h_theta(beta) = theta ||beta||_1.

    Coordinatewise,

                          beta_j^2 / (2 lambda)
        h_j^lambda = {
                          theta |beta_j|
                          - lambda theta^2 / 2

    according as

        |beta_j| <= lambda theta

    or

        |beta_j| > lambda theta.
    """
    beta = _as_1d_finite(beta, "beta")

    theta = float(theta)
    lambda_my = float(lambda_my)

    if not np.isfinite(theta) or theta < 0.0:
        raise ValueError("theta must be finite and nonnegative.")

    if not np.isfinite(lambda_my) or lambda_my <= 0.0:
        raise ValueError(
            "lambda_my must be finite and strictly positive."
        )

    abs_beta = np.abs(beta)
    cutoff = lambda_my * theta

    values = np.where(
        abs_beta <= cutoff,
        beta**2 / (2.0 * lambda_my),
        theta * abs_beta
        - 0.5 * lambda_my * theta**2,
    )

    return float(np.sum(values))


def laplace_moreau_grad(beta, theta, lambda_my):
    r"""
    Gradient of the Moreau-Yosida envelope of

        h_theta(beta) = theta ||beta||_1:

        grad h_theta^lambda(beta)
          = [beta - prox_{lambda theta ||.||_1}(beta)] / lambda.
    """
    beta = _as_1d_finite(beta, "beta")

    theta = float(theta)
    lambda_my = float(lambda_my)

    if not np.isfinite(theta) or theta < 0.0:
        raise ValueError("theta must be finite and nonnegative.")

    if not np.isfinite(lambda_my) or lambda_my <= 0.0:
        raise ValueError(
            "lambda_my must be finite and strictly positive."
        )

    prox = soft_threshold(
        beta,
        lambda_my * theta,
    )

    return (beta - prox) / lambda_my


# ============================================================
# Full objectives
# ============================================================

def posterior_objective(
    beta,
    y,
    X,
    sigma2,
    theta,
    *,
    a=None,
    b=None,
    tau_c=None,
):
    r"""
    Unsmoothed negative log-posterior, up to a constant:

        U(beta) = f(beta) + theta ||beta||_1.

    If a, b, tau_c are supplied, f also contains the soft
    affine pseudo-observation term.
    """
    if a is None and b is None and tau_c is None:
        f = gaussian_nll(beta, y, X, sigma2)

    else:
        if a is None or b is None or tau_c is None:
            raise ValueError(
                "For a soft affine model, a, b, and tau_c "
                "must all be supplied."
            )

        f = soft_affine_nll(
            beta,
            y,
            X,
            sigma2,
            a,
            b,
            tau_c,
        )

    return f + laplace_penalty(beta, theta)


def smoothed_posterior_objective(
    beta,
    y,
    X,
    sigma2,
    theta,
    lambda_my,
    *,
    a=None,
    b=None,
    tau_c=None,
):
    r"""
    Moreau-smoothed negative log-posterior:

        U_lambda(beta)
          = f(beta) + h_theta^lambda(beta),

    where the whole Laplace term h_theta = theta ||.||_1
    is Moreau-smoothed.
    """
    if a is None and b is None and tau_c is None:
        f = gaussian_nll(beta, y, X, sigma2)

    else:
        if a is None or b is None or tau_c is None:
            raise ValueError(
                "For a soft affine model, a, b, and tau_c "
                "must all be supplied."
            )

        f = soft_affine_nll(
            beta,
            y,
            X,
            sigma2,
            a,
            b,
            tau_c,
        )

    return f + laplace_moreau_value(
        beta,
        theta,
        lambda_my,
    )
