
"""
simulation.py

Synthetic-regression generators for the computational-statistics
sparse-regression paper.

Design convention
-----------------
Rows of the unscaled Gaussian design follow

    X_i ~ N(0, Sigma),

with AR(1) covariance

    Sigma_jk = rho^{|j-k|}.

The design is generated recursively, avoiding a p-by-p Cholesky
factorisation.

By default, each realised column is subsequently scaled to

    ||X_j||_2 = sqrt(n),

matching the convention used by the scaled-Lasso and the main
simulation study.

No intercept is included or silently estimated.

SNR convention
--------------
For the nominal AR(1) population design,

    SNR_pop
      =
      beta^T Sigma beta / sigma^2.

The realised-design counterpart is

    SNR_sample
      =
      ||X beta||_2^2 / (n sigma^2).

The main paper design currently uses population-SNR calibration,
but sample-SNR calibration is also provided for diagnostics or
future variants.

Constraint-specific generators
------------------------------
Separate wrappers are provided for:
    - unconstrained sparse regression,
    - hard sum-to-zero truth,
    - signed sum-to-one truth.

No nonnegativity constraint is imposed in the sum-to-one model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "RegressionSimulation",
    "ar1_design",
    "scale_columns_sqrt_n",
    "population_snr_ar1",
    "sample_snr",
    "rescale_beta_to_population_snr",
    "rescale_beta_to_sample_snr",
    "make_sparse_beta",
    "simulate_sparse_regression",
    "simulate_sum_zero_regression",
    "simulate_sum_one_regression",
]


# ============================================================
# Result container
# ============================================================

@dataclass(frozen=True)
class RegressionSimulation:
    X: np.ndarray
    y: np.ndarray
    beta_true: np.ndarray
    noise: np.ndarray

    sigma: float
    sigma2: float
    rho: float

    support: np.ndarray

    target_snr: float | None
    snr_mode: str | None
    population_snr: float
    sample_snr: float

    constraint: str
    constraint_rhs: float | None
    constraint_residual: float | None

    columns_scaled_sqrt_n: bool


# ============================================================
# Validation helpers
# ============================================================

def _validate_rng(rng):
    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator."
        )


def _validate_n_p(n, p):
    if (
        not isinstance(n, (int, np.integer))
        or n <= 0
    ):
        raise ValueError(
            "n must be a positive integer."
        )

    if (
        not isinstance(p, (int, np.integer))
        or p <= 0
    ):
        raise ValueError(
            "p must be a positive integer."
        )

    return int(n), int(p)


def _validate_rho(rho):
    rho = float(rho)

    if (
        not np.isfinite(rho)
        or abs(rho) >= 1.0
    ):
        raise ValueError(
            "rho must satisfy |rho| < 1."
        )

    return rho


def _validate_sigma(sigma):
    sigma = float(sigma)

    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
    ):
        raise ValueError(
            "sigma must be finite and strictly positive."
        )

    return sigma


def _as_beta(beta):
    beta = np.asarray(
        beta,
        dtype=float,
    )

    if beta.ndim != 1:
        raise ValueError(
            "beta must be one-dimensional."
        )

    if beta.size == 0:
        raise ValueError(
            "beta must be non-empty."
        )

    if not np.all(
        np.isfinite(beta)
    ):
        raise ValueError(
            "beta contains non-finite values."
        )

    return beta


# ============================================================
# AR(1) design
# ============================================================

def ar1_design(
    n,
    p,
    rho,
    *,
    rng,
):
    r"""
    Generate an n-by-p Gaussian AR(1) design with

        Cov(X_j, X_k) = rho^{|j-k|}.

    Generation is rowwise-equivalent to

        X_1 = Z_1,
        X_j = rho X_{j-1}
              + sqrt(1-rho^2) Z_j.

    Returns
    -------
    X : ndarray, shape (n,p)
    """
    _validate_rng(rng)

    n, p = _validate_n_p(
        n,
        p,
    )

    rho = _validate_rho(
        rho
    )

    Z = rng.normal(
        size=(n, p)
    )

    X = np.empty(
        (n, p),
        dtype=float,
    )

    X[:, 0] = Z[:, 0]

    innovation_scale = np.sqrt(
        1.0 - rho**2
    )

    for j in range(
        1,
        p,
    ):
        X[:, j] = (
            rho * X[:, j - 1]
            + innovation_scale * Z[:, j]
        )

    return X


# ============================================================
# Column scaling
# ============================================================

def scale_columns_sqrt_n(X):
    r"""
    Scale each design column to Euclidean norm sqrt(n).

    No centering is performed.

    Parameters
    ----------
    X : array_like, shape (n,p)

    Returns
    -------
    X_scaled : ndarray, shape (n,p)
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if (
        X.shape[0] == 0
        or X.shape[1] == 0
    ):
        raise ValueError(
            "X must have non-zero dimensions."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "X contains non-finite values."
        )

    n = X.shape[0]

    norms = np.linalg.norm(
        X,
        axis=0,
    )

    if np.any(norms == 0.0):
        raise ValueError(
            "Cannot scale an identically-zero design column."
        )

    return (
        X
        * (
            np.sqrt(n)
            / norms
        )
    )


# ============================================================
# SNR calculations
# ============================================================

def population_snr_ar1(
    beta,
    rho,
    sigma,
):
    r"""
    Population signal-to-noise ratio under AR(1) design:

        beta^T Sigma beta / sigma^2,

    where

        Sigma_jk = rho^{|j-k|}.
    """
    beta = _as_beta(
        beta
    )

    rho = _validate_rho(
        rho
    )

    sigma = _validate_sigma(
        sigma
    )

    p = beta.size

    indices = np.arange(
        p
    )

    Sigma = rho ** np.abs(
        indices[:, None]
        - indices[None, :]
    )

    signal_variance = float(
        beta @ Sigma @ beta
    )

    # Numerical roundoff can produce a minute negative value
    # only in degenerate cases.
    signal_variance = max(
        0.0,
        signal_variance,
    )

    return float(
        signal_variance
        / sigma**2
    )


def sample_snr(
    X,
    beta,
    sigma,
):
    r"""
    Realised-design signal-to-noise ratio:

        ||X beta||^2 / (n sigma^2).
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    beta = _as_beta(
        beta
    )

    sigma = _validate_sigma(
        sigma
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if X.shape[1] != beta.size:
        raise ValueError(
            "X and beta have incompatible dimensions."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "X must have at least one row."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "X contains non-finite values."
        )

    signal = X @ beta

    return float(
        (signal @ signal)
        / (
            X.shape[0]
            * sigma**2
        )
    )


# ============================================================
# SNR calibration
# ============================================================

def rescale_beta_to_population_snr(
    beta,
    target_snr,
    *,
    rho,
    sigma,
):
    r"""
    Multiply beta by a scalar so that

        beta^T Sigma beta / sigma^2
          = target_snr

    under the nominal AR(1) population covariance.
    """
    beta = _as_beta(
        beta
    )

    target_snr = float(
        target_snr
    )

    if (
        not np.isfinite(target_snr)
        or target_snr <= 0.0
    ):
        raise ValueError(
            "target_snr must be finite and strictly positive."
        )

    current = population_snr_ar1(
        beta,
        rho,
        sigma,
    )

    if current <= 0.0:
        raise ValueError(
            "Cannot SNR-rescale a beta vector with zero "
            "population signal variance."
        )

    factor = np.sqrt(
        target_snr / current
    )

    return (
        factor * beta
    )


def rescale_beta_to_sample_snr(
    beta,
    target_snr,
    *,
    X,
    sigma,
):
    r"""
    Multiply beta by a scalar so that the realised-design SNR

        ||X beta||^2/(n sigma^2)

    equals target_snr.
    """
    beta = _as_beta(
        beta
    )

    target_snr = float(
        target_snr
    )

    if (
        not np.isfinite(target_snr)
        or target_snr <= 0.0
    ):
        raise ValueError(
            "target_snr must be finite and strictly positive."
        )

    current = sample_snr(
        X,
        beta,
        sigma,
    )

    if current <= 0.0:
        raise ValueError(
            "Cannot SNR-rescale a beta vector with zero "
            "realised signal."
        )

    factor = np.sqrt(
        target_snr / current
    )

    return (
        factor * beta
    )


# ============================================================
# Sparse coefficient construction
# ============================================================

def make_sparse_beta(
    p,
    support,
    values,
):
    r"""
    Construct a sparse coefficient vector of length p.

    Parameters
    ----------
    p : int

    support : array_like of int
        Zero-based active coordinates.

    values : array_like
        Nonzero coefficient values corresponding to support.
    """
    if (
        not isinstance(p, (int, np.integer))
        or p <= 0
    ):
        raise ValueError(
            "p must be a positive integer."
        )

    support = np.asarray(
        support,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if support.ndim != 1:
        raise ValueError(
            "support must be one-dimensional."
        )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if support.size != values.size:
        raise ValueError(
            "support and values must have the same length."
        )

    if support.size == 0:
        raise ValueError(
            "support must be non-empty."
        )

    if not np.issubdtype(
        support.dtype,
        np.integer,
    ):
        # Permit integer-valued floats but reject arbitrary
        # noninteger coordinates.
        support_float = support.astype(
            float
        )

        if not np.all(
            support_float
            == np.floor(support_float)
        ):
            raise ValueError(
                "support entries must be integers."
            )

        support = support_float.astype(
            int
        )
    else:
        support = support.astype(
            int,
            copy=False,
        )

    if np.any(
        support < 0
    ) or np.any(
        support >= p
    ):
        raise ValueError(
            "support contains indices outside [0,p)."
        )

    if np.unique(
        support
    ).size != support.size:
        raise ValueError(
            "support indices must be unique."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "values contains non-finite entries."
        )

    if np.any(
        values == 0.0
    ):
        raise ValueError(
            "Active values must be nonzero."
        )

    beta = np.zeros(
        p,
        dtype=float,
    )

    beta[support] = values

    return beta


# ============================================================
# Internal generic simulator
# ============================================================

def _simulate_from_beta(
    n,
    beta_true,
    rho,
    sigma,
    *,
    rng,
    target_snr=None,
    snr_mode=None,
    scale_columns=True,
    constraint="none",
    constraint_rhs=None,
):
    _validate_rng(
        rng
    )

    beta_true = _as_beta(
        beta_true
    ).copy()

    n, p = _validate_n_p(
        n,
        beta_true.size,
    )

    rho = _validate_rho(
        rho
    )

    sigma = _validate_sigma(
        sigma
    )

    X = ar1_design(
        n,
        p,
        rho,
        rng=rng,
    )

    if scale_columns:
        X = scale_columns_sqrt_n(
            X
        )

    # --------------------------------------------------------
    # Optional SNR calibration
    # --------------------------------------------------------

    if target_snr is not None:
        target_snr = float(
            target_snr
        )

        if snr_mode is None:
            snr_mode = "population"

        if snr_mode == "population":
            beta_true = (
                rescale_beta_to_population_snr(
                    beta_true,
                    target_snr,
                    rho=rho,
                    sigma=sigma,
                )
            )

        elif snr_mode == "sample":
            beta_true = (
                rescale_beta_to_sample_snr(
                    beta_true,
                    target_snr,
                    X=X,
                    sigma=sigma,
                )
            )

        else:
            raise ValueError(
                "snr_mode must be 'population' or 'sample'."
            )

    else:
        if snr_mode is not None:
            raise ValueError(
                "snr_mode should be None when target_snr is None."
            )

    population_snr_value = (
        population_snr_ar1(
            beta_true,
            rho,
            sigma,
        )
    )

    sample_snr_value = (
        sample_snr(
            X,
            beta_true,
            sigma,
        )
    )

    noise = rng.normal(
        scale=sigma,
        size=n,
    )

    y = (
        X @ beta_true
        + noise
    )

    support = np.flatnonzero(
        beta_true != 0.0
    )

    # --------------------------------------------------------
    # Constraint metadata
    # --------------------------------------------------------

    if constraint == "none":
        residual = None
        rhs = None

    elif constraint in {
        "sum_zero",
        "sum_one",
    }:
        if constraint_rhs is None:
            raise ValueError(
                "constraint_rhs is required for constrained truth."
            )

        rhs = float(
            constraint_rhs
        )

        residual = float(
            np.sum(beta_true)
            - rhs
        )

    else:
        raise ValueError(
            "Unknown constraint label."
        )

    return RegressionSimulation(
        X=np.asarray(
            X,
            dtype=float,
        ),

        y=np.asarray(
            y,
            dtype=float,
        ),

        beta_true=np.asarray(
            beta_true,
            dtype=float,
        ),

        noise=np.asarray(
            noise,
            dtype=float,
        ),

        sigma=float(
            sigma
        ),

        sigma2=float(
            sigma**2
        ),

        rho=float(
            rho
        ),

        support=np.asarray(
            support,
            dtype=int,
        ),

        target_snr=(
            None
            if target_snr is None
            else float(target_snr)
        ),

        snr_mode=snr_mode,

        population_snr=float(
            population_snr_value
        ),

        sample_snr=float(
            sample_snr_value
        ),

        constraint=constraint,

        constraint_rhs=rhs,

        constraint_residual=residual,

        columns_scaled_sqrt_n=bool(
            scale_columns
        ),
    )


# ============================================================
# Main unconstrained sparse regression generator
# ============================================================

def simulate_sparse_regression(
    n,
    p,
    support,
    values,
    *,
    rho,
    sigma,
    rng,
    target_snr=None,
    snr_mode="population",
    scale_columns=True,
):
    r"""
    Generate an unconstrained sparse-regression problem.

    If target_snr is supplied, the default interpretation is

        beta^T Sigma beta / sigma^2 = target_snr.

    This is the convention currently frozen for Study 1.
    """
    beta = make_sparse_beta(
        p,
        support,
        values,
    )

    if target_snr is None:
        snr_mode_used = None
    else:
        snr_mode_used = snr_mode

    return _simulate_from_beta(
        n,
        beta,
        rho,
        sigma,
        rng=rng,
        target_snr=target_snr,
        snr_mode=snr_mode_used,
        scale_columns=scale_columns,
        constraint="none",
    )


# ============================================================
# Hard sum-zero truth
# ============================================================

def simulate_sum_zero_regression(
    n,
    beta_true,
    *,
    rho,
    sigma,
    rng,
    target_snr=None,
    snr_mode="population",
    scale_columns=True,
    constraint_tol=1e-12,
):
    r"""
    Generate regression data from a sparse signed coefficient
    vector satisfying

        1^T beta_true = 0.

    Scalar SNR rescaling preserves the homogeneous constraint,
    so optional target-SNR calibration is allowed.
    """
    beta_true = _as_beta(
        beta_true
    )

    constraint_tol = float(
        constraint_tol
    )

    if (
        not np.isfinite(constraint_tol)
        or constraint_tol < 0.0
    ):
        raise ValueError(
            "constraint_tol must be finite and nonnegative."
        )

    if abs(
        np.sum(beta_true)
    ) > constraint_tol:
        raise ValueError(
            "beta_true does not satisfy the sum-zero constraint."
        )

    if target_snr is None:
        snr_mode_used = None
    else:
        snr_mode_used = snr_mode

    result = _simulate_from_beta(
        n,
        beta_true,
        rho,
        sigma,
        rng=rng,
        target_snr=target_snr,
        snr_mode=snr_mode_used,
        scale_columns=scale_columns,
        constraint="sum_zero",
        constraint_rhs=0.0,
    )

    if abs(
        result.constraint_residual
    ) > max(
        constraint_tol,
        1e-12,
    ):
        raise RuntimeError(
            "SNR calibration unexpectedly violated sum-zero truth."
        )

    return result


# ============================================================
# Signed sum-to-one truth
# ============================================================

def simulate_sum_one_regression(
    n,
    beta_true,
    *,
    rho,
    sigma,
    rng,
    scale_columns=True,
    constraint_tol=1e-12,
):
    r"""
    Generate regression data from a signed coefficient vector
    satisfying

        1^T beta_true = 1.

    IMPORTANT:
    No nonnegativity constraint is imposed.

    SNR rescaling is deliberately NOT performed here because
    ordinary scalar rescaling would destroy the affine condition
    sum(beta)=1.
    """
    beta_true = _as_beta(
        beta_true
    )

    constraint_tol = float(
        constraint_tol
    )

    if (
        not np.isfinite(constraint_tol)
        or constraint_tol < 0.0
    ):
        raise ValueError(
            "constraint_tol must be finite and nonnegative."
        )

    if abs(
        np.sum(beta_true) - 1.0
    ) > constraint_tol:
        raise ValueError(
            "beta_true does not satisfy the sum-to-one constraint."
        )

    return _simulate_from_beta(
        n,
        beta_true,
        rho,
        sigma,
        rng=rng,
        target_snr=None,
        snr_mode=None,
        scale_columns=scale_columns,
        constraint="sum_one",
        constraint_rhs=1.0,
    )
