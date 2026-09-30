
"""
noise.py

Stage-0 noise-scale estimators for the computational-statistics
sparse-regression pipeline.

Implemented branches
--------------------
1. Classical residual variance estimation when residual degrees
   of freedom are available.

2. Scaled Lasso for sparse high-dimensional regression.

Conventions
-----------
- The supplied model is

      y = X beta + epsilon,
      epsilon ~ N(0, sigma2 I).

- No routine centers, rescales, standardises, or adds an
  intercept silently.

- For scaled Lasso, the theoretically conventional scaling is

      ||X_j||_2 = sqrt(n).

  The routine can check this convention, but does not enforce it.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.linear_model import Lasso


__all__ = [
    "ResidualVarianceResult",
    "ScaledLassoResult",
    "residual_variance",
    "scaled_lasso_variance",
]


# ============================================================
# Result containers
# ============================================================

@dataclass(frozen=True)
class ResidualVarianceResult:
    sigma2: float
    sigma: float
    beta_ls: np.ndarray
    residuals: np.ndarray
    rss: float
    rank: int
    df_resid: int


@dataclass(frozen=True)
class ScaledLassoResult:
    sigma2: float
    sigma: float
    beta_nuisance: np.ndarray
    residuals: np.ndarray
    lambda0: float
    effective_alpha: float
    n_outer_iter: int
    converged: bool
    sigma_path: np.ndarray
    objective_path: np.ndarray
    nnz: int


# ============================================================
# Input validation
# ============================================================

def _validate_y_X(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if y.size == 0:
        raise ValueError("y must be non-empty.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have non-zero dimensions.")

    if X.shape[0] != y.size:
        raise ValueError(
            f"y has length {y.size}, but X has {X.shape[0]} rows."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")

    return y, X


# ============================================================
# Classical residual estimator
# ============================================================

def residual_variance(
    y,
    X,
    *,
    intercept_df=0,
    rcond=None,
):
    r"""
    Classical residual variance estimator.

    Computes the least-squares residuals and

        sigma2_hat = RSS / df_resid,

    where

        df_resid = n - rank(X) - intercept_df.

    Parameters
    ----------
    y : array_like, shape (n,)
        Response vector.

    X : array_like, shape (n,p)
        Design matrix.

    intercept_df : int, optional
        Additional degrees of freedom consumed by an intercept
        or preprocessing step.

        Default is 0 because the core pipeline assumes that
        intercept handling / centering has already been decided
        before this routine is called.

        If an intercept has been estimated and removed by
        centering the observed data, use intercept_df=1.

    rcond : float or None, optional
        Cutoff used by np.linalg.lstsq for numerical rank
        determination.

    Returns
    -------
    ResidualVarianceResult
    """
    y, X = _validate_y_X(y, X)

    if not isinstance(intercept_df, (int, np.integer)):
        raise ValueError("intercept_df must be an integer.")

    if intercept_df < 0:
        raise ValueError("intercept_df must be nonnegative.")

    beta_ls, _, rank, _ = np.linalg.lstsq(
        X,
        y,
        rcond=rcond,
    )

    rank = int(rank)
    residuals = y - X @ beta_ls
    rss = float(residuals @ residuals)

    df_resid = int(y.size - rank - intercept_df)

    if df_resid <= 0:
        raise ValueError(
            "No residual degrees of freedom are available: "
            f"n={y.size}, rank(X)={rank}, "
            f"intercept_df={intercept_df}."
        )

    sigma2 = rss / df_resid
    sigma = float(np.sqrt(sigma2))

    return ResidualVarianceResult(
        sigma2=float(sigma2),
        sigma=sigma,
        beta_ls=np.asarray(beta_ls, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        rss=rss,
        rank=rank,
        df_resid=df_resid,
    )


# ============================================================
# Scaled Lasso
# ============================================================

def scaled_lasso_variance(
    y,
    X,
    *,
    lambda0=None,
    A=1.0,
    sigma_init=None,
    tol_outer=1e-6,
    max_outer_iter=100,
    tol_inner=1e-10,
    max_inner_iter=20000,
    check_column_norms=True,
    column_norm_rtol=0.05,
):
    r"""
    Scaled-Lasso noise-scale estimator.

    We use the joint scaled-Lasso criterion

        Q(beta, sigma)
          =
          ||y-X beta||_2^2 / (2 n sigma)
          + sigma / 2
          + lambda0 ||beta||_1,

    for sigma > 0.

    Alternating minimisation gives:

    beta-step:
        beta <- argmin_beta
            ||y-X beta||^2/(2n)
            + (lambda0 * sigma) ||beta||_1,

    sigma-step:
        sigma <- ||y-X beta||_2 / sqrt(n).

    The beta-step uses sklearn.linear_model.Lasso, whose
    objective convention is exactly

        ||y-X beta||^2/(2n) + alpha ||beta||_1.

    Hence

        alpha = lambda0 * sigma.

    Parameters
    ----------
    y : array_like, shape (n,)
    X : array_like, shape (n,p)

    lambda0 : float or None
        Scale-free Lasso tuning parameter. If None,

            lambda0 = A * sqrt(2 log(p) / n).

    A : float, optional
        Multiplicative factor in the default lambda0.

    sigma_init : float or None
        Initial noise standard deviation. If None,

            ||y||_2 / sqrt(n)

        is used.

    tol_outer : float
        Relative tolerance for the sigma fixed point.

    max_outer_iter : int
        Maximum number of scaled-Lasso outer iterations.

    tol_inner : float
        Coordinate-descent tolerance for each Lasso solve.

    max_inner_iter : int
        Maximum coordinate-descent iterations for each
        Lasso solve.

    check_column_norms : bool
        If True, warn when columns materially deviate from
        the convention ||X_j||_2 = sqrt(n).

    column_norm_rtol : float
        Relative tolerance for that check.

    Returns
    -------
    ScaledLassoResult

    Notes
    -----
    The returned beta_nuisance is the internal sparse regression
    estimate required by the scale procedure. In the proposed
    paper workflow, this vector is discarded after sigma2 has
    been estimated.
    """
    y, X = _validate_y_X(y, X)

    n, p = X.shape

    if p < 1:
        raise ValueError("X must contain at least one predictor.")

    A = float(A)

    if not np.isfinite(A) or A <= 0.0:
        raise ValueError("A must be finite and strictly positive.")

    if lambda0 is None:
        if p == 1:
            # log(1)=0 would give no regularisation.
            # Keep p_eff >= 2 solely for the default tuning scale.
            p_eff = 2
        else:
            p_eff = p

        lambda0 = A * np.sqrt(
            2.0 * np.log(p_eff) / n
        )
    else:
        lambda0 = float(lambda0)

    if not np.isfinite(lambda0) or lambda0 <= 0.0:
        raise ValueError(
            "lambda0 must be finite and strictly positive."
        )

    tol_outer = float(tol_outer)
    tol_inner = float(tol_inner)

    if not np.isfinite(tol_outer) or tol_outer <= 0.0:
        raise ValueError("tol_outer must be positive and finite.")

    if not np.isfinite(tol_inner) or tol_inner <= 0.0:
        raise ValueError("tol_inner must be positive and finite.")

    if (
        not isinstance(max_outer_iter, (int, np.integer))
        or max_outer_iter <= 0
    ):
        raise ValueError(
            "max_outer_iter must be a positive integer."
        )

    if (
        not isinstance(max_inner_iter, (int, np.integer))
        or max_inner_iter <= 0
    ):
        raise ValueError(
            "max_inner_iter must be a positive integer."
        )

    column_norm_rtol = float(column_norm_rtol)

    if (
        not np.isfinite(column_norm_rtol)
        or column_norm_rtol < 0.0
    ):
        raise ValueError(
            "column_norm_rtol must be finite and nonnegative."
        )

    # --------------------------------------------------------
    # Check, but never silently alter, design scaling.
    # --------------------------------------------------------

    column_norms = np.linalg.norm(X, axis=0)

    if np.any(column_norms == 0.0):
        raise ValueError(
            "X contains at least one identically zero column."
        )

    if check_column_norms:
        target_norm = np.sqrt(n)

        relative_error = np.abs(
            column_norms / target_norm - 1.0
        )

        if np.max(relative_error) > column_norm_rtol:
            warnings.warn(
                "Scaled Lasso is being run with columns that "
                "deviate materially from ||X_j||_2=sqrt(n). "
                "The routine will NOT rescale X automatically. "
                f"Maximum relative norm deviation: "
                f"{np.max(relative_error):.3f}.",
                RuntimeWarning,
                stacklevel=2,
            )

    # --------------------------------------------------------
    # Initial sigma
    # --------------------------------------------------------

    if sigma_init is None:
        sigma = float(
            np.linalg.norm(y) / np.sqrt(n)
        )

        # Handle the degenerate y=0 case without pretending
        # sigma=0 is an admissible optimization variable.
        sigma = max(
            sigma,
            np.sqrt(np.finfo(float).eps),
        )
    else:
        sigma = float(sigma_init)

        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "sigma_init must be finite and strictly positive."
            )

    sigma_path = [sigma]
    objective_path = []

    # One estimator object with warm starts across outer
    # iterations.
    lasso = Lasso(
        alpha=lambda0 * sigma,
        fit_intercept=False,
        max_iter=max_inner_iter,
        tol=tol_inner,
        warm_start=True,
        selection="cyclic",
    )

    # Explicit zero initialisation for reproducibility.
    lasso.coef_ = np.zeros(p, dtype=float)

    converged = False
    beta = np.zeros(p, dtype=float)

    # --------------------------------------------------------
    # Alternating minimisation
    # --------------------------------------------------------

    for outer in range(1, max_outer_iter + 1):

        sigma_old = sigma

        # beta-step
        lasso.alpha = float(lambda0 * sigma_old)
        lasso.fit(X, y)

        beta = np.asarray(
            lasso.coef_,
            dtype=float,
        ).copy()

        # sigma-step
        residuals = y - X @ beta
        residual_norm = float(np.linalg.norm(residuals))

        sigma = residual_norm / np.sqrt(n)

        # In an exact noiseless perfect-fit case the joint
        # optimum may approach the boundary sigma -> 0.
        # This pipeline assumes a genuine positive noise scale,
        # so flag that situation explicitly.
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise RuntimeError(
                "Scaled-Lasso sigma update reached a nonpositive "
                "or non-finite value. Check whether the supplied "
                "problem admits an essentially exact fit."
            )

        sigma_path.append(float(sigma))

        # Joint scaled-Lasso objective evaluated at the updated
        # (beta, sigma).
        scaled_objective = (
            residual_norm**2 / (2.0 * n * sigma)
            + 0.5 * sigma
            + lambda0 * np.sum(np.abs(beta))
        )

        objective_path.append(
            float(scaled_objective)
        )

        relative_change = (
            abs(sigma - sigma_old)
            / max(
                abs(sigma_old),
                np.sqrt(np.finfo(float).eps),
            )
        )

        if relative_change <= tol_outer:
            converged = True
            break

    # --------------------------------------------------------
    # Final consistency solve
    #
    # The beta currently corresponds to alpha=lambda0*sigma_old,
    # while the final reported sigma is the subsequent residual
    # update. Usually their difference is negligible at
    # convergence, but we perform one final beta solve at the
    # reported sigma so that effective_alpha and beta_nuisance
    # are exactly aligned.
    # --------------------------------------------------------

    lasso.alpha = float(lambda0 * sigma)
    lasso.fit(X, y)

    beta = np.asarray(
        lasso.coef_,
        dtype=float,
    ).copy()

    residuals = y - X @ beta
    sigma_final = float(
        np.linalg.norm(residuals) / np.sqrt(n)
    )

    # If the final consistency solve moves sigma materially,
    # retain the fixed-point estimate as the actual report.
    # Iterate a few additional times, bounded by max_outer_iter,
    # only if necessary.
    extra_iter = 0

    while (
        abs(sigma_final - sigma)
        / max(abs(sigma), np.sqrt(np.finfo(float).eps))
        > tol_outer
        and (outer + extra_iter) < max_outer_iter
    ):
        sigma = sigma_final

        lasso.alpha = float(lambda0 * sigma)
        lasso.fit(X, y)

        beta = np.asarray(
            lasso.coef_,
            dtype=float,
        ).copy()

        residuals = y - X @ beta
        sigma_final = float(
            np.linalg.norm(residuals) / np.sqrt(n)
        )

        extra_iter += 1

    sigma = sigma_final
    sigma2 = float(sigma**2)

    effective_alpha = float(lambda0 * sigma)
    nnz = int(np.count_nonzero(np.abs(beta) > 0.0))

    total_outer = int(outer + extra_iter)

    # Re-evaluate final objective.
    residual_norm = float(np.linalg.norm(residuals))

    final_objective = (
        residual_norm**2 / (2.0 * n * sigma)
        + 0.5 * sigma
        + lambda0 * np.sum(np.abs(beta))
    )

    if len(objective_path) == 0:
        objective_path.append(float(final_objective))
    elif (
        abs(objective_path[-1] - final_objective)
        > 10.0 * np.finfo(float).eps
    ):
        objective_path.append(float(final_objective))

    # Final fixed-point convergence assessment.
    # We use the latest beta and reported sigma.
    fixed_point_sigma = (
        np.linalg.norm(y - X @ beta) / np.sqrt(n)
    )

    fixed_point_rel_error = (
        abs(fixed_point_sigma - sigma)
        / max(abs(sigma), np.sqrt(np.finfo(float).eps))
    )

    converged = bool(
        converged
        and fixed_point_rel_error <= 10.0 * tol_outer
    )

    return ScaledLassoResult(
        sigma2=sigma2,
        sigma=float(sigma),
        beta_nuisance=beta,
        residuals=np.asarray(residuals, dtype=float),
        lambda0=float(lambda0),
        effective_alpha=effective_alpha,
        n_outer_iter=total_outer,
        converged=converged,
        sigma_path=np.asarray(sigma_path, dtype=float),
        objective_path=np.asarray(
            objective_path,
            dtype=float,
        ),
        nnz=nnz,
    )
