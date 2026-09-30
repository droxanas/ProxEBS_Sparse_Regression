
"""
selection.py

Posterior-informed variable-selection utilities for the
computational-statistics sparse-regression pipeline.

Current working rule
--------------------
The primary point estimate is the MAP of the ORIGINAL
nonsmoothed posterior.

Posterior uncertainty is estimated from MYULA draws associated
with the Moreau-smoothed posterior pi_lambda.

Given retained posterior draws beta^(m), define

    s_j(lambda)
      =
      sd_{pi_lambda}(beta_j | y),

and

    tau_post(lambda)
      =
      k * median_j s_j(lambda).

The activation probability is

    pi_j(lambda)
      =
      P_{pi_lambda}(
          |beta_j| >= tau_post(lambda)
          | y
        ),

estimated empirically from the retained MYULA draws.

The current provisional two-gate rule is

    S_hat_lambda
      =
      {
        j :
        |beta_MAP,j| >= tau_post(lambda)
        and
        pi_j(lambda) >= pi_star
      }.

The MAP itself is unsmoothed. The decision nevertheless depends
on lambda because both tau_post and the activation probabilities
are computed from pi_lambda.

This final rule is intentionally modular: its components can be
retained even if the precise downstream decision rule is revised
later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "SelectionResult",
    "posterior_sd",
    "posterior_scale_threshold",
    "activation_probabilities",
    "select_support",
]


@dataclass(frozen=True)
class SelectionResult:
    support_mask: np.ndarray
    support_indices: np.ndarray
    selected_size: int

    threshold: float
    k: float
    pi_star: float
    lambda_my: float

    posterior_sd: np.ndarray
    activation_probability: np.ndarray

    map_gate: np.ndarray
    probability_gate: np.ndarray

    beta_map: np.ndarray

    n_draws: int
    n_variables: int
    ddof: int


# ============================================================
# Validation helpers
# ============================================================

def _validate_samples(samples):
    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape (n_draws, p)."
        )

    if (
        samples.shape[0] == 0
        or samples.shape[1] == 0
    ):
        raise ValueError(
            "samples must have non-zero dimensions."
        )

    if not np.all(
        np.isfinite(samples)
    ):
        raise ValueError(
            "samples contains non-finite values."
        )

    return samples


def _validate_ddof(ddof, n_draws):
    if not isinstance(
        ddof,
        (int, np.integer),
    ):
        raise ValueError(
            "ddof must be an integer."
        )

    if ddof < 0:
        raise ValueError(
            "ddof must be nonnegative."
        )

    if n_draws <= ddof:
        raise ValueError(
            "Need n_draws > ddof to estimate posterior SDs."
        )

    return int(ddof)


# ============================================================
# Posterior marginal scales
# ============================================================

def posterior_sd(
    samples,
    *,
    ddof=1,
):
    r"""
    Marginal posterior standard deviations estimated from
    retained posterior samples.

    Parameters
    ----------
    samples : ndarray, shape (M,p)
        Retained MYULA draws.

    ddof : int, optional
        Degrees-of-freedom convention used by np.std.
        Default is 1.

    Returns
    -------
    sd : ndarray, shape (p,)
    """
    samples = _validate_samples(
        samples
    )

    ddof = _validate_ddof(
        ddof,
        samples.shape[0],
    )

    sd = np.std(
        samples,
        axis=0,
        ddof=ddof,
    )

    if not np.all(
        np.isfinite(sd)
    ):
        raise RuntimeError(
            "Posterior SD calculation produced non-finite values."
        )

    return np.asarray(
        sd,
        dtype=float,
    )


def posterior_scale_threshold(
    samples,
    k,
    *,
    ddof=1,
):
    r"""
    Posterior practical-effect threshold

        tau_post(lambda)
          =
          k * median_j s_j(lambda),

    where s_j(lambda) is the marginal posterior SD estimated
    from the supplied smoothed-posterior draws.

    Returns
    -------
    threshold : float
    sd : ndarray, shape (p,)
    """
    k = float(k)

    if not np.isfinite(k) or k < 0.0:
        raise ValueError(
            "k must be finite and nonnegative."
        )

    sd = posterior_sd(
        samples,
        ddof=ddof,
    )

    threshold = float(
        k * np.median(sd)
    )

    return threshold, sd


# ============================================================
# Activation probabilities
# ============================================================

def activation_probabilities(
    samples,
    threshold,
):
    r"""
    Empirical posterior activation probabilities

        pi_j(lambda)
          =
          P(
            |beta_j| >= threshold
            | y
          ),

    estimated by the fraction of retained posterior draws
    satisfying the event.

    Parameters
    ----------
    samples : ndarray, shape (M,p)

    threshold : float
        Nonnegative practical-effect threshold.

    Returns
    -------
    probabilities : ndarray, shape (p,)
    """
    samples = _validate_samples(
        samples
    )

    threshold = float(
        threshold
    )

    if (
        not np.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError(
            "threshold must be finite and nonnegative."
        )

    active = (
        np.abs(samples)
        >= threshold
    )

    probabilities = np.mean(
        active,
        axis=0,
    )

    return np.asarray(
        probabilities,
        dtype=float,
    )


# ============================================================
# Current provisional two-gate rule
# ============================================================

def select_support(
    beta_map,
    samples,
    *,
    k,
    pi_star,
    lambda_my,
    ddof=1,
):
    r"""
    Apply the current provisional posterior-informed support rule.

    The unsmoothed MAP supplies the point-estimate magnitude gate:

        |beta_MAP,j| >= tau_post(lambda).

    MYULA draws supply both

        tau_post(lambda)

    and

        pi_j(lambda)
          =
          P_{pi_lambda}(
            |beta_j| >= tau_post(lambda)
            | y
          ).

    Variable j is selected if both

        |beta_MAP,j| >= tau_post(lambda)

    and

        pi_j(lambda) >= pi_star.

    Parameters
    ----------
    beta_map : array_like, shape (p,)
        MAP of the ORIGINAL unsmoothed posterior.

    samples : array_like, shape (M,p)
        Retained MYULA draws from the lambda-smoothed
        posterior approximation.

    k : float
        Multiplier defining tau_post(lambda).

    pi_star : float
        Activation-probability threshold in [0,1].

    lambda_my : float
        Moreau parameter associated with the posterior sample.
        This is retained as metadata so that downstream decisions
        remain explicitly lambda-dependent.

    ddof : int, optional
        Posterior SD convention. Default 1.

    Returns
    -------
    SelectionResult
    """
    samples = _validate_samples(
        samples
    )

    beta_map = np.asarray(
        beta_map,
        dtype=float,
    )

    if beta_map.ndim != 1:
        raise ValueError(
            "beta_map must be one-dimensional."
        )

    if beta_map.size != samples.shape[1]:
        raise ValueError(
            "beta_map must contain one entry per sampled variable."
        )

    if not np.all(
        np.isfinite(beta_map)
    ):
        raise ValueError(
            "beta_map contains non-finite values."
        )

    k = float(k)
    pi_star = float(pi_star)
    lambda_my = float(lambda_my)

    if not np.isfinite(k) or k < 0.0:
        raise ValueError(
            "k must be finite and nonnegative."
        )

    if (
        not np.isfinite(pi_star)
        or pi_star < 0.0
        or pi_star > 1.0
    ):
        raise ValueError(
            "pi_star must lie in [0,1]."
        )

    if (
        not np.isfinite(lambda_my)
        or lambda_my <= 0.0
    ):
        raise ValueError(
            "lambda_my must be finite and strictly positive."
        )

    ddof = _validate_ddof(
        ddof,
        samples.shape[0],
    )

    threshold, sd = (
        posterior_scale_threshold(
            samples,
            k,
            ddof=ddof,
        )
    )

    probabilities = (
        activation_probabilities(
            samples,
            threshold,
        )
    )

    map_gate = (
        np.abs(beta_map)
        >= threshold
    )

    probability_gate = (
        probabilities
        >= pi_star
    )

    support_mask = (
        map_gate
        & probability_gate
    )

    support_indices = np.flatnonzero(
        support_mask
    )

    return SelectionResult(
        support_mask=np.asarray(
            support_mask,
            dtype=bool,
        ),

        support_indices=np.asarray(
            support_indices,
            dtype=int,
        ),

        selected_size=int(
            support_indices.size
        ),

        threshold=float(
            threshold
        ),

        k=float(
            k
        ),

        pi_star=float(
            pi_star
        ),

        lambda_my=float(
            lambda_my
        ),

        posterior_sd=np.asarray(
            sd,
            dtype=float,
        ),

        activation_probability=np.asarray(
            probabilities,
            dtype=float,
        ),

        map_gate=np.asarray(
            map_gate,
            dtype=bool,
        ),

        probability_gate=np.asarray(
            probability_gate,
            dtype=bool,
        ),

        beta_map=np.asarray(
            beta_map,
            dtype=float,
        ).copy(),

        n_draws=int(
            samples.shape[0]
        ),

        n_variables=int(
            samples.shape[1]
        ),

        ddof=int(
            ddof
        ),
    )
