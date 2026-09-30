
"""
diagnostics.py

Diagnostics for the computational-statistics sparse-regression
pipeline.

This module covers:

1. MCMC diagnostics
   - autocorrelation function (ACF),
   - integrated autocorrelation time (IACT),
   - effective sample size (ESS),
   - Monte Carlo standard error (MCSE).

2. SAPG tail diagnostics
   - tail mean / variability of eta and theta,
   - tail score behaviour,
   - linear drift in eta,
   - frequency of projection-boundary hits.

3. Approximation diagnostics
   - discrepancy between the unsmoothed MAP and the
     Moreau-smoothed MAP.

4. Decision stability diagnostics
   - pairwise Jaccard similarity of support masks,
   - normalized Hamming disagreement.

No deterministic "convergence" flag is assigned to SAPG.
The returned quantities are intended to be inspected jointly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


__all__ = [
    "ChainSummary",
    "SapgTailSummary",
    "MapDiscrepancy",
    "SupportStability",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "mcse_mean",
    "chain_summary",
    "sapg_tail_summary",
    "map_discrepancy",
    "support_stability",
]


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class ChainSummary:
    mean: np.ndarray
    sd: np.ndarray
    iact: np.ndarray
    ess: np.ndarray
    mcse: np.ndarray
    n_draws: int
    n_variables: int
    max_lag: int | None


@dataclass(frozen=True)
class SapgTailSummary:
    tail_start: int
    n_tail: int

    eta_mean: float
    eta_sd: float
    eta_slope_per_iter: float

    theta_mean: float
    theta_sd: float
    theta_cv: float

    score_mean: float
    score_sd: float
    score_abs_mean: float

    lower_bound_hit_fraction: float
    upper_bound_hit_fraction: float
    any_bound_hit_fraction: float


@dataclass(frozen=True)
class MapDiscrepancy:
    l2: float
    relative_l2: float
    linf: float
    relative_linf: float

    threshold: float | None
    threshold_scaled_linf: float | None

    sign_disagreements: int
    n_variables: int


@dataclass(frozen=True)
class SupportStability:
    jaccard_matrix: np.ndarray
    hamming_matrix: np.ndarray

    mean_offdiag_jaccard: float
    min_offdiag_jaccard: float
    mean_offdiag_hamming: float
    max_offdiag_hamming: float

    support_sizes: np.ndarray
    n_settings: int
    n_variables: int


# ============================================================
# Validation helpers
# ============================================================

def _as_1d_finite(x, name="x"):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if x.size == 0:
        raise ValueError(
            f"{name} must be non-empty."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return x


def _as_samples(samples):
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


def _validate_max_lag(max_lag, n):
    if max_lag is None:
        return n - 1

    if not isinstance(
        max_lag,
        (int, np.integer),
    ):
        raise ValueError(
            "max_lag must be an integer or None."
        )

    if max_lag < 0:
        raise ValueError(
            "max_lag must be nonnegative."
        )

    return int(
        min(max_lag, n - 1)
    )


# ============================================================
# Autocorrelation
# ============================================================

def autocorrelation(
    x,
    *,
    max_lag=None,
):
    r"""
    Sample autocorrelation function.

    The autocovariance estimate uses denominator n:

        gamma_hat(k)
          =
          (1/n)
          sum_{t=1}^{n-k}
          (x_t-xbar)(x_{t+k}-xbar).

    The returned ACF is

        rho_hat(k)
          =
          gamma_hat(k)/gamma_hat(0).

    A constant series is treated specially:
        rho(0)=1 and rho(k)=0 for k>=1.

    Parameters
    ----------
    x : array_like, shape (n,)
    max_lag : int or None

    Returns
    -------
    acf : ndarray, shape (max_lag+1,)
    """
    x = _as_1d_finite(
        x,
        "x",
    )

    n = x.size

    max_lag = _validate_max_lag(
        max_lag,
        n,
    )

    centered = (
        x - np.mean(x)
    )

    variance_sum = float(
        centered @ centered
    )

    if variance_sum <= (
        100.0
        * np.finfo(float).eps
        * max(1.0, float(x @ x))
    ):
        acf = np.zeros(
            max_lag + 1,
            dtype=float,
        )

        acf[0] = 1.0

        return acf

    # FFT autocorrelation.
    n_fft = 1 << (
        2 * n - 1
    ).bit_length()

    fft_values = np.fft.rfft(
        centered,
        n=n_fft,
    )

    autocov_sum = np.fft.irfft(
        fft_values
        * np.conjugate(fft_values),
        n=n_fft,
    )[:max_lag + 1]

    acf = (
        autocov_sum
        / autocov_sum[0]
    )

    acf[0] = 1.0

    return np.asarray(
        acf,
        dtype=float,
    )


# ============================================================
# IACT and ESS
# ============================================================

def integrated_autocorrelation_time(
    x,
    *,
    max_lag=None,
):
    r"""
    Estimate integrated autocorrelation time using an
    initial-positive-sequence rule.

    With autocorrelations rho_k,

        tau
          =
          1 + 2 sum_{k>=1} rho_k.

    To reduce noise, consecutive autocorrelations are paired:

        Gamma_m
          =
          rho_{2m-1} + rho_{2m}.

    Summation stops before the first nonpositive pair.

    The estimate is truncated below at 1.

    Returns
    -------
    iact : float
    """
    x = _as_1d_finite(
        x,
        "x",
    )

    n = x.size

    if n < 2:
        return 1.0

    if max_lag is None:
        max_lag = min(
            n - 1,
            max(1, n // 2),
        )
    else:
        max_lag = _validate_max_lag(
            max_lag,
            n,
        )

    acf = autocorrelation(
        x,
        max_lag=max_lag,
    )

    if acf.size <= 1:
        return 1.0

    running_sum = 0.0

    lag = 1

    while lag < acf.size:
        pair_sum = acf[lag]

        if lag + 1 < acf.size:
            pair_sum += acf[
                lag + 1
            ]

        if pair_sum <= 0.0:
            break

        running_sum += pair_sum
        lag += 2

    tau = (
        1.0 + 2.0 * running_sum
    )

    return float(
        max(1.0, tau)
    )


def effective_sample_size(
    x,
    *,
    max_lag=None,
):
    r"""
    Effective sample size

        ESS = n / IACT.

    The estimate is clipped to [1,n].
    """
    x = _as_1d_finite(
        x,
        "x",
    )

    n = x.size

    tau = integrated_autocorrelation_time(
        x,
        max_lag=max_lag,
    )

    ess = n / tau

    return float(
        np.clip(
            ess,
            1.0,
            float(n),
        )
    )


def mcse_mean(
    x,
    *,
    max_lag=None,
    ddof=1,
):
    r"""
    Monte Carlo standard error of the sample mean:

        MCSE
          =
          sample_sd / sqrt(ESS).

    For a constant chain, MCSE=0.
    """
    x = _as_1d_finite(
        x,
        "x",
    )

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

    if x.size <= ddof:
        raise ValueError(
            "Need len(x) > ddof."
        )

    sd = float(
        np.std(
            x,
            ddof=ddof,
        )
    )

    if sd == 0.0:
        return 0.0

    ess = effective_sample_size(
        x,
        max_lag=max_lag,
    )

    return float(
        sd / np.sqrt(ess)
    )


# ============================================================
# Multivariate chain summary
# ============================================================

def chain_summary(
    samples,
    *,
    max_lag=None,
    ddof=1,
):
    r"""
    Marginal MCMC summary for samples of shape (M,p).

    Returns arrays of:
        mean,
        posterior SD,
        IACT,
        ESS,
        MCSE(mean).
    """
    samples = _as_samples(
        samples
    )

    M, p = samples.shape

    if not isinstance(
        ddof,
        (int, np.integer),
    ):
        raise ValueError(
            "ddof must be an integer."
        )

    if ddof < 0 or M <= ddof:
        raise ValueError(
            "Need 0 <= ddof < n_draws."
        )

    mean = np.mean(
        samples,
        axis=0,
    )

    sd = np.std(
        samples,
        axis=0,
        ddof=ddof,
    )

    iact = np.empty(
        p,
        dtype=float,
    )

    ess = np.empty(
        p,
        dtype=float,
    )

    mcse = np.empty(
        p,
        dtype=float,
    )

    for j in range(p):
        xj = samples[:, j]

        iact[j] = (
            integrated_autocorrelation_time(
                xj,
                max_lag=max_lag,
            )
        )

        ess[j] = (
            M / iact[j]
        )

        ess[j] = np.clip(
            ess[j],
            1.0,
            float(M),
        )

        if sd[j] == 0.0:
            mcse[j] = 0.0
        else:
            mcse[j] = (
                sd[j]
                / np.sqrt(
                    ess[j]
                )
            )

    return ChainSummary(
        mean=np.asarray(
            mean,
            dtype=float,
        ),
        sd=np.asarray(
            sd,
            dtype=float,
        ),
        iact=iact,
        ess=ess,
        mcse=mcse,
        n_draws=int(M),
        n_variables=int(p),
        max_lag=(
            None
            if max_lag is None
            else int(max_lag)
        ),
    )


# ============================================================
# SAPG tail diagnostics
# ============================================================

def sapg_tail_summary(
    eta_path,
    theta_path,
    score_path,
    bound_hit_path,
    *,
    tail_start=None,
):
    r"""
    Summarise tail behaviour of an SAPG run.

    Conventions
    -----------
    eta_path and theta_path have length n_iter+1.

    score_path and bound_hit_path have length n_iter.

    If tail_start=s, diagnostics use:
        eta_path[s:],
        theta_path[s:],
        score_path[s:],
        bound_hit_path[s:].

    This mirrors the averaging convention used in sapg.py.

    Returns
    -------
    SapgTailSummary
    """
    eta_path = _as_1d_finite(
        eta_path,
        "eta_path",
    )

    theta_path = _as_1d_finite(
        theta_path,
        "theta_path",
    )

    score_path = _as_1d_finite(
        score_path,
        "score_path",
    )

    bound_hit_path = np.asarray(
        bound_hit_path
    )

    if bound_hit_path.ndim != 1:
        raise ValueError(
            "bound_hit_path must be one-dimensional."
        )

    if not np.all(
        np.isin(
            bound_hit_path,
            [-1, 0, 1],
        )
    ):
        raise ValueError(
            "bound_hit_path entries must belong to {-1,0,1}."
        )

    n_iter = score_path.size

    if eta_path.size != n_iter + 1:
        raise ValueError(
            "eta_path must have length len(score_path)+1."
        )

    if theta_path.size != n_iter + 1:
        raise ValueError(
            "theta_path must have length len(score_path)+1."
        )

    if bound_hit_path.size != n_iter:
        raise ValueError(
            "bound_hit_path must have length len(score_path)."
        )

    if tail_start is None:
        tail_start = n_iter // 2

    if not isinstance(
        tail_start,
        (int, np.integer),
    ):
        raise ValueError(
            "tail_start must be an integer."
        )

    if (
        tail_start < 0
        or tail_start >= n_iter
    ):
        raise ValueError(
            "tail_start must satisfy 0 <= tail_start < n_iter."
        )

    eta_tail = eta_path[
        tail_start:
    ]

    theta_tail = theta_path[
        tail_start:
    ]

    score_tail = score_path[
        tail_start:
    ]

    bound_tail = bound_hit_path[
        tail_start:
    ]

    # Linear least-squares slope of eta versus iteration index.
    x = np.arange(
        eta_tail.size,
        dtype=float,
    )

    x_centered = (
        x - np.mean(x)
    )

    eta_centered = (
        eta_tail
        - np.mean(eta_tail)
    )

    denominator = float(
        x_centered @ x_centered
    )

    if denominator == 0.0:
        eta_slope = 0.0
    else:
        eta_slope = float(
            (
                x_centered
                @ eta_centered
            )
            / denominator
        )

    eta_mean = float(
        np.mean(eta_tail)
    )

    eta_sd = float(
        np.std(
            eta_tail,
            ddof=1,
        )
    ) if eta_tail.size > 1 else 0.0

    theta_mean = float(
        np.mean(theta_tail)
    )

    theta_sd = float(
        np.std(
            theta_tail,
            ddof=1,
        )
    ) if theta_tail.size > 1 else 0.0

    theta_cv = (
        theta_sd / theta_mean
        if theta_mean != 0.0
        else np.inf
    )

    score_mean = float(
        np.mean(score_tail)
    )

    score_sd = float(
        np.std(
            score_tail,
            ddof=1,
        )
    ) if score_tail.size > 1 else 0.0

    score_abs_mean = float(
        np.mean(
            np.abs(score_tail)
        )
    )

    lower_fraction = float(
        np.mean(
            bound_tail == -1
        )
    )

    upper_fraction = float(
        np.mean(
            bound_tail == 1
        )
    )

    any_fraction = float(
        np.mean(
            bound_tail != 0
        )
    )

    return SapgTailSummary(
        tail_start=int(
            tail_start
        ),

        n_tail=int(
            eta_tail.size
        ),

        eta_mean=eta_mean,
        eta_sd=eta_sd,
        eta_slope_per_iter=eta_slope,

        theta_mean=theta_mean,
        theta_sd=theta_sd,
        theta_cv=float(theta_cv),

        score_mean=score_mean,
        score_sd=score_sd,
        score_abs_mean=score_abs_mean,

        lower_bound_hit_fraction=lower_fraction,
        upper_bound_hit_fraction=upper_fraction,
        any_bound_hit_fraction=any_fraction,
    )


# ============================================================
# MAP discrepancy
# ============================================================

def map_discrepancy(
    beta_map,
    beta_map_lambda,
    *,
    threshold=None,
):
    r"""
    Compare the primary unsmoothed MAP and the smoothed MAP.

    Returns absolute and relative L2 / Linf discrepancies.

    If `threshold` is supplied, also reports

        ||beta_MAP_lambda - beta_MAP||_inf / threshold.

    This is useful because small absolute MAP differences can be
    consequential when they occur near the downstream gate.

    `sign_disagreements` counts coordinates whose nonzero signs
    differ. Exact zeros are assigned sign 0.
    """
    beta_map = _as_1d_finite(
        beta_map,
        "beta_map",
    )

    beta_map_lambda = _as_1d_finite(
        beta_map_lambda,
        "beta_map_lambda",
    )

    if (
        beta_map.size
        != beta_map_lambda.size
    ):
        raise ValueError(
            "MAP vectors must have the same length."
        )

    diff = (
        beta_map_lambda
        - beta_map
    )

    l2 = float(
        np.linalg.norm(diff)
    )

    linf = float(
        np.max(
            np.abs(diff)
        )
    )

    base_l2 = float(
        np.linalg.norm(beta_map)
    )

    base_linf = float(
        np.max(
            np.abs(beta_map)
        )
    )

    relative_l2 = (
        l2
        / max(1.0, base_l2)
    )

    relative_linf = (
        linf
        / max(1.0, base_linf)
    )

    sign_disagreements = int(
        np.count_nonzero(
            np.sign(beta_map)
            != np.sign(beta_map_lambda)
        )
    )

    if threshold is None:
        threshold_value = None
        threshold_scaled_linf = None

    else:
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

        threshold_value = threshold

        if threshold == 0.0:
            threshold_scaled_linf = (
                0.0
                if linf == 0.0
                else np.inf
            )
        else:
            threshold_scaled_linf = (
                linf / threshold
            )

    return MapDiscrepancy(
        l2=l2,
        relative_l2=float(
            relative_l2
        ),
        linf=linf,
        relative_linf=float(
            relative_linf
        ),

        threshold=threshold_value,
        threshold_scaled_linf=(
            None
            if threshold_scaled_linf is None
            else float(
                threshold_scaled_linf
            )
        ),

        sign_disagreements=sign_disagreements,
        n_variables=int(
            beta_map.size
        ),
    )


# ============================================================
# Support stability
# ============================================================

def support_stability(
    support_masks,
):
    r"""
    Pairwise stability of support decisions across settings
    such as different lambda values.

    Parameters
    ----------
    support_masks : array_like, shape (K,p)
        Boolean support indicators.

    Returns
    -------
    SupportStability

    Jaccard similarity:
        |A intersect B| / |A union B|.

    If both supports are empty, Jaccard similarity is defined
    as 1.

    Normalized Hamming disagreement:
        number of differing coordinates / p.
    """
    masks = np.asarray(
        support_masks
    )

    if masks.ndim != 2:
        raise ValueError(
            "support_masks must have shape (K,p)."
        )

    if (
        masks.shape[0] == 0
        or masks.shape[1] == 0
    ):
        raise ValueError(
            "support_masks must have non-zero dimensions."
        )

    masks = masks.astype(
        bool,
        copy=False,
    )

    K, p = masks.shape

    jaccard = np.empty(
        (K, K),
        dtype=float,
    )

    hamming = np.empty(
        (K, K),
        dtype=float,
    )

    for i in range(K):
        for j in range(K):
            intersection = int(
                np.count_nonzero(
                    masks[i] & masks[j]
                )
            )

            union = int(
                np.count_nonzero(
                    masks[i] | masks[j]
                )
            )

            if union == 0:
                jaccard[i, j] = 1.0
            else:
                jaccard[i, j] = (
                    intersection / union
                )

            hamming[i, j] = (
                np.count_nonzero(
                    masks[i] != masks[j]
                )
                / p
            )

    support_sizes = np.sum(
        masks,
        axis=1,
    ).astype(int)

    if K == 1:
        offdiag_jaccard = np.array(
            [1.0]
        )

        offdiag_hamming = np.array(
            [0.0]
        )

    else:
        offdiag = ~np.eye(
            K,
            dtype=bool,
        )

        offdiag_jaccard = (
            jaccard[offdiag]
        )

        offdiag_hamming = (
            hamming[offdiag]
        )

    return SupportStability(
        jaccard_matrix=jaccard,
        hamming_matrix=hamming,

        mean_offdiag_jaccard=float(
            np.mean(
                offdiag_jaccard
            )
        ),

        min_offdiag_jaccard=float(
            np.min(
                offdiag_jaccard
            )
        ),

        mean_offdiag_hamming=float(
            np.mean(
                offdiag_hamming
            )
        ),

        max_offdiag_hamming=float(
            np.max(
                offdiag_hamming
            )
        ),

        support_sizes=support_sizes,
        n_settings=int(K),
        n_variables=int(p),
    )
