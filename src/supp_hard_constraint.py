"""
Supplement: hard homogeneous constraint.

This script reproduces the hard sum-zero experiment reported in the
supplement. It compares empirical-Bayes calibration using the intrinsic
dimension d = p - 1 with a deliberate ambient-dimensional comparator
that uses p in the SAPG score.

The two branches use the same data, initial state, numerical settings
and random-number streams. After calibration, the script compares the
constrained MAPs, posterior summaries and final support decisions.

Numerical summaries are printed to the terminal. No intermediate
result files are written.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from diagnostics import effective_sample_size, mcse_mean, sapg_tail_summary
from fista import fista_map_sum_zero, smoothed_map_sum_zero
from myula import myula_sum_zero
from sapg import sapg_laplace
from selection import activation_probabilities, posterior_scale_threshold, posterior_sd


# ============================================================
# Experiment settings
# ============================================================

MASTER_SEED = 2026090202
R = 100

N = 200
P = 10
SIGMA = 1.0
SIGMA2 = SIGMA**2

BETA_TRUE = np.array(
    [1.0, -1.0, 0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
TRUE_SUPPORT = np.flatnonzero(BETA_TRUE != 0.0)
TRUE_MASK = np.zeros(P, dtype=bool)
TRUE_MASK[TRUE_SUPPORT] = True

INTRINSIC_DIM = P - 1
AMBIENT_DIM = P
DIMENSION_RATIO = AMBIENT_DIM / INTRINSIC_DIM

SAPG_N_ITER = 15_000
SAPG_AVERAGE_START = 4_000
SAPG_STEP_SCALE = 1.0
SAPG_STEP_OFFSET = 200.0
SAPG_BOUND_FACTOR = 1e6

C_DELTA = 0.9

MAP_TOL = 1e-8
MAP_MAX_ITER = 20_000

POST_BURN_IN = 4_000
POST_N_SAMPLES = 4_000
POST_THIN = 1

K_GRID = np.array([2.0, 2.5, 3.0], dtype=float)
PI_GRID = np.array([0.50, 0.75, 0.90], dtype=float)
K_REFERENCE = 2.5
PI_REFERENCE = 0.75

DATA_SEED_BASE = MASTER_SEED + 10_000_000
SAPG_SEED_BASE = MASTER_SEED + 20_000_000
POSTERIOR_SEED_BASE = MASTER_SEED + 30_000_000


# ============================================================
# Sum-zero geometry
# ============================================================

# Q has orthonormal columns spanning {beta : 1^T beta = 0}.
_, _, vt = np.linalg.svd(
    np.ones(P, dtype=float).reshape(1, -1),
    full_matrices=True,
)
Q = vt[1:, :].T

assert Q.shape == (P, INTRINSIC_DIM)
assert np.allclose(Q.T @ Q, np.eye(INTRINSIC_DIM))
assert np.allclose(Q.T @ np.ones(P), 0.0)
assert np.isclose(BETA_TRUE.sum(), 0.0)


# ============================================================
# Small helpers
# ============================================================

def support_metrics(mask: np.ndarray) -> dict:
    """Return the support quantities used in the supplement."""
    mask = np.asarray(mask, dtype=bool)

    tp = int(np.count_nonzero(mask & TRUE_MASK))
    fp = int(np.count_nonzero(mask & ~TRUE_MASK))
    size = int(np.count_nonzero(mask))

    return {
        "size": size,
        "tpr": tp / TRUE_SUPPORT.size,
        "fdp": fp / max(size, 1),
        "exact": bool(np.array_equal(mask, TRUE_MASK)),
    }


def relative_l2_error(beta: np.ndarray) -> float:
    """Relative Euclidean error against the true coefficient vector."""
    return float(
        np.linalg.norm(beta - BETA_TRUE)
        / np.linalg.norm(BETA_TRUE)
    )


def support_rmse(beta: np.ndarray) -> float:
    """RMSE on the four true signal coordinates."""
    error = beta[TRUE_SUPPORT] - BETA_TRUE[TRUE_SUPPORT]
    return float(np.sqrt(np.mean(error**2)))


def prediction_risk(beta: np.ndarray, X: np.ndarray) -> float:
    """Mean squared prediction error relative to the true signal."""
    error = beta - BETA_TRUE
    return float(np.mean((X @ error) ** 2))


def paired_mean_mcse(samples_a: np.ndarray, samples_b: np.ndarray) -> np.ndarray:
    """MCSE of coordinatewise paired posterior-mean differences."""
    difference = np.asarray(samples_b - samples_a, dtype=float)
    out = np.empty(P, dtype=float)

    for j in range(P):
        out[j] = mcse_mean(difference[:, j], ddof=1)

    return out


def coordinate_ess(samples: np.ndarray) -> np.ndarray:
    """Coordinatewise effective sample sizes."""
    out = np.empty(P, dtype=float)

    for j in range(P):
        out[j] = effective_sample_size(samples[:, j])

    return out


def describe(values: np.ndarray) -> dict:
    """Common five-number ensemble summary."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ============================================================
# One paired replicate
# ============================================================

def run_replicate(replicate: int) -> tuple[dict, list[dict]]:
    """Run one complete intrinsic-versus-ambient comparison."""
    replicate = int(replicate)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    rng_data = np.random.default_rng(DATA_SEED_BASE + replicate)

    X = rng_data.normal(size=(N, P))
    X *= np.sqrt(N) / np.linalg.norm(X, axis=0)

    noise = rng_data.normal(scale=SIGMA, size=N)
    y = X @ BETA_TRUE + noise

    empirical_snr = float(
        np.mean((X @ BETA_TRUE) ** 2)
        / np.mean(noise**2)
    )

    # --------------------------------------------------------
    # Constrained least squares and restricted curvature
    # --------------------------------------------------------
    XQ = X @ Q
    z_cls, _, _, _ = np.linalg.lstsq(XQ, y, rcond=None)
    beta_cls = Q @ z_cls

    theta_init = float(
        INTRINSIC_DIM / np.sum(np.abs(beta_cls))
    )

    restricted_eigenvalues = np.linalg.eigvalsh(XQ.T @ XQ / SIGMA2)
    restricted_condition_number = float(
        restricted_eigenvalues[-1] / restricted_eigenvalues[0]
    )

    singular_values = np.linalg.svd(XQ, compute_uv=False)
    Lf_restricted = float(singular_values[0] ** 2 / SIGMA2)

    lambda_my = 1.0 / Lf_restricted
    delta_myula = C_DELTA / (Lf_restricted + 1.0 / lambda_my)

    # --------------------------------------------------------
    # Paired SAPG calibration
    # --------------------------------------------------------
    sapg_seed = SAPG_SEED_BASE + replicate
    rng_intrinsic = np.random.default_rng(sapg_seed)
    rng_ambient = np.random.default_rng(sapg_seed)

    sapg_common = dict(
        y=y,
        X=X,
        sigma2=SIGMA2,
        n_iter=SAPG_N_ITER,
        theta_init=theta_init,
        beta_ref=beta_cls,
        beta0=beta_cls,
        constraint="sum_zero",
        bound_factor=SAPG_BOUND_FACTOR,
        step_scale=SAPG_STEP_SCALE,
        step_offset=SAPG_STEP_OFFSET,
        step_power=1.0,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=1,
        mcmc_warmup=0,
        lambda_my=lambda_my,
        step_size_myula=delta_myula,
        c_delta=C_DELTA,
        store_state_path=False,
    )

    sapg_intrinsic = sapg_laplace(
        rng=rng_intrinsic,
        intrinsic_dim=INTRINSIC_DIM,
        **sapg_common,
    )
    # The ambient value is supplied deliberately for this comparison.
    # sapg_laplace warns whenever the geometric default is overridden;
    # silence that expected warning here so the terminal output stays clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="intrinsic_dim differs from the geometric default.*",
            category=RuntimeWarning,
        )
        sapg_ambient = sapg_laplace(
            rng=rng_ambient,
            intrinsic_dim=AMBIENT_DIM,
            **sapg_common,
        )

    theta_intrinsic = float(sapg_intrinsic.theta_hat)
    theta_ambient = float(sapg_ambient.theta_hat)
    theta_ratio = theta_ambient / theta_intrinsic

    diag_intrinsic = sapg_tail_summary(
        sapg_intrinsic.eta_path,
        sapg_intrinsic.theta_path,
        sapg_intrinsic.score_path,
        sapg_intrinsic.bound_hit_path,
        tail_start=SAPG_AVERAGE_START,
    )
    diag_ambient = sapg_tail_summary(
        sapg_ambient.eta_path,
        sapg_ambient.theta_path,
        sapg_ambient.score_path,
        sapg_ambient.bound_hit_path,
        tail_start=SAPG_AVERAGE_START,
    )

    g_intrinsic = float(
        np.mean(sapg_intrinsic.g_path[SAPG_AVERAGE_START:])
    )
    g_ambient = float(
        np.mean(sapg_ambient.g_path[SAPG_AVERAGE_START:])
    )

    moment_ratio_prediction = float(
        DIMENSION_RATIO * g_intrinsic / g_ambient
    )
    moment_relative_error = float(
        (theta_ratio - moment_ratio_prediction)
        / moment_ratio_prediction
    )

    # --------------------------------------------------------
    # Nonsmoothed and smoothed constrained MAPs
    # --------------------------------------------------------
    map_intrinsic = fista_map_sum_zero(
        y,
        X,
        SIGMA2,
        theta_intrinsic,
        beta0=beta_cls,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
    )
    map_ambient = fista_map_sum_zero(
        y,
        X,
        SIGMA2,
        theta_ambient,
        beta0=beta_cls,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
    )

    beta_map_intrinsic = np.asarray(map_intrinsic.beta, dtype=float)
    beta_map_ambient = np.asarray(map_ambient.beta, dtype=float)

    smooth_intrinsic = smoothed_map_sum_zero(
        y,
        X,
        SIGMA2,
        theta_intrinsic,
        lambda_my,
        beta0=beta_map_intrinsic,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
    )
    smooth_ambient = smoothed_map_sum_zero(
        y,
        X,
        SIGMA2,
        theta_ambient,
        lambda_my,
        beta0=beta_map_ambient,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
    )

    beta_smooth_intrinsic = np.asarray(smooth_intrinsic.beta, dtype=float)
    beta_smooth_ambient = np.asarray(smooth_ambient.beta, dtype=float)

    map_branch_l2 = float(
        np.linalg.norm(beta_map_ambient - beta_map_intrinsic)
    )

    # --------------------------------------------------------
    # Paired posterior calculation
    # --------------------------------------------------------
    posterior_seed = POSTERIOR_SEED_BASE + replicate
    rng_post_intrinsic = np.random.default_rng(posterior_seed)
    rng_post_ambient = np.random.default_rng(posterior_seed)

    post_intrinsic = myula_sum_zero(
        y,
        X,
        SIGMA2,
        theta_intrinsic,
        rng=rng_post_intrinsic,
        n_samples=POST_N_SAMPLES,
        burn_in=POST_BURN_IN,
        thin=POST_THIN,
        beta0=beta_map_intrinsic,
        lambda_my=lambda_my,
        step_size=delta_myula,
        c_delta=C_DELTA,
        store_potential=False,
    )
    post_ambient = myula_sum_zero(
        y,
        X,
        SIGMA2,
        theta_ambient,
        rng=rng_post_ambient,
        n_samples=POST_N_SAMPLES,
        burn_in=POST_BURN_IN,
        thin=POST_THIN,
        beta0=beta_map_ambient,
        lambda_my=lambda_my,
        step_size=delta_myula,
        c_delta=C_DELTA,
        store_potential=False,
    )

    samples_intrinsic = np.asarray(post_intrinsic.samples, dtype=float)
    samples_ambient = np.asarray(post_ambient.samples, dtype=float)

    posterior_mean_intrinsic = np.mean(samples_intrinsic, axis=0)
    posterior_mean_ambient = np.mean(samples_ambient, axis=0)
    posterior_sd_intrinsic = posterior_sd(samples_intrinsic)
    posterior_sd_ambient = posterior_sd(samples_ambient)

    ess_intrinsic = coordinate_ess(samples_intrinsic)
    ess_ambient = coordinate_ess(samples_ambient)
    paired_mcse = paired_mean_mcse(samples_intrinsic, samples_ambient)

    # --------------------------------------------------------
    # Full decision grid
    # --------------------------------------------------------
    grid_rows: list[dict] = []

    for k in K_GRID:
        tau_intrinsic, _ = posterior_scale_threshold(samples_intrinsic, k)
        tau_ambient, _ = posterior_scale_threshold(samples_ambient, k)

        pi_intrinsic = activation_probabilities(
            samples_intrinsic,
            tau_intrinsic,
        )
        pi_ambient = activation_probabilities(
            samples_ambient,
            tau_ambient,
        )

        for pi_star in PI_GRID:
            selected_intrinsic = (
                (np.abs(beta_map_intrinsic) >= tau_intrinsic)
                & (pi_intrinsic >= pi_star)
            )
            selected_ambient = (
                (np.abs(beta_map_ambient) >= tau_ambient)
                & (pi_ambient >= pi_star)
            )

            metrics_intrinsic = support_metrics(selected_intrinsic)
            metrics_ambient = support_metrics(selected_ambient)

            grid_rows.append(
                {
                    "replicate": replicate,
                    "k": float(k),
                    "pi_star": float(pi_star),
                    "tpr_intrinsic": metrics_intrinsic["tpr"],
                    "tpr_ambient": metrics_ambient["tpr"],
                    "fdp_intrinsic": metrics_intrinsic["fdp"],
                    "fdp_ambient": metrics_ambient["fdp"],
                    "size_intrinsic": metrics_intrinsic["size"],
                    "size_ambient": metrics_ambient["size"],
                    "exact_intrinsic": metrics_intrinsic["exact"],
                    "exact_ambient": metrics_ambient["exact"],
                    "same_support": bool(
                        np.array_equal(selected_intrinsic, selected_ambient)
                    ),
                }
            )

    # --------------------------------------------------------
    # Replicate-level quantities used in the supplement
    # --------------------------------------------------------
    summary = {
        "replicate": replicate,
        "empirical_snr": empirical_snr,
        "restricted_condition_number": restricted_condition_number,
        "theta_intrinsic": theta_intrinsic,
        "theta_ambient": theta_ambient,
        "theta_ratio": theta_ratio,
        "moment_ratio_prediction": moment_ratio_prediction,
        "moment_relative_error": moment_relative_error,
        "eta_slope_intrinsic": diag_intrinsic.eta_slope_per_iter,
        "eta_slope_ambient": diag_ambient.eta_slope_per_iter,
        "bound_hit_intrinsic": diag_intrinsic.any_bound_hit_fraction,
        "bound_hit_ambient": diag_ambient.any_bound_hit_fraction,
        "constraint_residual_cls": float(abs(beta_cls.sum())),
        "constraint_residual_sapg_intrinsic": float(
            abs(sapg_intrinsic.beta_final.sum())
        ),
        "constraint_residual_sapg_ambient": float(
            abs(sapg_ambient.beta_final.sum())
        ),
        "map_l1_ratio": float(
            np.sum(np.abs(beta_map_ambient))
            / np.sum(np.abs(beta_map_intrinsic))
        ),
        "map_branch_l2": map_branch_l2,
        "map_branch_l2_over_truth_norm": float(
            map_branch_l2 / np.linalg.norm(BETA_TRUE)
        ),
        "relative_l2_difference": float(
            relative_l2_error(beta_map_ambient)
            - relative_l2_error(beta_map_intrinsic)
        ),
        "support_rmse_difference": float(
            support_rmse(beta_map_ambient)
            - support_rmse(beta_map_intrinsic)
        ),
        "prediction_risk_difference": float(
            prediction_risk(beta_map_ambient, X)
            - prediction_risk(beta_map_intrinsic, X)
        ),
        "smooth_shift_intrinsic": float(
            np.linalg.norm(beta_smooth_intrinsic - beta_map_intrinsic)
        ),
        "smooth_shift_ambient": float(
            np.linalg.norm(beta_smooth_ambient - beta_map_ambient)
        ),
        "posterior_sd_ratio": float(
            np.median(posterior_sd_ambient)
            / np.median(posterior_sd_intrinsic)
        ),
        "posterior_mean_branch_l2": float(
            np.linalg.norm(posterior_mean_ambient - posterior_mean_intrinsic)
        ),
        "posterior_sd_branch_l2": float(
            np.linalg.norm(posterior_sd_ambient - posterior_sd_intrinsic)
        ),
        "median_ess_intrinsic": float(np.median(ess_intrinsic)),
        "median_ess_ambient": float(np.median(ess_ambient)),
        "median_paired_mean_mcse": float(np.median(paired_mcse)),
        "max_constraint_posterior_intrinsic": float(
            np.max(np.abs(np.sum(samples_intrinsic, axis=1)))
        ),
        "max_constraint_posterior_ambient": float(
            np.max(np.abs(np.sum(samples_ambient, axis=1)))
        ),
        "all_maps_converged": bool(
            map_intrinsic.converged
            and map_ambient.converged
            and smooth_intrinsic.converged
            and smooth_ambient.converged
        ),
    }

    return summary, grid_rows


# ============================================================
# Ensemble run
# ============================================================

def main() -> None:
    """Run all paired replicates and print the reported summaries."""
    start = time.perf_counter()

    summaries: list[dict] = []
    grid_rows: list[dict] = []

    print("Hard homogeneous constraint")
    print("---------------------------")
    print(f"R={R}, n={N}, p={P}, d={INTRINSIC_DIM}, sigma={SIGMA:g}")
    print(f"p/d reference ratio: {DIMENSION_RATIO:.6f}")
    print()

    for replicate in range(R):
        summary, replicate_grid = run_replicate(replicate)
        summaries.append(summary)
        grid_rows.extend(replicate_grid)

        if (replicate + 1) % 10 == 0 or replicate == 0:
            print(
                f"[{replicate + 1:03d}/{R:03d}] "
                f"theta_d={summary['theta_intrinsic']:.4f}, "
                f"theta_p={summary['theta_ambient']:.4f}, "
                f"ratio={summary['theta_ratio']:.5f}"
            )

    summary_df = pd.DataFrame(summaries)
    grid_df = pd.DataFrame(grid_rows)

    # --------------------------------------------------------
    # Calibration table
    # --------------------------------------------------------
    calibration = pd.DataFrame(
        {
            "theta_intrinsic": describe(summary_df["theta_intrinsic"]),
            "theta_ambient": describe(summary_df["theta_ambient"]),
            "theta_ratio": describe(summary_df["theta_ratio"]),
            "moment_prediction": describe(summary_df["moment_ratio_prediction"]),
        }
    ).T

    print("\nCalibration summary")
    print("-------------------")
    print(calibration.to_string(float_format=lambda x: f"{x:.6f}"))

    print(
        "\nMean absolute relative moment error: "
        f"{np.mean(np.abs(summary_df['moment_relative_error'])):.6e}"
    )
    print(
        "Maximum absolute relative moment error: "
        f"{np.max(np.abs(summary_df['moment_relative_error'])):.6e}"
    )
    print(
        "Fraction theta_ambient > theta_intrinsic: "
        f"{np.mean(summary_df['theta_ambient'] > summary_df['theta_intrinsic']):.3f}"
    )
    print(
        "Fraction theta ratio > p/d: "
        f"{np.mean(summary_df['theta_ratio'] > DIMENSION_RATIO):.3f}"
    )

    # --------------------------------------------------------
    # Geometry and SAPG checks
    # --------------------------------------------------------
    max_bound_hit = float(
        np.max(
            summary_df[["bound_hit_intrinsic", "bound_hit_ambient"]].to_numpy()
        )
    )
    max_constraint = float(
        np.max(
            summary_df[
                [
                    "constraint_residual_cls",
                    "constraint_residual_sapg_intrinsic",
                    "constraint_residual_sapg_ambient",
                    "max_constraint_posterior_intrinsic",
                    "max_constraint_posterior_ambient",
                ]
            ].to_numpy()
        )
    )

    print("\nGeometry and numerical checks")
    print("-----------------------------")
    print(
        "Mean restricted condition number: "
        f"{summary_df['restricted_condition_number'].mean():.4f}"
    )
    print(
        "Mean empirical SNR: "
        f"{summary_df['empirical_snr'].mean():.4f}"
    )
    print(
        "Median |eta slope|, intrinsic: "
        f"{np.median(np.abs(summary_df['eta_slope_intrinsic'])):.3e}"
    )
    print(
        "Median |eta slope|, ambient: "
        f"{np.median(np.abs(summary_df['eta_slope_ambient'])):.3e}"
    )
    print(f"Maximum SAPG bound-hit fraction: {max_bound_hit:.3e}")
    print(f"Maximum sum-zero residual: {max_constraint:.3e}")
    print(
        "All exact and smoothed MAP solves converged: "
        f"{bool(summary_df['all_maps_converged'].all())}"
    )

    # --------------------------------------------------------
    # Downstream MAP and posterior consequences
    # --------------------------------------------------------
    print("\nDownstream consequences")
    print("-----------------------")
    print(
        "Mean MAP L1 ratio, ambient/intrinsic: "
        f"{summary_df['map_l1_ratio'].mean():.6f}"
    )
    print(
        "Fraction MAP L1 ratio below one: "
        f"{np.mean(summary_df['map_l1_ratio'] < 1.0):.3f}"
    )
    print(
        "Mean MAP branch displacement: "
        f"{summary_df['map_branch_l2'].mean():.6e}"
    )
    print(
        "Median MAP displacement / ||truth||: "
        f"{summary_df['map_branch_l2_over_truth_norm'].median():.6e}"
    )
    print(
        "Mean relative-L2 error difference (ambient - intrinsic): "
        f"{summary_df['relative_l2_difference'].mean():+.6e}"
    )
    print(
        "Mean true-support RMSE difference (ambient - intrinsic): "
        f"{summary_df['support_rmse_difference'].mean():+.6e}"
    )
    print(
        "Mean prediction-risk difference (ambient - intrinsic): "
        f"{summary_df['prediction_risk_difference'].mean():+.6e}"
    )
    print(
        "Median Moreau-MAP shift, intrinsic: "
        f"{summary_df['smooth_shift_intrinsic'].median():.6e}"
    )
    print(
        "Median Moreau-MAP shift, ambient: "
        f"{summary_df['smooth_shift_ambient'].median():.6e}"
    )
    print(
        "Median posterior-SD ratio, ambient/intrinsic: "
        f"{summary_df['posterior_sd_ratio'].median():.6f}"
    )
    print(
        "Mean posterior-mean branch displacement: "
        f"{summary_df['posterior_mean_branch_l2'].mean():.6e}"
    )
    print(
        "Mean posterior-SD branch displacement: "
        f"{summary_df['posterior_sd_branch_l2'].mean():.6e}"
    )
    print(
        "Median coordinate ESS, intrinsic: "
        f"{summary_df['median_ess_intrinsic'].median():.1f}"
    )
    print(
        "Median coordinate ESS, ambient: "
        f"{summary_df['median_ess_ambient'].median():.1f}"
    )
    print(
        "Median paired-difference MCSE: "
        f"{summary_df['median_paired_mean_mcse'].median():.6e}"
    )

    # --------------------------------------------------------
    # Full decision grid
    # --------------------------------------------------------
    grid_summary = (
        grid_df.groupby(["k", "pi_star"], as_index=False)
        .agg(
            tpr_intrinsic=("tpr_intrinsic", "mean"),
            tpr_ambient=("tpr_ambient", "mean"),
            fdr_intrinsic=("fdp_intrinsic", "mean"),
            fdr_ambient=("fdp_ambient", "mean"),
            size_intrinsic=("size_intrinsic", "mean"),
            size_ambient=("size_ambient", "mean"),
            exact_intrinsic=("exact_intrinsic", "mean"),
            exact_ambient=("exact_ambient", "mean"),
            same=("same_support", "mean"),
        )
        .sort_values(["k", "pi_star"])
        .reset_index(drop=True)
    )

    print("\nDecision grid")
    print("-------------")
    print(
        grid_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    reference = grid_summary.loc[
        np.isclose(grid_summary["k"], K_REFERENCE)
        & np.isclose(grid_summary["pi_star"], PI_REFERENCE)
    ].iloc[0]

    print("\nReference decision")
    print("------------------")
    print(
        f"Exact-support rate, intrinsic: {reference['exact_intrinsic']:.3f}"
    )
    print(
        f"Exact-support rate, ambient:   {reference['exact_ambient']:.3f}"
    )
    print(f"Same-support rate:              {reference['same']:.3f}")

    print(f"\nTotal wall time: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
