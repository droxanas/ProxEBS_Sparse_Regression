"""
Study 1: end-to-end sparse regression.

This script reproduces the final Study 1 experiment in the paper.
It runs both regression regimes, uses the final 40,000-draw MYULA
calculation, and obtains the reported 4,000-draw comparison from the
first 4,000 retained draws of the same chain.

The script writes only the four Study 1 figures used in the paper or
supplement. Numerical summaries are printed to the terminal. No
intermediate result files are written.
"""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics import (
    chain_summary,
    effective_sample_size,
    map_discrepancy,
    mcse_mean,
    sapg_tail_summary,
)
from fista import fista_map, smoothed_map
from myula import myula
from noise import residual_variance, scaled_lasso_variance
from sapg import sapg_laplace
from selection import activation_probabilities, posterior_scale_threshold
from simulation import simulate_sparse_regression


# ============================================================
# Study settings
# ============================================================

MASTER_SEED = 2026090102
R_PER_REGIME = 100

REGIMES = (
    {
        "name": "low_dim",
        "label": r"$n=500,\ p=200$",
        "seed_code": 101,
        "n": 500,
        "p": 200,
        "noise_method": "residual",
    },
    {
        "name": "high_dim",
        "label": r"$n=200,\ p=500$",
        "seed_code": 202,
        "n": 200,
        "p": 500,
        "noise_method": "scaled_lasso_postols",
    },
)

RHO = 0.4
SIGMA_TRUE = 1.0
TARGET_SNR = 3.0

# Zero-based signal locations. The same ten signals are used in both
# regimes; the p=500 regime simply has another 300 null predictors.
SUPPORT_TRUE = np.array(
    [10, 30, 50, 70, 90, 110, 130, 150, 170, 190],
    dtype=int,
)

# The simulator rescales this template so that the population SNR is 3.
BETA_TEMPLATE = np.array(
    [1.4, -1.3, 1.2, -1.1, 1.0, -0.9, 0.8, -0.7, 0.6, -0.5],
    dtype=float,
)

K_GRID = np.array([2.0, 2.5, 3.0], dtype=float)
PI_GRID = np.array([0.50, 0.75, 0.90], dtype=float)
K_REFERENCE = 2.5
PI_REFERENCE = 0.75

SAPG_N_ITER = 4000
SAPG_AVERAGE_START = 2000
SAPG_MCMC_STEPS = 1
SAPG_MCMC_WARMUP = 500
SAPG_STEP_SCALE = 1.0
SAPG_STEP_OFFSET = 200.0
SAPG_STEP_POWER = 1.0
SAPG_BOUND_FACTOR = 1e6

C_DELTA = 0.9

MAP_TOL = 1e-8
MAP_MAX_ITER = 30000

SHORT_N_SAMPLES = 4_000
LONG_N_SAMPLES = 40_000
POST_BURN_IN = 2_000
POST_THIN = 1
DIAGNOSTIC_MAX_LAG = 1000

POSTOLS_SUPPORT_TOL = 1e-12

# Run the script from the repository root. Figures are written here.
FIGURE_DIR = Path("figures")


# ============================================================
# Small study-specific helpers
# ============================================================

def ar1_covariance(p: int, rho: float) -> np.ndarray:
    """Return the AR(1) population covariance matrix."""
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


def post_selection_ols_variance(
    y: np.ndarray,
    X: np.ndarray,
    beta_selector: np.ndarray,
    *,
    support_tol: float = POSTOLS_SUPPORT_TOL,
) -> dict:
    """
    Use the scaled-Lasso support, refit OLS on that support, and
    estimate sigma^2 as RSS / (n - rank(X_S)).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta_selector = np.asarray(beta_selector, dtype=float)

    support = np.flatnonzero(np.abs(beta_selector) > support_tol)

    if support.size == 0:
        residual = y.copy()
        rank_selected = 0
    else:
        X_selected = X[:, support]
        beta_selected, _, rank_selected, _ = np.linalg.lstsq(
            X_selected,
            y,
            rcond=None,
        )
        residual = y - X_selected @ beta_selected

    rss = float(residual @ residual)
    df_resid = int(X.shape[0] - rank_selected)

    if df_resid <= 0:
        raise RuntimeError(
            "Post-selection OLS has no residual degrees of freedom."
        )

    sigma2 = rss / df_resid

    return {
        "support": support,
        "support_size": int(support.size),
        "rank_selected": int(rank_selected),
        "df_resid": df_resid,
        "sigma2": float(sigma2),
        "sigma": float(np.sqrt(sigma2)),
    }


def activation_indicator_diagnostics(
    samples: np.ndarray,
    threshold: float,
    *,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ESS and MCSE for I{|beta_j| >= threshold}, one
    coordinate at a time.
    """
    indicators = (np.abs(samples) >= threshold).astype(float)
    p = indicators.shape[1]

    ess = np.empty(p, dtype=float)
    mcse = np.empty(p, dtype=float)

    for j in range(p):
        ess[j] = effective_sample_size(
            indicators[:, j],
            max_lag=max_lag,
        )
        mcse[j] = mcse_mean(
            indicators[:, j],
            max_lag=max_lag,
            ddof=1,
        )

    return ess, mcse


def gate_margin_mcse(
    activation_probability: np.ndarray,
    activation_mcse: np.ndarray,
    pi_star: float,
) -> np.ndarray:
    """
    Return |pi_hat_j - pi_star| / MCSE(pi_hat_j).

    A constant indicator sequence has MCSE zero. If its estimated
    probability is away from the gate, the returned distance is
    infinity because the retained run shows no Monte Carlo ambiguity
    about the side of the gate.
    """
    probability = np.asarray(activation_probability, dtype=float)
    mcse = np.asarray(activation_mcse, dtype=float)

    raw_margin = np.abs(probability - pi_star)
    ratio = np.full(probability.size, np.inf, dtype=float)

    nonzero = mcse > 0.0
    ratio[nonzero] = raw_margin[nonzero] / mcse[nonzero]

    zero_zero = (mcse == 0.0) & (raw_margin == 0.0)
    ratio[zero_zero] = 0.0

    return ratio


def support_metrics(
    selected: np.ndarray,
    true_mask: np.ndarray,
) -> dict:
    """Return the support-recovery quantities used in the paper."""
    selected = np.asarray(selected, dtype=bool)

    tp = int(np.count_nonzero(selected & true_mask))
    fp = int(np.count_nonzero(selected & ~true_mask))
    fn = int(np.count_nonzero(~selected & true_mask))
    size = int(np.count_nonzero(selected))

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TPR": float(tp / np.count_nonzero(true_mask)),
        "FDP": float(fp / max(size, 1)),
        "selected_size": size,
        "exact_support": int(np.array_equal(selected, true_mask)),
    }


def evaluate_gate_grid(
    beta_map: np.ndarray,
    samples: np.ndarray,
    *,
    lambda_my: float,
) -> dict:
    """
    Evaluate the full 3 x 3 decision grid and the decision-level
    Monte Carlo diagnostic for one posterior sample.
    """
    p = beta_map.size
    posterior_sd = np.std(samples, axis=0, ddof=1)

    thresholds = np.empty(K_GRID.size, dtype=float)
    probabilities = np.empty((K_GRID.size, p), dtype=float)
    activation_ess = np.empty((K_GRID.size, p), dtype=float)
    activation_mcse = np.empty((K_GRID.size, p), dtype=float)
    support_masks = np.empty(
        (K_GRID.size, PI_GRID.size, p),
        dtype=bool,
    )

    rows = []

    for ik, k_value in enumerate(K_GRID):
        threshold, sd_check = posterior_scale_threshold(
            samples,
            k=float(k_value),
            ddof=1,
        )

        if not np.allclose(sd_check, posterior_sd):
            raise RuntimeError("Posterior SD calculation is inconsistent.")

        probability = activation_probabilities(samples, threshold)
        ess, mcse = activation_indicator_diagnostics(
            samples,
            threshold,
            max_lag=DIAGNOSTIC_MAX_LAG,
        )

        thresholds[ik] = threshold
        probabilities[ik] = probability
        activation_ess[ik] = ess
        activation_mcse[ik] = mcse

        map_gate = np.abs(beta_map) >= threshold

        for ip, pi_star in enumerate(PI_GRID):
            selected = map_gate & (probability >= pi_star)
            support_masks[ik, ip] = selected

            margin = gate_margin_mcse(
                probability,
                mcse,
                float(pi_star),
            )
            eligible_margin = margin[map_gate]
            finite_eligible = eligible_margin[
                np.isfinite(eligible_margin)
            ]

            min_eligible = (
                float(np.min(finite_eligible))
                if finite_eligible.size
                else np.inf
            )

            rows.append(
                {
                    "k": float(k_value),
                    "pi_star": float(pi_star),
                    "threshold": float(threshold),
                    "n_map_eligible": int(np.count_nonzero(map_gate)),
                    "min_gate_margin_mcse_eligible": min_eligible,
                    "n_eligible_within_2_mcse": int(
                        np.count_nonzero(eligible_margin < 2.0)
                    ),
                }
            )

    return {
        "posterior_sd": posterior_sd,
        "thresholds": thresholds,
        "probabilities": probabilities,
        "activation_ess": activation_ess,
        "activation_mcse": activation_mcse,
        "support_masks": support_masks,
        "rows": rows,
        "lambda_my": float(lambda_my),
    }


def grid_row(grid: dict, k: float, pi_star: float) -> tuple[int, int, dict]:
    """Locate one decision setting in an evaluated grid."""
    ik = int(np.flatnonzero(np.isclose(K_GRID, k))[0])
    ip = int(np.flatnonzero(np.isclose(PI_GRID, pi_star))[0])

    row = next(
        item
        for item in grid["rows"]
        if np.isclose(item["k"], k)
        and np.isclose(item["pi_star"], pi_star)
    )

    return ik, ip, row


# ============================================================
# One replicate
# ============================================================

def run_replicate(regime: dict, replicate: int) -> tuple[dict, list[dict], list[dict]]:
    """
    Run one complete Study 1 replicate.

    Returns:
      - one replicate-level summary;
      - nine final 40k gate rows;
      - ten signal-level rows used for the signal-profile figure.
    """
    n = int(regime["n"])
    p = int(regime["p"])
    regime_name = str(regime["name"])

    seed = np.random.SeedSequence(
        [MASTER_SEED, int(regime["seed_code"]), int(replicate)]
    )
    seed_sim, seed_sapg, seed_post = seed.spawn(3)

    rng_sim = np.random.default_rng(seed_sim)
    rng_sapg = np.random.default_rng(seed_sapg)
    rng_post = np.random.default_rng(seed_post)

    start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Simulate the regression problem.
    # --------------------------------------------------------
    sim = simulate_sparse_regression(
        n=n,
        p=p,
        support=SUPPORT_TRUE,
        values=BETA_TEMPLATE,
        rho=RHO,
        sigma=SIGMA_TRUE,
        rng=rng_sim,
        target_snr=TARGET_SNR,
        snr_mode="population",
        scale_columns=True,
    )

    # --------------------------------------------------------
    # 2. Estimate the observation scale.
    # --------------------------------------------------------
    raw_scaled_lasso_sigma = np.nan

    if regime["noise_method"] == "residual":
        noise_fit = residual_variance(
            sim.y,
            sim.X,
            intercept_df=0,
        )
        sigma_hat = float(noise_fit.sigma)
        sigma2_hat = float(noise_fit.sigma2)
        beta_reference = np.asarray(noise_fit.beta_ls, dtype=float)

    elif regime["noise_method"] == "scaled_lasso_postols":
        scaled_fit = scaled_lasso_variance(sim.y, sim.X)

        if not scaled_fit.converged:
            raise RuntimeError("Scaled Lasso did not converge.")

        raw_scaled_lasso_sigma = float(scaled_fit.sigma)
        beta_reference = np.asarray(
            scaled_fit.beta_nuisance,
            dtype=float,
        )

        postols = post_selection_ols_variance(
            sim.y,
            sim.X,
            beta_reference,
        )
        sigma_hat = float(postols["sigma"])
        sigma2_hat = float(postols["sigma2"])

    else:
        raise ValueError(f"Unknown noise method: {regime['noise_method']}")

    # --------------------------------------------------------
    # 3. Empirical-Bayes calibration of theta.
    # --------------------------------------------------------
    sapg_fit = sapg_laplace(
        sim.y,
        sim.X,
        sigma2_hat,
        rng=rng_sapg,
        n_iter=SAPG_N_ITER,
        beta_ref=beta_reference,
        beta0=beta_reference,
        intrinsic_dim=p,
        constraint="none",
        bound_factor=SAPG_BOUND_FACTOR,
        step_scale=SAPG_STEP_SCALE,
        step_offset=SAPG_STEP_OFFSET,
        step_power=SAPG_STEP_POWER,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=SAPG_MCMC_STEPS,
        mcmc_warmup=SAPG_MCMC_WARMUP,
        lambda_my=None,
        step_size_myula=None,
        c_delta=C_DELTA,
        store_state_path=False,
    )

    theta_hat = float(sapg_fit.theta_hat)
    lambda_my = float(sapg_fit.lambda_my)

    # --------------------------------------------------------
    # 4. Unsmoothed MAP and smoothed-MAP diagnostic.
    # --------------------------------------------------------
    map_fit = fista_map(
        sim.y,
        sim.X,
        sigma2_hat,
        theta_hat,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
    )

    if not map_fit.converged:
        raise RuntimeError("FISTA MAP did not converge.")

    beta_map = np.asarray(map_fit.beta, dtype=float)

    smoothed_fit = smoothed_map(
        sim.y,
        sim.X,
        sigma2_hat,
        theta_hat,
        lambda_my,
        beta0=beta_map,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
    )

    if not smoothed_fit.converged:
        raise RuntimeError("Smoothed MAP did not converge.")

    beta_map_lambda = np.asarray(smoothed_fit.beta, dtype=float)

    # --------------------------------------------------------
    # 5. Final posterior calculation.
    #
    # The first 4,000 retained draws are the short-run calculation
    # reported in the Monte Carlo refinement comparison.
    # --------------------------------------------------------
    post_fit = myula(
        sim.y,
        sim.X,
        sigma2_hat,
        theta_hat,
        rng=rng_post,
        n_samples=LONG_N_SAMPLES,
        burn_in=POST_BURN_IN,
        thin=POST_THIN,
        beta0=beta_map,
        lambda_my=lambda_my,
        step_size=None,
        c_delta=C_DELTA,
        store_potential=False,
    )

    samples_40k = np.asarray(post_fit.samples, dtype=float)
    samples_4k = samples_40k[:SHORT_N_SAMPLES]

    # --------------------------------------------------------
    # 6. Final and short-run decision grids.
    # --------------------------------------------------------
    grid_40k = evaluate_gate_grid(
        beta_map,
        samples_40k,
        lambda_my=lambda_my,
    )
    grid_4k = evaluate_gate_grid(
        beta_map,
        samples_4k,
        lambda_my=lambda_my,
    )

    true_mask = np.zeros(p, dtype=bool)
    true_mask[SUPPORT_TRUE] = True

    gate_rows = []

    for ik, k_value in enumerate(K_GRID):
        for ip, pi_star in enumerate(PI_GRID):
            metrics = support_metrics(
                grid_40k["support_masks"][ik, ip],
                true_mask,
            )

            _, _, diagnostic_row = grid_row(
                grid_40k,
                float(k_value),
                float(pi_star),
            )

            gate_rows.append(
                {
                    "regime": regime_name,
                    "replicate": int(replicate),
                    "n": n,
                    "p": p,
                    "k": float(k_value),
                    "pi_star": float(pi_star),
                    **metrics,
                    "threshold": float(grid_40k["thresholds"][ik]),
                    "min_gate_margin_mcse_eligible": float(
                        diagnostic_row["min_gate_margin_mcse_eligible"]
                    ),
                    "n_eligible_within_2_mcse": int(
                        diagnostic_row["n_eligible_within_2_mcse"]
                    ),
                }
            )

    ik_ref, ip_ref, diagnostic_40k = grid_row(
        grid_40k,
        K_REFERENCE,
        PI_REFERENCE,
    )
    _, _, diagnostic_4k = grid_row(
        grid_4k,
        K_REFERENCE,
        PI_REFERENCE,
    )

    selected_40k = grid_40k["support_masks"][ik_ref, ip_ref]
    selected_4k = grid_4k["support_masks"][ik_ref, ip_ref]

    reference_40k = support_metrics(selected_40k, true_mask)
    reference_4k = support_metrics(selected_4k, true_mask)

    # --------------------------------------------------------
    # 7. Diagnostics and estimation error.
    # --------------------------------------------------------
    chain_40k = chain_summary(
        samples_40k,
        max_lag=DIAGNOSTIC_MAX_LAG,
        ddof=1,
    )
    chain_4k = chain_summary(
        samples_4k,
        max_lag=DIAGNOSTIC_MAX_LAG,
        ddof=1,
    )

    sapg_diag = sapg_tail_summary(
        sapg_fit.eta_path,
        sapg_fit.theta_path,
        sapg_fit.score_path,
        sapg_fit.bound_hit_path,
        tail_start=sapg_fit.average_start,
    )

    map_diag = map_discrepancy(
        beta_map,
        beta_map_lambda,
        threshold=float(grid_40k["thresholds"][ik_ref]),
    )

    beta_error = beta_map - sim.beta_true
    relative_l2 = float(
        np.linalg.norm(beta_error) / np.linalg.norm(sim.beta_true)
    )
    support_rmse = float(
        np.sqrt(np.mean(beta_error[SUPPORT_TRUE] ** 2))
    )

    Sigma = ar1_covariance(p, RHO)
    prediction_risk = float(
        beta_error @ Sigma @ beta_error / sim.sigma2
    )

    support_symdiff = int(
        np.count_nonzero(selected_4k ^ selected_40k)
    )

    elapsed = time.perf_counter() - start

    summary = {
        "regime": regime_name,
        "replicate": int(replicate),
        "n": n,
        "p": p,
        "sigma_hat": sigma_hat,
        "raw_scaled_lasso_sigma": raw_scaled_lasso_sigma,
        "theta_hat": theta_hat,
        "lambda_my": lambda_my,
        "myula_delta": float(post_fit.step_size),
        "map_relative_l2": relative_l2,
        "support_rmse": support_rmse,
        "prediction_risk": prediction_risk,
        "map_stationarity": float(map_fit.stationarity),
        "map_shift_relative_l2": float(map_diag.relative_l2),
        "sapg_tail_score_abs_mean": float(sapg_diag.score_abs_mean),
        "sapg_bound_hit_fraction": float(
            sapg_diag.any_bound_hit_fraction
        ),
        "TPR_40k": reference_40k["TPR"],
        "FDP_40k": reference_40k["FDP"],
        "selected_size_40k": reference_40k["selected_size"],
        "exact_support_40k": reference_40k["exact_support"],
        "TPR_4k": reference_4k["TPR"],
        "FDP_4k": reference_4k["FDP"],
        "selected_size_4k": reference_4k["selected_size"],
        "exact_support_4k": reference_4k["exact_support"],
        "support_identical_4k_40k": int(support_symdiff == 0),
        "support_symdiff_4k_40k": support_symdiff,
        "coeff_ess_median_4k": float(np.median(chain_4k.ess)),
        "coeff_ess_median_40k": float(np.median(chain_40k.ess)),
        "min_gate_margin_mcse_4k": float(
            diagnostic_4k["min_gate_margin_mcse_eligible"]
        ),
        "min_gate_margin_mcse_40k": float(
            diagnostic_40k["min_gate_margin_mcse_eligible"]
        ),
        "n_eligible_within_2_mcse_4k": int(
            diagnostic_4k["n_eligible_within_2_mcse"]
        ),
        "n_eligible_within_2_mcse_40k": int(
            diagnostic_40k["n_eligible_within_2_mcse"]
        ),
        "wall_time_sec": float(elapsed),
    }

    signal_rows = []

    for signal_rank, j in enumerate(SUPPORT_TRUE, start=1):
        signal_rows.append(
            {
                "regime": regime_name,
                "replicate": int(replicate),
                "signal_rank": int(signal_rank),
                "predictor": int(j),
                "beta_true": float(sim.beta_true[j]),
                "abs_beta_true": float(abs(sim.beta_true[j])),
                "map_eligible": int(
                    abs(beta_map[j])
                    >= grid_40k["thresholds"][ik_ref]
                ),
                "selected": int(selected_40k[j]),
            }
        )

    return summary, gate_rows, signal_rows


# ============================================================
# Printed summaries
# ============================================================

def mean_se(values: pd.Series) -> str:
    """Format an ensemble mean with its Monte Carlo standard error."""
    x = values.to_numpy(dtype=float)
    return f"{np.mean(x):.4f} ({np.std(x, ddof=1) / np.sqrt(x.size):.4f})"


def print_reference_table(summary: pd.DataFrame) -> None:
    """Print the quantities corresponding to the main Study 1 table."""
    print("\nReference decision rule: k=2.5, pi_star=0.75")
    print(f"Mean (standard error) across {R_PER_REGIME} datasets\n")

    for regime in REGIMES:
        name = regime["name"]
        g = summary.loc[summary["regime"] == name]

        print(regime["label"])
        print(f"  sigma_hat / sigma : {mean_se(g['sigma_hat'] / SIGMA_TRUE)}")
        print(f"  theta_hat         : {mean_se(g['theta_hat'])}")
        print(f"  MAP relative L2   : {mean_se(g['map_relative_l2'])}")
        print(f"  support RMSE      : {mean_se(g['support_rmse'])}")
        print(f"  prediction risk   : {mean_se(g['prediction_risk'])}")
        print(f"  TPR               : {mean_se(g['TPR_40k'])}")
        print(f"  FDR               : {mean_se(g['FDP_40k'])}")
        print(f"  selected size     : {mean_se(g['selected_size_40k'])}")
        print(f"  exact support     : {mean_se(g['exact_support_40k'])}")
        print()


def print_gate_grid(gates: pd.DataFrame) -> None:
    """Print the final 40k decision grid."""
    print("\nFinal 40k decision grid: ensemble means")

    for regime in REGIMES:
        name = regime["name"]
        print(f"\n{regime['label']}")

        g = (
            gates.loc[gates["regime"] == name]
            .groupby(["k", "pi_star"], as_index=False)
            .agg(
                TPR=("TPR", "mean"),
                FDR=("FDP", "mean"),
                size=("selected_size", "mean"),
                exact=("exact_support", "mean"),
            )
        )

        print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_refinement_summary(summary: pd.DataFrame) -> None:
    """Print the reported 4k-versus-40k Monte Carlo comparison."""
    print("\n4k versus 40k posterior calculation")

    for regime in REGIMES:
        name = regime["name"]
        g = summary.loc[summary["regime"] == name]

        finite4 = g["min_gate_margin_mcse_4k"].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        finite40 = g["min_gate_margin_mcse_40k"].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        print(f"\n{regime['label']}")
        print(
            "  median coordinate ESS : "
            f"{g['coeff_ess_median_4k'].median():.1f} -> "
            f"{g['coeff_ess_median_40k'].median():.1f}"
        )
        print(
            "  identical selected sets: "
            f"{g['support_identical_4k_40k'].mean():.3f}"
        )
        print(
            "  mean symmetric difference: "
            f"{g['support_symdiff_4k_40k'].mean():.3f}"
        )
        print(
            "  median smallest eligible D_j: "
            f"{finite4.median():.2f} -> {finite40.median():.2f}"
        )
        print(
            "  fraction with eligible D_j < 2 after 40k: "
            f"{np.mean(g['n_eligible_within_2_mcse_40k'] > 0):.3f}"
        )


# ============================================================
# Figures
# ============================================================

def set_plot_defaults() -> None:
    """Use the plotting settings used for the paper figures."""
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save one figure as PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_figures(
    summary: pd.DataFrame,
    signals: pd.DataFrame,
) -> None:
    """Create the four Study 1 figures retained in the paper."""
    set_plot_defaults()

    # --------------------------------------------------------
    # Main paper: high-dimensional signal profile.
    # --------------------------------------------------------
    profile = (
        signals.loc[signals["regime"] == "high_dim"]
        .groupby(["signal_rank", "predictor"], as_index=False)
        .agg(
            abs_beta_true=("abs_beta_true", "mean"),
            map_eligibility_frequency=("map_eligible", "mean"),
            final_selection_frequency=("selected", "mean"),
        )
        .sort_values("abs_beta_true")
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        profile["abs_beta_true"],
        profile["map_eligibility_frequency"],
        marker="o",
        linewidth=1.6,
        label="Passes MAP magnitude gate",
    )
    ax.plot(
        profile["abs_beta_true"],
        profile["final_selection_frequency"],
        marker="s",
        linewidth=1.6,
        label="Final selection",
    )
    ax.set_xlabel(r"True signal magnitude $|\beta_j^\star|$")
    ax.set_ylabel("Frequency across 100 datasets")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(r"Signal-wise selection profile, $n=200,\ p=500$")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_study1_signal_profile_highdim")

    # --------------------------------------------------------
    # Supplement: noise-scale calibration.
    # --------------------------------------------------------
    low = summary.loc[
        summary["regime"] == "low_dim",
        "sigma_hat",
    ].to_numpy(dtype=float)

    high_raw = summary.loc[
        summary["regime"] == "high_dim",
        "raw_scaled_lasso_sigma",
    ].to_numpy(dtype=float)

    high_post = summary.loc[
        summary["regime"] == "high_dim",
        "sigma_hat",
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.boxplot(
        [low, high_raw, high_post],
        tick_labels=[
            "Residual estimator\n$n>p$",
            "Raw scaled Lasso\n$p>n$",
            "Post-selection OLS\n$p>n$",
        ],
        showmeans=True,
        widths=0.55,
    )
    ax.axhline(
        SIGMA_TRUE,
        linestyle="--",
        linewidth=1.2,
        label=rf"True $\sigma={SIGMA_TRUE:g}$",
    )
    ax.set_ylabel(r"Estimated noise scale $\widehat{\sigma}$")
    ax.set_title("Noise-scale calibration")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig_study1_noise_calibration_revised")

    # --------------------------------------------------------
    # Supplement: change in the selected set, 4k versus 40k.
    # --------------------------------------------------------
    differences = sorted(
        summary["support_symdiff_4k_40k"].unique()
    )
    low_counts = np.array(
        [
            np.count_nonzero(
                (summary["regime"] == "low_dim")
                & (summary["support_symdiff_4k_40k"] == value)
            )
            for value in differences
        ],
        dtype=float,
    )
    high_counts = np.array(
        [
            np.count_nonzero(
                (summary["regime"] == "high_dim")
                & (summary["support_symdiff_4k_40k"] == value)
            )
            for value in differences
        ],
        dtype=float,
    )

    x = np.arange(len(differences))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        x - width / 2,
        low_counts,
        width,
        label=r"$n=500,\ p=200$",
    )
    ax.bar(
        x + width / 2,
        high_counts,
        width,
        label=r"$n=200,\ p=500$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in differences])
    ax.set_xlabel("Number of coordinates selected in exactly one run")
    ax.set_ylabel("Number of datasets")
    ax.set_title("Change in selected set: 4k versus 40k retained draws")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    save_figure(fig, "fig_study1_refinement_selected_set_change")

    # --------------------------------------------------------
    # Main paper: decision-level Monte Carlo precision.
    # --------------------------------------------------------
    groups = []
    labels = []

    for regime_name, regime_short in (
        ("low_dim", r"$n>p$"),
        ("high_dim", r"$p>n$"),
    ):
        for draws, column in (
            ("4k", "min_gate_margin_mcse_4k"),
            ("40k", "min_gate_margin_mcse_40k"),
        ):
            values = summary.loc[
                summary["regime"] == regime_name,
                column,
            ].to_numpy(dtype=float)

            finite = values[
                np.isfinite(values) & (values > 0.0)
            ]
            groups.append(finite)
            labels.append(regime_short + "\n" + draws)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.boxplot(
        groups,
        tick_labels=labels,
        showmeans=True,
    )
    ax.axhline(
        2.0,
        linestyle="--",
        linewidth=1.1,
        label="Two MCSE from probability boundary",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"Smallest eligible $D_j$ (MCSE units)")
    ax.set_title("Monte Carlo precision near the probability gate")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_study1_refinement_probability_precision")


# ============================================================
# Main run
# ============================================================

def main() -> None:
    print("Study 1: end-to-end sparse regression")
    print(f"Replicates per regime: {R_PER_REGIME}")
    print(f"Posterior draws: {LONG_N_SAMPLES:,}")
    print(
        "The 4k comparison uses the first "
        f"{SHORT_N_SAMPLES:,} retained draws of each final chain."
    )

    summaries = []
    gate_rows = []
    signal_rows = []

    study_start = time.perf_counter()

    for regime in REGIMES:
        print(
            f"\n{regime['name']}: "
            f"n={regime['n']}, p={regime['p']}"
        )

        for replicate in range(R_PER_REGIME):
            summary, gates, signals = run_replicate(
                regime,
                replicate,
            )

            summaries.append(summary)
            gate_rows.extend(gates)
            signal_rows.extend(signals)

            if (
                replicate == 0
                or (replicate + 1) % 10 == 0
                or replicate + 1 == R_PER_REGIME
            ):
                print(
                    f"  completed {replicate + 1:3d}/"
                    f"{R_PER_REGIME}"
                )

    summary = pd.DataFrame(summaries)
    gates = pd.DataFrame(gate_rows)
    signals = pd.DataFrame(signal_rows)

    print_reference_table(summary)
    print_gate_grid(gates)
    print_refinement_summary(summary)

    make_figures(summary, signals)

    elapsed = time.perf_counter() - study_start

    print(f"\nFigures written to: {FIGURE_DIR}")
    print(f"Total wall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
