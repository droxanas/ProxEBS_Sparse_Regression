"""
Study 3: diabetes data and external benchmarking.

This script reproduces the final diabetes analysis in the paper.
It uses the public diabetes data file from Zhou et al., runs the
extended SAPG calibration, computes the nonsmoothed and smoothed MAPs,
runs the final plain-MYULA calculation, evaluates the full decision
grid, and rebuilds the external interval comparison.

The extended SAPG estimate is assigned once to `theta_hat` and the same
value is used for every downstream calculation.

The script writes the two figures used in the paper and supplement.
Numerical summaries are printed to the terminal. No intermediate
result files are written.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diagnostics
import fista
import myula
import sapg
import selection


# ============================================================
# Files and labels
# ============================================================

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")

DIABETES_FILE = DATA_DIR / "diabetes-442-10.csv"
ZHOU_INTERVAL_FILE = DATA_DIR / "zhou_figure2_intervals.csv"
ZHOU_METADATA_FILE = DATA_DIR / "zhou_figure2_metadata.csv"

FEATURE_CODES = [f"x{j}" for j in range(1, 11)]
FEATURE_NAMES = [
    "Age",
    "Sex",
    "BMI",
    "BP",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
]

EXPECTED_DATA_SHA256 = (
    "de152c0f68944f9aa4e5ca5051d4057459dfec3d31e241b6c8aa2f43b3710552"
)


# ============================================================
# Frozen numerical settings
# ============================================================

SAPG_SEED = 5126090701
SAPG_N_ITER = 60_000
SAPG_AVERAGE_START = 20_000

# The first 15,000 iterations are also used to reproduce the
# preliminary calibration check reported in the supplement.
SAPG_SHORT_N_ITER = 15_000
SAPG_SHORT_AVERAGE_START = 4_000

POSTERIOR_SEED = 5126090702
POSTERIOR_INITIAL_BURN = 4_000
POSTERIOR_STORED = 96_000
POSTERIOR_EXTRA_DISCARD = 16_000
POSTERIOR_LONG_KEEP = 80_000
POSTERIOR_SHORT_KEEP = 4_000

C_DELTA = 0.9

MAP_TOL = 1e-10
MAP_MAX_ITER = 100_000

K_GRID = (2.0, 2.5, 3.0)
PI_GRID = (0.50, 0.75, 0.90)
K_REFERENCE = 2.5
PI_REFERENCE = 0.75

OUR_METHOD = "EB-Laplace + MYULA"


# ============================================================
# Small helpers
# ============================================================


def safe_ess(x: np.ndarray) -> float:
    """Return ESS, including a simple guard for constant sequences."""
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size < 2:
        return np.nan

    if np.all(x == x[0]):
        return float(x.size)

    try:
        return float(diagnostics.effective_sample_size(x))
    except Exception:
        return np.nan



def finite_min(x: np.ndarray) -> float:
    """Return the smallest finite entry, or infinity if none exists."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)

    if np.any(good):
        return float(np.min(x[good]))

    return np.inf



def activation_mcse(
    samples: np.ndarray,
    threshold: float,
    pi_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Activation probabilities, ESS, MCSE and decision distance."""
    indicators = (np.abs(samples) >= threshold).astype(float)
    probabilities = np.mean(indicators, axis=0)

    p = samples.shape[1]
    ess = np.empty(p, dtype=float)
    mcse = np.empty(p, dtype=float)
    distance = np.empty(p, dtype=float)

    for j in range(p):
        z = indicators[:, j]

        if np.all(z == z[0]):
            ess[j] = float(z.size)
            mcse[j] = 0.0
            distance[j] = (
                0.0 if np.isclose(probabilities[j], pi_star) else np.inf
            )
            continue

        ess_j = safe_ess(z)
        ess[j] = ess_j

        if np.isfinite(ess_j) and ess_j > 0.0:
            mcse_j = np.sqrt(
                probabilities[j]
                * (1.0 - probabilities[j])
                / ess_j
            )
        else:
            mcse_j = np.nan

        mcse[j] = mcse_j

        if np.isfinite(mcse_j) and mcse_j > 0.0:
            distance[j] = abs(probabilities[j] - pi_star) / mcse_j
        elif np.isfinite(mcse_j):
            distance[j] = np.inf
        else:
            distance[j] = np.nan

    return probabilities, ess, mcse, distance



def posterior_summary(
    samples: np.ndarray,
    eigvecs_xtx: np.ndarray,
    potential: np.ndarray | None = None,
) -> dict:
    """Compute coefficient and mixing summaries for one posterior sample."""
    mean = np.mean(samples, axis=0)
    sd = np.std(samples, axis=0, ddof=1)
    q025 = np.quantile(samples, 0.025, axis=0)
    q975 = np.quantile(samples, 0.975, axis=0)
    p_positive = np.mean(samples > 0.0, axis=0)
    p_negative = np.mean(samples < 0.0, axis=0)

    coordinate_ess = np.array(
        [safe_ess(samples[:, j]) for j in range(samples.shape[1])],
        dtype=float,
    )

    projected = samples @ eigvecs_xtx
    eigendirection_ess = np.array(
        [safe_ess(projected[:, j]) for j in range(projected.shape[1])],
        dtype=float,
    )

    half = samples.shape[0] // 2
    mean_first = np.mean(samples[:half], axis=0)
    mean_second = np.mean(samples[half:], axis=0)
    half_split_gap = float(
        np.linalg.norm(mean_second - mean_first)
        / max(np.linalg.norm(mean), 1e-15)
    )

    if potential is None or np.asarray(potential).size == 0:
        potential_ess = np.nan
    else:
        potential_ess = safe_ess(np.asarray(potential, dtype=float))

    return {
        "mean": mean,
        "sd": sd,
        "q025": q025,
        "q975": q975,
        "p_positive": p_positive,
        "p_negative": p_negative,
        "coordinate_ess": coordinate_ess,
        "eigendirection_ess": eigendirection_ess,
        "potential_ess": potential_ess,
        "half_split_gap": half_split_gap,
    }


# ============================================================
# Data and likelihood scale
# ============================================================


def load_diabetes() -> dict:
    """Load the exact public diabetes file and reproduce its working scale."""
    if not DIABETES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DIABETES_FILE}. Run this script from the repository root."
        )

    raw = pd.read_csv(DIABETES_FILE)
    expected_columns = FEATURE_CODES + ["y"]

    if list(raw.columns) != expected_columns:
        raise RuntimeError(
            f"Unexpected columns. Expected {expected_columns}, got {list(raw.columns)}."
        )

    if raw.shape != (442, 11):
        raise RuntimeError(f"Unexpected diabetes shape: {raw.shape}")

    if raw.isna().to_numpy().any():
        raise RuntimeError("Missing values found in diabetes data.")

    sha256 = hashlib.sha256(DIABETES_FILE.read_bytes()).hexdigest()
    if sha256 != EXPECTED_DATA_SHA256:
        raise RuntimeError(
            "The diabetes file does not match the public file used in the analysis."
        )

    X = raw[FEATURE_CODES].to_numpy(dtype=float)
    y_raw = raw["y"].to_numpy(dtype=float)

    # This matches the Zhou et al. demonstration: divide y by
    # its sample standard deviation and use X as supplied.
    y_scale = float(np.std(y_raw, ddof=1))
    y = y_raw / y_scale

    n, p = X.shape
    XtX = X.T @ X
    eigvals_xtx, eigvecs_xtx = np.linalg.eigh(XtX)

    beta_ols, _, rank_ols, _ = np.linalg.lstsq(X, y, rcond=None)
    if int(rank_ols) != p:
        raise RuntimeError("The diabetes design is not full rank.")

    residual = y - X @ beta_ols
    rss = float(residual @ residual)

    # The response and predictors are centred. The residual variance
    # still accounts for the fitted intercept through one extra df.
    df_residual = int(n - p - 1)
    sigma2_hat = float(rss / df_residual)
    sigma_hat = float(np.sqrt(sigma2_hat))

    Lf = float(eigvals_xtx[-1] / sigma2_hat)
    lambda_my = float(1.0 / Lf)
    delta = float(C_DELTA / (Lf + 1.0 / lambda_my))
    theta_init = float(p / np.sum(np.abs(beta_ols)))

    corr = np.corrcoef(X, rowvar=False)
    corr_abs = np.abs(corr.copy())
    np.fill_diagonal(corr_abs, -np.inf)
    i_corr, j_corr = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "y_scale": y_scale,
        "n": n,
        "p": p,
        "XtX": XtX,
        "eigvals_xtx": eigvals_xtx,
        "eigvecs_xtx": eigvecs_xtx,
        "beta_ols": beta_ols,
        "rss": rss,
        "df_residual": df_residual,
        "sigma2_hat": sigma2_hat,
        "sigma_hat": sigma_hat,
        "Lf": Lf,
        "lambda_my": lambda_my,
        "delta": delta,
        "theta_init": theta_init,
        "rank_X": int(np.linalg.matrix_rank(X)),
        "condition_xtx": float(eigvals_xtx[-1] / eigvals_xtx[0]),
        "max_abs_corr": float(abs(corr[i_corr, j_corr])),
        "max_corr_pair": (FEATURE_NAMES[i_corr], FEATURE_NAMES[j_corr]),
    }


# ============================================================
# Extended SAPG calibration
# ============================================================


def run_sapg(data: dict) -> dict:
    """Run the final 60,000-iteration SAPG calculation."""
    rng = np.random.default_rng(SAPG_SEED)
    start = time.perf_counter()

    result = sapg.sapg_laplace(
        data["y"],
        data["X"],
        data["sigma2_hat"],
        rng=rng,
        n_iter=SAPG_N_ITER,
        theta_init=data["theta_init"],
        beta_ref=data["beta_ols"].copy(),
        beta0=data["beta_ols"].copy(),
        intrinsic_dim=data["p"],
        constraint="none",
        theta_bounds=None,
        bound_factor=10.0,
        step_scale=1.0,
        step_offset=200.0,
        step_power=1.0,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=1,
        mcmc_warmup=0,
        lambda_my=data["lambda_my"],
        step_size_myula=None,
        c_delta=C_DELTA,
        store_state_path=False,
    )

    runtime = float(time.perf_counter() - start)

    eta_path = np.asarray(result.eta_path, dtype=float).reshape(-1)
    theta_path = np.asarray(result.theta_path, dtype=float).reshape(-1)
    score_path = np.asarray(result.score_path, dtype=float).reshape(-1)
    bound_path = np.asarray(result.bound_hit_path).reshape(-1)

    # The first 15,000 iterations reproduce the earlier calculation.
    # The estimate below uses the same log-scale averaging rule as SAPG.
    theta_short = float(
        np.exp(
            np.mean(
                eta_path[
                    SAPG_SHORT_AVERAGE_START : SAPG_SHORT_N_ITER + 1
                ]
            )
        )
    )

    theta_hat = float(result.theta_hat)
    theta_final = float(result.theta_final)

    averaging_region = theta_path[SAPG_AVERAGE_START:]
    middle = averaging_region.size // 2
    first_half = float(np.mean(averaging_region[:middle]))
    second_half = float(np.mean(averaging_region[middle:]))
    split_change = float((second_half - first_half) / theta_hat)

    tail = theta_path[-5000:]

    return {
        "result": result,
        "theta_hat": theta_hat,
        "theta_short": theta_short,
        "theta_final": theta_final,
        "first_half": first_half,
        "second_half": second_half,
        "split_change": split_change,
        "tail_mean": float(np.mean(tail)),
        "tail_sd": float(np.std(tail, ddof=1)),
        "bound_hits": int(np.sum(bound_path.astype(int))),
        "runtime": runtime,
        "score_path": score_path,
    }


# ============================================================
# MAP calculations
# ============================================================


def run_maps(data: dict, theta_hat: float) -> dict:
    """Compute the nonsmoothed MAP and the Moreau-smoothed diagnostic MAP."""
    start = time.perf_counter()
    exact = fista.fista_map(
        data["y"],
        data["X"],
        data["sigma2_hat"],
        theta_hat,
        beta0=data["beta_ols"].copy(),
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    runtime_exact = float(time.perf_counter() - start)

    if not exact.converged:
        raise RuntimeError("FISTA did not converge.")

    beta_map = np.asarray(exact.beta, dtype=float)

    start = time.perf_counter()
    smooth = fista.smoothed_map(
        data["y"],
        data["X"],
        data["sigma2_hat"],
        theta_hat,
        data["lambda_my"],
        beta0=beta_map.copy(),
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    runtime_smooth = float(time.perf_counter() - start)

    if not smooth.converged:
        raise RuntimeError("Smoothed MAP calculation did not converge.")

    beta_smooth = np.asarray(smooth.beta, dtype=float)
    relative_gap = float(
        np.linalg.norm(beta_smooth - beta_map)
        / max(np.linalg.norm(beta_map), 1e-15)
    )

    support = np.abs(beta_map) > MAP_TOL

    return {
        "exact": exact,
        "smooth": smooth,
        "beta_map": beta_map,
        "beta_smooth": beta_smooth,
        "support": support,
        "support_size": int(np.sum(support)),
        "relative_gap": relative_gap,
        "runtime_exact": runtime_exact,
        "runtime_smooth": runtime_smooth,
    }


# ============================================================
# Final posterior calculation
# ============================================================


def run_posterior(
    data: dict,
    theta_hat: float,
    beta_map: np.ndarray,
) -> dict:
    """Run one 100,000-transition MYULA path and form short and long samples."""
    rng = np.random.default_rng(POSTERIOR_SEED)
    start = time.perf_counter()

    result = myula.myula(
        data["y"],
        data["X"],
        data["sigma2_hat"],
        theta_hat,
        rng=rng,
        n_samples=POSTERIOR_STORED,
        burn_in=POSTERIOR_INITIAL_BURN,
        thin=1,
        beta0=beta_map.copy(),
        lambda_my=data["lambda_my"],
        step_size=None,
        c_delta=C_DELTA,
        store_potential=True,
    )

    runtime = float(time.perf_counter() - start)
    stored = np.asarray(result.samples, dtype=float)

    expected_shape = (POSTERIOR_STORED, data["p"])
    if stored.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected MYULA sample shape: {stored.shape}; expected {expected_shape}."
        )

    short_samples = stored[:POSTERIOR_SHORT_KEEP]
    long_samples = stored[POSTERIOR_EXTRA_DISCARD:]

    if long_samples.shape[0] != POSTERIOR_LONG_KEEP:
        raise RuntimeError("The long posterior sample has the wrong length.")

    potential_stored = np.asarray(result.potential_path, dtype=float).reshape(-1)
    short_potential = potential_stored[:POSTERIOR_SHORT_KEEP]
    long_potential = potential_stored[POSTERIOR_EXTRA_DISCARD:]

    short = posterior_summary(
        short_samples,
        data["eigvecs_xtx"],
        potential=short_potential,
    )
    long = posterior_summary(
        long_samples,
        data["eigvecs_xtx"],
        potential=long_potential,
    )

    postmean_gap = float(
        np.linalg.norm(long["mean"] - short["mean"])
        / max(np.linalg.norm(long["mean"]), 1e-15)
    )

    max_sign_change = float(
        np.max(
            np.maximum(
                np.abs(long["p_positive"] - short["p_positive"]),
                np.abs(long["p_negative"] - short["p_negative"]),
            )
        )
    )

    return {
        "result": result,
        "stored": stored,
        "short_samples": short_samples,
        "long_samples": long_samples,
        "short": short,
        "long": long,
        "postmean_gap": postmean_gap,
        "max_sign_change": max_sign_change,
        "runtime": runtime,
        "delta": float(result.step_size),
    }


# ============================================================
# Decision grid
# ============================================================


def evaluate_decision_grid(
    samples: np.ndarray,
    beta_map: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate all nine decision settings and return the reference detail."""
    grid_rows = []
    detail_rows = []
    reference = None

    for k_gate in K_GRID:
        threshold, _ = selection.posterior_scale_threshold(samples, k_gate)
        threshold = float(threshold)

        activation = np.asarray(
            selection.activation_probabilities(samples, threshold),
            dtype=float,
        )
        map_eligible = np.abs(beta_map) >= threshold

        for pi_star in PI_GRID:
            selected = map_eligible & (activation >= pi_star)
            _, indicator_ess, mcse, distance = activation_mcse(
                samples,
                threshold,
                pi_star,
            )

            grid_rows.append(
                {
                    "k": float(k_gate),
                    "pi_star": float(pi_star),
                    "threshold": threshold,
                    "map_eligible_size": int(np.sum(map_eligible)),
                    "selected_size": int(np.sum(selected)),
                    "map_eligible_support": ", ".join(
                        np.asarray(FEATURE_NAMES)[map_eligible]
                    ),
                    "selected_support": ", ".join(
                        np.asarray(FEATURE_NAMES)[selected]
                    ),
                    "min_D_map_eligible": finite_min(distance[map_eligible]),
                }
            )

            for j, name in enumerate(FEATURE_NAMES):
                detail_rows.append(
                    {
                        "k": float(k_gate),
                        "pi_star": float(pi_star),
                        "feature_index": j + 1,
                        "feature_code": FEATURE_CODES[j],
                        "feature_name": name,
                        "beta_map": float(beta_map[j]),
                        "threshold": threshold,
                        "activation_probability": float(activation[j]),
                        "map_eligible": int(map_eligible[j]),
                        "selected": int(selected[j]),
                        "indicator_ess": float(indicator_ess[j]),
                        "activation_mcse": float(mcse[j]),
                        "decision_distance": float(distance[j]),
                    }
                )

            if np.isclose(k_gate, K_REFERENCE) and np.isclose(
                pi_star, PI_REFERENCE
            ):
                reference = {
                    "threshold": threshold,
                    "activation": activation.copy(),
                    "map_eligible": map_eligible.copy(),
                    "selected": selected.copy(),
                    "indicator_ess": indicator_ess.copy(),
                    "mcse": mcse.copy(),
                    "distance": distance.copy(),
                }

    if reference is None:
        raise RuntimeError("Reference decision setting was not found.")

    return (
        pd.DataFrame(grid_rows),
        pd.DataFrame(detail_rows),
        reference,
    )


# ============================================================
# External benchmark inputs and figures
# ============================================================


def load_zhou_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and audit the interval endpoints transcribed from Zhou et al."""
    for path in (ZHOU_INTERVAL_FILE, ZHOU_METADATA_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark input: {path}")

    intervals = pd.read_csv(ZHOU_INTERVAL_FILE)
    metadata = pd.read_csv(ZHOU_METADATA_FILE)

    expected_methods = [
        "ProxMCMC",
        "bls-RJ-T",
        "bls-RJ-F",
        "Horseshoe",
        "SelInf",
    ]

    if intervals.shape[0] != 50:
        raise RuntimeError("Unexpected number of Zhou benchmark rows.")

    if set(intervals["feature_name"]) != set(FEATURE_NAMES):
        raise RuntimeError("Unexpected features in Zhou benchmark input.")

    if set(intervals["method"]) != set(expected_methods):
        raise RuntimeError("Unexpected methods in Zhou benchmark input.")

    duplicates = intervals.duplicated(subset=["feature_index", "method"])
    if duplicates.any():
        raise RuntimeError("Duplicate feature/method rows in Zhou benchmark input.")

    selinf_selected = intervals.loc[
        (intervals["method"] == "SelInf")
        & (intervals["selected_model"] == 1),
        "feature_code",
    ].tolist()

    if selinf_selected != ["x2", "x3", "x4", "x7", "x9"]:
        raise RuntimeError("Unexpected selective-inference support in benchmark input.")

    available = intervals[
        intervals["lower"].notna() & intervals["upper"].notna()
    ]
    if not np.all(available["lower"].to_numpy() <= available["upper"].to_numpy()):
        raise RuntimeError("Invalid interval endpoints in Zhou benchmark input.")

    return intervals, metadata



def build_interval_table(
    zhou: pd.DataFrame,
    posterior: dict,
    reference: dict,
) -> pd.DataFrame:
    """Append the present 95% credible intervals to the benchmark table."""
    our_intervals = pd.DataFrame(
        {
            "feature_index": np.arange(1, 11, dtype=int),
            "feature_code": FEATURE_CODES,
            "feature_name": FEATURE_NAMES,
            "method": OUR_METHOD,
            "estimate": posterior["mean"],
            "lower": posterior["q025"],
            "upper": posterior["q975"],
            "interval_level": 0.95,
            "interval_type": "credible",
            "selected_model": reference["selected"].astype(int),
            "source_file": "study3_diabetes.py",
            "source_cell": np.nan,
            "source_status": "reproduced_by_public_script",
        }
    )

    return (
        pd.concat([zhou, our_intervals], ignore_index=True)
        .sort_values(["feature_index", "method"])
        .reset_index(drop=True)
    )



def plot_benchmark_intervals(all_intervals: pd.DataFrame) -> None:
    """Rebuild the publication-style diabetes interval comparison."""
    display_labels = {
        "ProxMCMC": "ProxMCMC",
        "bls-RJ-T": "Bayesian lasso + RJ",
        "bls-RJ-F": "Bayesian lasso",
        "Horseshoe": "Horseshoe",
        "SelInf": "Selective inference",
        OUR_METHOD: "Present method",
    }

    methods = [
        "ProxMCMC",
        "bls-RJ-T",
        "bls-RJ-F",
        "Horseshoe",
        "SelInf",
        OUR_METHOD,
    ]

    markers = {
        "ProxMCMC": "o",
        "bls-RJ-T": "s",
        "bls-RJ-F": "D",
        "Horseshoe": "^",
        "SelInf": "x",
        OUR_METHOD: "P",
    }

    offsets = np.linspace(-0.30, 0.30, len(methods))
    x_base = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(12.5, 6.4))

    for offset, method in zip(offsets, methods):
        frame = (
            all_intervals.loc[all_intervals["method"] == method]
            .sort_values("feature_index")
            .copy()
        )
        frame = frame.loc[frame["lower"].notna() & frame["upper"].notna()]

        x = frame["feature_index"].to_numpy(dtype=float) + offset
        lower = frame["lower"].to_numpy(dtype=float)
        upper = frame["upper"].to_numpy(dtype=float)

        # Midpoints are used only to align the intervals visually.
        centre = 0.5 * (lower + upper)
        yerr = np.vstack([centre - lower, upper - centre])

        is_present = method == OUR_METHOD

        ax.errorbar(
            x,
            centre,
            yerr=yerr,
            fmt=markers[method],
            markersize=7.0 if is_present else 5.5,
            elinewidth=1.7 if is_present else 1.05,
            markeredgewidth=1.4 if is_present else 1.0,
            capsize=3.0,
            capthick=1.4 if is_present else 1.0,
            linestyle="none",
            label=display_labels[method],
        )

    ax.axhline(0.0, linewidth=1.0, color="0.25")
    ax.set_xticks(x_base)
    ax.set_xticklabels(FEATURE_NAMES)
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Coefficient")
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / "diabetes_benchmarking.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)



def plot_eigendirection_ess(
    eigvals_xtx: np.ndarray,
    sigma2_hat: float,
    short_ess: np.ndarray,
    long_ess: np.ndarray,
) -> None:
    """Plot ESS against likelihood curvature for short and long chains."""
    curvature = eigvals_xtx / sigma2_hat

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        curvature,
        short_ess,
        marker="o",
        linestyle="--",
        linewidth=1.25,
        markersize=5.5,
        label="Initial: 4,000 retained draws",
    )
    ax.plot(
        curvature,
        long_ess,
        marker="s",
        linestyle="-",
        linewidth=1.8,
        markersize=5.8,
        label="Extended: 80,000 retained draws",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"Likelihood curvature $\lambda_i(X^\top X)/\widehat{\sigma}^{\,2}$"
    )
    ax.set_ylabel("Eigendirection ESS")
    ax.grid(which="major", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.margins(x=0.04, y=0.08)

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / "diabetes_eigendirection_ess_vs_curvature.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Printed summaries
# ============================================================


def print_data_and_calibration(data: dict, sapg_fit: dict) -> None:
    print("\nData and likelihood geometry")
    print("----------------------------")
    print(f"n, p                         : {data['n']}, {data['p']}")
    print(f"rank(X)                      : {data['rank_X']}")
    print(f"lambda_min(X'X)              : {data['eigvals_xtx'][0]:.8f}")
    print(f"lambda_max(X'X)              : {data['eigvals_xtx'][-1]:.8f}")
    print(f"condition number X'X         : {data['condition_xtx']:.2f}")
    print(f"maximum absolute correlation : {data['max_abs_corr']:.6f}")
    print(f"strongest pair               : {data['max_corr_pair']}")
    print(f"RSS                           : {data['rss']:.6f}")
    print(f"residual df                   : {data['df_residual']}")
    print(f"sigma_hat                     : {data['sigma_hat']:.6f}")
    print(f"sigma2_hat                    : {data['sigma2_hat']:.6f}")
    print(f"L_f                           : {data['Lf']:.6f}")
    print(f"lambda_MY                     : {data['lambda_my']:.6f}")
    print(f"delta                         : {data['delta']:.6f}")
    print(f"theta_0                       : {data['theta_init']:.6f}")

    print("\nSAPG calibration")
    print("----------------")
    print(f"15k estimate                  : {sapg_fit['theta_short']:.9f}")
    print(f"60k estimate                  : {sapg_fit['theta_hat']:.9f}")
    print(f"final iterate                 : {sapg_fit['theta_final']:.9f}")
    print(f"first-half mean               : {sapg_fit['first_half']:.9f}")
    print(f"second-half mean              : {sapg_fit['second_half']:.9f}")
    print(f"relative split difference     : {sapg_fit['split_change']:+.3%}")
    print(f"last-5000 mean                : {sapg_fit['tail_mean']:.9f}")
    print(f"last-5000 SD                  : {sapg_fit['tail_sd']:.9f}")
    print(f"bound hits                    : {sapg_fit['bound_hits']}")
    print(f"runtime                       : {sapg_fit['runtime']:.2f} s")



def print_map_summary(data: dict, map_fit: dict) -> None:
    table = pd.DataFrame(
        {
            "Variable": FEATURE_NAMES,
            "OLS": data["beta_ols"],
            "MAP": map_fit["beta_map"],
            "Smoothed MAP": map_fit["beta_smooth"],
            "Nonzero": map_fit["support"].astype(int),
        }
    )

    print("\nMAP calculations")
    print("----------------")
    print(f"FISTA iterations              : {map_fit['exact'].n_iter}")
    print(f"MAP support size              : {map_fit['support_size']}")
    print(f"smoothed/exact relative gap   : {map_fit['relative_gap']:.6f}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))



def print_posterior_summary(posterior: dict) -> None:
    short = posterior["short"]
    long = posterior["long"]

    rows = pd.DataFrame(
        {
            "Diagnostic": [
                "Retained draws",
                "Median coordinate ESS",
                "Minimum coordinate ESS",
                "Weakest eigendirection ESS",
                "Median eigendirection ESS",
                "Strongest eigendirection ESS",
                "Potential ESS",
                "Median posterior SD",
                "Posterior-mean half-split gap",
            ],
            "Initial": [
                POSTERIOR_SHORT_KEEP,
                np.nanmedian(short["coordinate_ess"]),
                np.nanmin(short["coordinate_ess"]),
                short["eigendirection_ess"][0],
                np.nanmedian(short["eigendirection_ess"]),
                short["eigendirection_ess"][-1],
                short["potential_ess"],
                np.median(short["sd"]),
                short["half_split_gap"],
            ],
            "Extended": [
                POSTERIOR_LONG_KEEP,
                np.nanmedian(long["coordinate_ess"]),
                np.nanmin(long["coordinate_ess"]),
                long["eigendirection_ess"][0],
                np.nanmedian(long["eigendirection_ess"]),
                long["eigendirection_ess"][-1],
                long["potential_ess"],
                np.median(long["sd"]),
                long["half_split_gap"],
            ],
        }
    )

    print("\nShort-versus-long MYULA audit")
    print("------------------------------")
    print(rows.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"short-to-long posterior-mean gap : {posterior['postmean_gap']:.4f}")
    print(f"largest sign-probability change  : {posterior['max_sign_change']:.4f}")
    print(f"100,000-transition runtime       : {posterior['runtime']:.2f} s")



def print_decisions(
    grid: pd.DataFrame,
    posterior_long: dict,
    beta_map: np.ndarray,
    reference: dict,
) -> None:
    print("\nDecision grid")
    print("-------------")
    print(
        grid[
            [
                "k",
                "pi_star",
                "threshold",
                "map_eligible_support",
                "selected_support",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    selected_frequency = {}
    for name in FEATURE_NAMES:
        selected_frequency[name] = int(
            grid["selected_support"]
            .str.split(", ")
            .apply(lambda names: name in names if names != [""] else False)
            .sum()
        )

    coefficient_table = pd.DataFrame(
        {
            "Variable": FEATURE_NAMES,
            "MAP": beta_map,
            "Post. mean": posterior_long["mean"],
            "Post. SD": posterior_long["sd"],
            "2.5%": posterior_long["q025"],
            "97.5%": posterior_long["q975"],
            "Activation": reference["activation"],
            "Selected": reference["selected"].astype(int),
        }
    )

    print("\nReference coefficient summary")
    print("-----------------------------")
    print(coefficient_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSelected grid points:", selected_frequency)



def print_benchmark_summary(all_intervals: pd.DataFrame) -> None:
    frame = all_intervals.copy()
    frame["interval_available"] = (
        frame["lower"].notna() & frame["upper"].notna()
    )
    frame["excludes_zero"] = np.nan
    available = frame["interval_available"]
    frame.loc[available, "excludes_zero"] = (
        (frame.loc[available, "lower"] > 0.0)
        | (frame.loc[available, "upper"] < 0.0)
    ).astype(int)

    method_order = [
        "ProxMCMC",
        "bls-RJ-T",
        "bls-RJ-F",
        "Horseshoe",
        "SelInf",
        OUR_METHOD,
    ]

    pivot = (
        frame.loc[available]
        .pivot(index="feature_name", columns="method", values="excludes_zero")
        .reindex(FEATURE_NAMES)
        .reindex(columns=method_order)
    )

    print("\n95% interval zero-exclusion pattern")
    print("-----------------------------------")
    print(pivot.to_string())


# ============================================================
# Main run
# ============================================================


def main() -> None:
    total_start = time.perf_counter()

    print("Study 3: diabetes data and external benchmarking")
    print("=" * 53)

    data = load_diabetes()
    sapg_fit = run_sapg(data)

    # One final empirical-Bayes value is used everywhere below.
    theta_hat = float(sapg_fit["theta_hat"])

    map_fit = run_maps(data, theta_hat)
    posterior = run_posterior(data, theta_hat, map_fit["beta_map"])

    grid, gate_detail, reference = evaluate_decision_grid(
        posterior["long_samples"],
        map_fit["beta_map"],
    )

    zhou, zhou_metadata = load_zhou_benchmark()
    all_intervals = build_interval_table(
        zhou,
        posterior["long"],
        reference,
    )

    print_data_and_calibration(data, sapg_fit)
    print_map_summary(data, map_fit)
    print_posterior_summary(posterior)
    print_decisions(
        grid,
        posterior["long"],
        map_fit["beta_map"],
        reference,
    )
    print_benchmark_summary(all_intervals)

    print("\nZhou benchmark provenance")
    print("-------------------------")
    for field in ("source_authors", "source_title", "source_item", "source_role"):
        row = zhou_metadata.loc[zhou_metadata["field"] == field, "value"]
        if not row.empty:
            print(f"{field:16s}: {row.iloc[0]}")

    plot_benchmark_intervals(all_intervals)
    plot_eigendirection_ess(
        data["eigvals_xtx"],
        data["sigma2_hat"],
        posterior["short"]["eigendirection_ess"],
        posterior["long"]["eigendirection_ess"],
    )

    print("\nFigures written to:")
    print(f"  {FIGURE_DIR / 'diabetes_benchmarking.pdf'}")
    print(f"  {FIGURE_DIR / 'diabetes_eigendirection_ess_vs_curvature.pdf'}")
    print(f"\nTotal runtime: {time.perf_counter() - total_start:.2f} s")


if __name__ == "__main__":
    main()
