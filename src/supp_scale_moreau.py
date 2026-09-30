"""
Supplementary sensitivity study: observation scale and Moreau parameter.

This script reproduces the final sensitivity analysis reported in the
supplement. One fixed synthetic regression dataset is used throughout.

The script has two parts:

1. Observation-variance sensitivity.
   The supplied variance is multiplied by c_sigma while the absolute
   Moreau parameter is kept fixed. The leverage-adjusted residual MAD
   branch is included as a finite-sample anchor. Two severe downward
   variance branches are also rerun for approximately the same total
   Langevin time as the baseline calculation.

2. Moreau-parameter sensitivity.
   The supplied variance is fixed and lambda/lambda_0 is varied over
   {0.25, 0.5, 1, 2, 4}. SAPG is rerun at every value.

Numerical summaries are printed to the terminal. No CSV, NPZ, figure,
or intermediate result files are written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from diagnostics import effective_sample_size
from fista import fista_map
from myula import myula
from sapg import sapg_laplace
from selection import activation_probabilities, posterior_scale_threshold


# ============================================================
# Study settings
# ============================================================

DATA_SEED = 2_126_090_202
SAPG_SEED = 4_126_090_202
POST_SEED = 5_126_090_202

N = 200
P = 100
SIGMA_TRUE = 1.0

BETA_TRUE = np.zeros(P, dtype=float)
BETA_TRUE[:8] = np.array(
    [1.0, 0.8, 0.6, 0.4, -0.8, -0.5, -0.3, -0.2],
    dtype=float,
)
TRUE_MASK = BETA_TRUE != 0.0
TRUE_SIZE = int(np.count_nonzero(TRUE_MASK))
BETA_TRUE_NORM = float(np.linalg.norm(BETA_TRUE))

K_GRID = (2.0, 2.5, 3.0)
PI_GRID = (0.50, 0.75, 0.90)
K_REFERENCE = 2.5
PI_REFERENCE = 0.75

SAPG_N_ITER = 15_000
SAPG_AVERAGE_START = 4_000
SAPG_MCMC_STEPS = 1
SAPG_MCMC_WARMUP = 0
SAPG_STEP_SCALE = 1.0
SAPG_STEP_OFFSET = 200.0
SAPG_STEP_POWER = 1.0
SAPG_BOUND_FACTOR = 1e6
SAPG_C_DELTA = 0.9

MAP_TOL = 1e-8
MAP_MAX_ITER = 20_000

POST_BURN = 4_000
POST_KEEP = 4_000
POST_THIN = 1
POST_C_DELTA = 0.9

C_SIGMA_GRID = (2.0, 1.0, 0.5, 0.1, 0.01)
C_SIGMA_RUN_ORDER = (1.0, "mad", 2.0, 0.5, 0.1, 0.01)

LAMBDA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDA_RUN_ORDER = (1.0, 0.5, 2.0, 0.25, 4.0)

# These two long posterior calculations reproduce the matched-total-time
# audits reported in the supplement. They are part of the final study.
RUN_LONG_AUDITS = True
AUDIT_C_SIGMA = (0.1, 0.01)
AUDIT_KEEP = 4_000

NORMAL_MAD_CONSTANT = 0.6744897501960817
EXPECTED_C_MAD = 0.8630186373901921


# ============================================================
# Fixed dataset and preliminary scales
# ============================================================

def make_fixed_dataset() -> dict:
    """Generate the one dataset used in both sensitivity experiments."""
    rng = np.random.default_rng(DATA_SEED)

    X = rng.normal(size=(N, P))
    norms = np.linalg.norm(X, axis=0)
    X = X * (np.sqrt(N) / norms)

    noise = rng.normal(scale=SIGMA_TRUE, size=N)
    y = X @ BETA_TRUE + noise

    beta_ols, _, rank_X, _ = np.linalg.lstsq(X, y, rcond=None)
    rank_X = int(rank_X)

    residual = y - X @ beta_ols
    rss = float(residual @ residual)
    df_resid = int(N - rank_X)

    if df_resid <= 0:
        raise RuntimeError("The fixed dataset has no residual degrees of freedom.")

    sigma2_hat = float(rss / df_resid)
    sigma_hat = float(np.sqrt(sigma2_hat))

    # Leverage-adjusted residual MAD.
    Q, _ = np.linalg.qr(X, mode="reduced")
    leverage = np.sum(Q * Q, axis=1)

    if np.any(leverage >= 1.0):
        raise RuntimeError("Encountered leverage >= 1 in the fixed dataset.")

    adjusted_residual = residual / np.sqrt(1.0 - leverage)
    centre = float(np.median(adjusted_residual))
    sigma_mad = float(
        np.median(np.abs(adjusted_residual - centre)) / NORMAL_MAD_CONSTANT
    )
    sigma2_mad = float(sigma_mad**2)
    c_mad = float(sigma2_mad / sigma2_hat)

    if not np.isclose(c_mad, EXPECTED_C_MAD, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "MAD-anchor mismatch. The generated dataset does not match "
            "the frozen sensitivity-analysis dataset."
        )

    XtX = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(XtX)
    lambda_max = float(eigvals[-1])
    Lf0 = float(lambda_max / sigma2_hat)
    lambda0 = float(1.0 / Lf0)

    theta0 = float(P / np.sum(np.abs(beta_ols)))

    return {
        "X": X,
        "y": y,
        "noise": noise,
        "beta_ols": beta_ols,
        "sigma2_hat": sigma2_hat,
        "sigma_hat": sigma_hat,
        "sigma_mad": sigma_mad,
        "sigma2_mad": sigma2_mad,
        "c_mad": c_mad,
        "XtX": XtX,
        "lambda_max_XtX": lambda_max,
        "Lf0": Lf0,
        "lambda0": lambda0,
        "theta0": theta0,
        "v_weak": np.asarray(eigvecs[:, 0], dtype=float),
        "v_strong": np.asarray(eigvecs[:, -1], dtype=float),
    }


# ============================================================
# Small numerical helpers
# ============================================================

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def moreau_l1_value(
    beta: np.ndarray,
    theta: float,
    lambda_my: float,
) -> float:
    """Moreau envelope of theta * ||beta||_1."""
    prox_beta = soft_threshold(beta, lambda_my * theta)
    difference = beta - prox_beta
    return float(
        theta * np.sum(np.abs(prox_beta))
        + 0.5 * float(difference @ difference) / lambda_my
    )


def smoothed_objective_and_grad(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    theta: float,
    lambda_my: float,
) -> tuple[float, np.ndarray]:
    residual = X @ beta - y
    value_data = 0.5 * float(residual @ residual) / sigma2
    grad_data = X.T @ residual / sigma2

    prox_beta = soft_threshold(beta, lambda_my * theta)
    grad_moreau = (beta - prox_beta) / lambda_my

    value = value_data + moreau_l1_value(beta, theta, lambda_my)
    grad = grad_data + grad_moreau

    return float(value), np.asarray(grad, dtype=float)


def smoothed_map_lbfgs(
    beta_start: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    theta: float,
    lambda_my: float,
):
    """Compute the smoothed MAP diagnostic used in the study notebooks."""
    return minimize(
        fun=lambda b: smoothed_objective_and_grad(
            b, y, X, sigma2, theta, lambda_my
        ),
        x0=np.asarray(beta_start, dtype=float).copy(),
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxiter": 20_000,
            "maxls": 50,
        },
    )


def relative_l2(estimate: np.ndarray, truth: np.ndarray = BETA_TRUE) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def safe_ess(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size < 2:
        return np.nan
    if np.all(x == x[0]):
        return float(x.size)

    return float(effective_sample_size(x))


def coordinate_ess(samples: np.ndarray) -> np.ndarray:
    return np.asarray(
        [safe_ess(samples[:, j]) for j in range(samples.shape[1])],
        dtype=float,
    )


def support_metrics(selected: np.ndarray) -> dict:
    selected = np.asarray(selected, dtype=bool)

    tp = int(np.count_nonzero(selected & TRUE_MASK))
    fp = int(np.count_nonzero(selected & ~TRUE_MASK))
    fn = int(np.count_nonzero(~selected & TRUE_MASK))
    size = int(np.count_nonzero(selected))

    return {
        "size": size,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tpr": float(tp / TRUE_SIZE),
        "fdr": float(fp / max(size, 1)),
        "exact": int(np.array_equal(selected, TRUE_MASK)),
    }


def activation_diagnostics(
    samples: np.ndarray,
    threshold: float,
    pi_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activation probabilities, MCSEs, and distances to pi_star."""
    indicators = (np.abs(samples) >= threshold).astype(float)
    probabilities = np.mean(indicators, axis=0)

    ess = np.empty(P, dtype=float)
    mcse = np.empty(P, dtype=float)
    distance = np.empty(P, dtype=float)

    for j in range(P):
        z = indicators[:, j]

        if np.all(z == z[0]):
            ess[j] = float(z.size)
            mcse[j] = 0.0
            distance[j] = 0.0 if np.isclose(probabilities[j], pi_star) else np.inf
            continue

        ess_j = safe_ess(z)
        ess[j] = ess_j
        mcse_j = np.sqrt(
            probabilities[j] * (1.0 - probabilities[j]) / ess_j
        )
        mcse[j] = mcse_j

        if mcse_j > 0.0:
            distance[j] = abs(probabilities[j] - pi_star) / mcse_j
        else:
            distance[j] = np.inf

    return probabilities, mcse, distance


def finite_min(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.inf
    return float(np.min(x[finite]))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = int(np.count_nonzero(mask_a | mask_b))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    return float(intersection / union)


# ============================================================
# SAPG and posterior branch runners
# ============================================================

def calibrate_theta(
    data: dict,
    sigma2: float,
    lambda_my: float,
) -> float:
    """Run the frozen SAPG calculation for one branch."""
    rng = np.random.default_rng(SAPG_SEED)

    result = sapg_laplace(
        data["y"],
        data["X"],
        sigma2,
        rng=rng,
        n_iter=SAPG_N_ITER,
        theta_init=data["theta0"],
        beta_ref=data["beta_ols"],
        beta0=data["beta_ols"].copy(),
        intrinsic_dim=P,
        constraint="none",
        theta_bounds=None,
        bound_factor=SAPG_BOUND_FACTOR,
        step_scale=SAPG_STEP_SCALE,
        step_offset=SAPG_STEP_OFFSET,
        step_power=SAPG_STEP_POWER,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=SAPG_MCMC_STEPS,
        mcmc_warmup=SAPG_MCMC_WARMUP,
        lambda_my=lambda_my,
        step_size_myula=None,
        c_delta=SAPG_C_DELTA,
        store_state_path=False,
    )

    return float(result.theta_hat)


def run_posterior_branch(
    data: dict,
    sigma2: float,
    theta_hat: float,
    lambda_my: float,
    *,
    burn_in: int = POST_BURN,
    n_samples: int = POST_KEEP,
    thin: int = POST_THIN,
) -> dict:
    """Run MAP, smoothed-MAP diagnostic, MYULA, and the gate grid."""
    X = data["X"]
    y = data["y"]

    map_result = fista_map(
        y,
        X,
        sigma2,
        theta_hat,
        beta0=data["beta_ols"].copy(),
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    beta_map = np.asarray(map_result.beta, dtype=float)

    smooth_result = smoothed_map_lbfgs(
        beta_map,
        y,
        X,
        sigma2,
        theta_hat,
        lambda_my,
    )
    beta_map_smooth = np.asarray(smooth_result.x, dtype=float)

    posterior_rng = np.random.default_rng(POST_SEED)
    post_result = myula(
        y,
        X,
        sigma2,
        theta_hat,
        rng=posterior_rng,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        beta0=beta_map.copy(),
        lambda_my=lambda_my,
        step_size=None,
        c_delta=POST_C_DELTA,
        store_potential=False,
    )

    samples = np.asarray(post_result.samples, dtype=float)
    posterior_mean = np.mean(samples, axis=0)
    posterior_sd = np.std(samples, axis=0, ddof=1)

    coord_ess = coordinate_ess(samples)
    weak_ess = safe_ess(samples @ data["v_weak"])
    strong_ess = safe_ess(samples @ data["v_strong"])

    grid = {}
    for k in K_GRID:
        threshold, _ = posterior_scale_threshold(samples, k)
        threshold = float(threshold)
        probabilities = activation_probabilities(samples, threshold)

        for pi_star in PI_GRID:
            selected = (
                (np.abs(beta_map) >= threshold)
                & (probabilities >= pi_star)
            )
            metrics = support_metrics(selected)

            _, _, distance = activation_diagnostics(
                samples,
                threshold,
                float(pi_star),
            )
            map_eligible = np.abs(beta_map) >= threshold
            d_eligible = (
                finite_min(distance[map_eligible])
                if np.any(map_eligible)
                else np.nan
            )

            grid[(float(k), float(pi_star))] = {
                "threshold": threshold,
                "selected": selected.copy(),
                "activation_probability": probabilities.copy(),
                "size": metrics["size"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tpr": metrics["tpr"],
                "fdr": metrics["fdr"],
                "exact": metrics["exact"],
                "D_eligible": d_eligible,
            }

    reference = grid[(K_REFERENCE, PI_REFERENCE)]

    smooth_gap = float(
        np.linalg.norm(beta_map_smooth - beta_map)
        / max(np.linalg.norm(beta_map), 1e-12)
    )

    return {
        "theta_hat": float(theta_hat),
        "sigma2": float(sigma2),
        "lambda_my": float(lambda_my),
        "beta_map": beta_map,
        "beta_map_smooth": beta_map_smooth,
        "samples": samples,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "median_sd": float(np.median(posterior_sd)),
        "map_rel_l2": relative_l2(beta_map),
        "postmean_rel_l2": relative_l2(posterior_mean),
        "smooth_gap": smooth_gap,
        "median_ess": float(np.median(coord_ess)),
        "min_ess": float(np.min(coord_ess)),
        "weak_ess": float(weak_ess),
        "strong_ess": float(strong_ess),
        "delta": float(post_result.step_size),
        "n_total_steps": int(post_result.n_total_steps),
        "diffusion_time": float(post_result.n_total_steps * post_result.step_size),
        "grid": grid,
        "reference": reference,
    }


# ============================================================
# Observation-variance sensitivity
# ============================================================

def run_variance_sensitivity(data: dict) -> dict:
    """Run the five reported c_sigma branches plus the MAD anchor."""
    branches = {}

    for item in C_SIGMA_RUN_ORDER:
        if item == "mad":
            c_sigma = float(data["c_mad"])
            label = "MAD"
        else:
            c_sigma = float(item)
            label = f"{c_sigma:g}"

        sigma2 = float(c_sigma * data["sigma2_hat"])
        theta_hat = calibrate_theta(data, sigma2, data["lambda0"])
        branch = run_posterior_branch(
            data,
            sigma2,
            theta_hat,
            data["lambda0"],
        )
        branch["c_sigma"] = c_sigma
        branch["label"] = label
        branch["Lf_over_Lf0"] = float(1.0 / c_sigma)
        branch["effective_penalty"] = float(sigma2 * theta_hat)
        branches[label] = branch

    return branches


def print_variance_results(data: dict, branches: dict) -> None:
    print("\n" + "=" * 78)
    print("OBSERVATION-VARIANCE SENSITIVITY")
    print("=" * 78)

    print(f"baseline sigma^2                 = {data['sigma2_hat']:.6f}")
    print(f"baseline sigma                   = {data['sigma_hat']:.6f}")
    print(f"leverage-adjusted MAD sigma      = {data['sigma_mad']:.6f}")
    print(f"leverage-adjusted MAD sigma^2    = {data['sigma2_mad']:.6f}")
    print(f"c_MAD                            = {data['c_mad']:.6f}")
    print(f"L_f(c_MAD) / L_f,0              = {1.0 / data['c_mad']:.6f}")
    print(f"lambda_0                         = {data['lambda0']:.10e}")

    rows = []
    for c_sigma in C_SIGMA_GRID:
        b = branches[f"{c_sigma:g}"]
        ref = b["reference"]
        rows.append(
            {
                "c_sigma": c_sigma,
                "sigma2": b["sigma2"],
                "Lf/Lf0": b["Lf_over_Lf0"],
                "theta": b["theta_hat"],
                "sigma2*theta": b["effective_penalty"],
                "MAP rel L2": b["map_rel_l2"],
                "Med SD": b["median_sd"],
                "tau_post": ref["threshold"],
                "Size": ref["size"],
                "TPR": ref["tpr"],
                "FDR": ref["fdr"],
            }
        )

    table = pd.DataFrame(rows)
    print("\nReported variance sweep:")
    print(
        table.to_string(
            index=False,
            formatters={
                "c_sigma": lambda x: f"{x:g}",
                "sigma2": lambda x: f"{x:.6f}",
                "Lf/Lf0": lambda x: f"{x:g}",
                "theta": lambda x: f"{x:.6f}",
                "sigma2*theta": lambda x: f"{x:.5g}",
                "MAP rel L2": lambda x: f"{x:.6f}",
                "Med SD": lambda x: f"{x:.6f}",
                "tau_post": lambda x: f"{x:.6f}",
                "TPR": lambda x: f"{x:.3f}",
                "FDR": lambda x: f"{x:.6f}",
            },
        )
    )

    baseline = branches["1"]
    mad = branches["MAD"]

    print("\nMAD anchor against the baseline:")
    print(f"theta:             {baseline['theta_hat']:.6f} -> {mad['theta_hat']:.6f}")
    print(
        "sigma^2*theta:     "
        f"{baseline['effective_penalty']:.4f} -> {mad['effective_penalty']:.4f}"
    )
    print(f"median SD:         {baseline['median_sd']:.5f} -> {mad['median_sd']:.5f}")
    print(
        "reference tau:     "
        f"{baseline['reference']['threshold']:.5f} -> {mad['reference']['threshold']:.5f}"
    )
    print(
        "reference support: "
        f"{baseline['reference']['size']} -> {mad['reference']['size']} variables"
    )

    agreement_rows = []
    for k in K_GRID:
        for pi_star in PI_GRID:
            base_mask = baseline["grid"][(k, pi_star)]["selected"]
            mad_mask = mad["grid"][(k, pi_star)]["selected"]
            agreement_rows.append(
                {
                    "k": k,
                    "pi_star": pi_star,
                    "same_support": bool(np.array_equal(base_mask, mad_mask)),
                    "baseline_size": int(np.count_nonzero(base_mask)),
                    "mad_size": int(np.count_nonzero(mad_mask)),
                }
            )

    agreement = pd.DataFrame(agreement_rows)
    print("\nBaseline/MAD decision-grid comparison:")
    print(agreement.to_string(index=False))
    print(
        f"Agreement in {int(agreement['same_support'].sum())}/"
        f"{len(agreement)} settings."
    )


def run_long_variance_audits(data: dict, branches: dict) -> list[dict]:
    """Match the baseline total Langevin time for c_sigma=0.1 and 0.01."""
    baseline = branches["1"]
    target_time = float(baseline["diffusion_time"])

    audit_rows = []

    for c_sigma in AUDIT_C_SIGMA:
        short = branches[f"{c_sigma:g}"]
        delta = float(short["delta"])

        target_total_steps = int(np.round(target_time / delta))

        # Nearest-integer rule used in the original notebook.
        thin = int(np.floor(target_total_steps / (2.0 * AUDIT_KEEP) + 0.5))
        thin = max(thin, 1)
        burn_in = int(target_total_steps - AUDIT_KEEP * thin)

        if burn_in < 0:
            thin = max(thin - 1, 1)
            burn_in = int(target_total_steps - AUDIT_KEEP * thin)

        audit = run_posterior_branch(
            data,
            short["sigma2"],
            short["theta_hat"],
            data["lambda0"],
            burn_in=burn_in,
            n_samples=AUDIT_KEEP,
            thin=thin,
        )

        short_mask = short["reference"]["selected"]
        audit_mask = audit["reference"]["selected"]

        audit_rows.append(
            {
                "c_sigma": c_sigma,
                "total_steps": audit["n_total_steps"],
                "median_ess_short": short["median_ess"],
                "median_ess_audit": audit["median_ess"],
                "weak_ess_short": short["weak_ess"],
                "weak_ess_audit": audit["weak_ess"],
                "D_elig_short": short["reference"]["D_eligible"],
                "D_elig_audit": audit["reference"]["D_eligible"],
                "size_short": short["reference"]["size"],
                "size_audit": audit["reference"]["size"],
                "fdr_short": short["reference"]["fdr"],
                "fdr_audit": audit["reference"]["fdr"],
                "jaccard": jaccard(short_mask, audit_mask),
            }
        )

    print("\nMatched-total-time posterior audits:")
    table = pd.DataFrame(audit_rows)
    print(
        table.to_string(
            index=False,
            formatters={
                "c_sigma": lambda x: f"{x:g}",
                "median_ess_short": lambda x: f"{x:.1f}",
                "median_ess_audit": lambda x: f"{x:.1f}",
                "weak_ess_short": lambda x: f"{x:.1f}",
                "weak_ess_audit": lambda x: f"{x:.1f}",
                "D_elig_short": lambda x: f"{x:.4f}",
                "D_elig_audit": lambda x: f"{x:.4f}",
                "fdr_short": lambda x: f"{x:.4f}",
                "fdr_audit": lambda x: f"{x:.4f}",
                "jaccard": lambda x: f"{x:.4f}",
            },
        )
    )

    return audit_rows


# ============================================================
# Moreau-parameter sensitivity
# ============================================================

def run_moreau_sensitivity(data: dict, variance_branches: dict) -> dict:
    """Run the five lambda/lambda_0 branches at fixed sigma^2."""
    branches = {}

    # The multiplier-one branch is exactly the baseline variance branch:
    # same dataset, sigma^2, lambda, SAPG settings, and posterior seed.
    branches[1.0] = variance_branches["1"]

    for multiplier in LAMBDA_RUN_ORDER:
        multiplier = float(multiplier)
        if np.isclose(multiplier, 1.0):
            continue

        lambda_my = float(multiplier * data["lambda0"])
        theta_hat = calibrate_theta(
            data,
            data["sigma2_hat"],
            lambda_my,
        )
        branch = run_posterior_branch(
            data,
            data["sigma2_hat"],
            theta_hat,
            lambda_my,
        )
        branches[multiplier] = branch

    baseline_map = branches[1.0]["beta_map"]
    baseline_map_norm = max(float(np.linalg.norm(baseline_map)), 1e-12)

    for multiplier, branch in branches.items():
        branch["lambda_multiplier"] = float(multiplier)
        branch["lambda_theta"] = float(branch["lambda_my"] * branch["theta_hat"])
        branch["calibration_gap"] = float(
            np.linalg.norm(branch["beta_map"] - baseline_map) / baseline_map_norm
        )

    return branches


def print_moreau_results(data: dict, branches: dict) -> None:
    print("\n" + "=" * 78)
    print("MOREAU-PARAMETER SENSITIVITY")
    print("=" * 78)

    rows_a = []
    rows_b = []

    for multiplier in LAMBDA_MULTIPLIERS:
        b = branches[float(multiplier)]
        ref = b["reference"]

        rows_a.append(
            {
                "lambda/lambda0": multiplier,
                "theta": b["theta_hat"],
                "lambda*theta": b["lambda_theta"],
                "Delta_calib": b["calibration_gap"],
                "Delta_smooth": b["smooth_gap"],
                "MAP rel L2": b["map_rel_l2"],
                "Postmean rel L2": b["postmean_rel_l2"],
                "Med SD": b["median_sd"],
            }
        )

        rows_b.append(
            {
                "lambda/lambda0": multiplier,
                "delta": b["delta"],
                "Med ESS": b["median_ess"],
                "Weak ESS": b["weak_ess"],
                "tau_post": ref["threshold"],
                "Size": ref["size"],
                "TPR": ref["tpr"],
                "FDR": ref["fdr"],
                "D_elig": ref["D_eligible"],
            }
        )

    table_a = pd.DataFrame(rows_a)
    table_b = pd.DataFrame(rows_b)

    print("\nEmpirical-Bayes, approximation, and estimation summaries:")
    print(
        table_a.to_string(
            index=False,
            formatters={
                "lambda/lambda0": lambda x: f"{x:g}",
                "theta": lambda x: f"{x:.6f}",
                "lambda*theta": lambda x: f"{x:.6f}",
                "Delta_calib": lambda x: f"{x:.6f}",
                "Delta_smooth": lambda x: f"{x:.6f}",
                "MAP rel L2": lambda x: f"{x:.6f}",
                "Postmean rel L2": lambda x: f"{x:.6f}",
                "Med SD": lambda x: f"{x:.6f}",
            },
        )
    )

    print("\nPosterior-computation and reference-decision summaries:")
    print(
        table_b.to_string(
            index=False,
            formatters={
                "lambda/lambda0": lambda x: f"{x:g}",
                "delta": lambda x: f"{x:.6f}",
                "Med ESS": lambda x: f"{x:.1f}",
                "Weak ESS": lambda x: f"{x:.1f}",
                "tau_post": lambda x: f"{x:.6f}",
                "TPR": lambda x: f"{x:.3f}",
                "FDR": lambda x: f"{x:.3f}",
                "D_elig": lambda x: f"{x:.2f}",
            },
        )
    )

    baseline_mask = branches[1.0]["reference"]["selected"]
    jaccards = {
        multiplier: jaccard(branches[multiplier]["reference"]["selected"], baseline_mask)
        for multiplier in LAMBDA_MULTIPLIERS
    }
    print("\nReference-support Jaccard indices relative to lambda/lambda0=1:")
    for multiplier in LAMBDA_MULTIPLIERS:
        print(f"  {multiplier:g}: {jaccards[multiplier]:.6f}")


# ============================================================
# Main
# ============================================================

def main(*, run_long_audits: bool = RUN_LONG_AUDITS) -> None:
    data = make_fixed_dataset()

    print("=" * 78)
    print("SUPPLEMENTARY SCALE / MOREAU SENSITIVITY STUDY")
    print("=" * 78)
    print(f"n, p                             = {N}, {P}")
    print(f"true sigma                       = {SIGMA_TRUE:.6f}")
    print(f"fixed data seed                  = {DATA_SEED}")
    print(f"SAPG seed                        = {SAPG_SEED}")
    print(f"posterior seed                   = {POST_SEED}")

    variance_branches = run_variance_sensitivity(data)
    print_variance_results(data, variance_branches)

    if run_long_audits:
        run_long_variance_audits(data, variance_branches)
    else:
        print("\nLong matched-time audits skipped for this run.")

    moreau_branches = run_moreau_sensitivity(data, variance_branches)
    print_moreau_results(data, moreau_branches)


if __name__ == "__main__":
    main()
