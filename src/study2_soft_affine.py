"""
Study 2: soft affine information and numerical geometry.

This script reproduces the final soft-affine simulation study in the
paper. It uses R=100 datasets and the affine-curvature grid

    kappa_c in {0, 0.1, 1, 10}.

For kappa_c <= 1, the posterior calculation uses scalar-step MYULA.
At kappa_c = 10, the script also runs the scalar-step calculation used
as a numerical reference and the retained rank-one preconditioned
calculation used in the paper.

The script also reproduces the reported timestep audit for the retained
kappa_c = 10 calculation. Numerical summaries are printed to the
terminal. No CSV or intermediate result files are written.
"""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fista import fista_map, smoothed_map
from sapg import sapg_laplace
from selection import activation_probabilities, posterior_scale_threshold, posterior_sd


# ============================================================
# Study settings
# ============================================================

N = 200
P = 100
R = 100

SIGMA = 1.0
SIGMA2 = SIGMA**2

BETA_TRUE = np.zeros(P, dtype=float)
BETA_TRUE[:8] = np.array(
    [1.0, 0.8, 0.6, 0.4, -0.8, -0.5, -0.3, -0.2],
    dtype=float,
)
TRUE_MASK = BETA_TRUE != 0.0
TRUE_SIZE = int(np.count_nonzero(TRUE_MASK))
BETA_TRUE_NORM = float(np.linalg.norm(BETA_TRUE))

AFFINE_A = np.ones(P, dtype=float)
AFFINE_B = 1.0
SUM_DIRECTION = AFFINE_A / np.linalg.norm(AFFINE_A)
P_PARALLEL = np.outer(SUM_DIRECTION, SUM_DIRECTION)
P_PERP = np.eye(P) - P_PARALLEL

KAPPA_GRID = (0.0, 0.1, 1.0, 10.0)
COMMON_KAPPAS = (0.0, 0.1, 1.0)

K_GRID = (2.0, 2.5, 3.0)
PI_GRID = (0.50, 0.75, 0.90)
K_REFERENCE = 2.5
PI_REFERENCE = 0.75

SAPG_N_ITER = 15_000
SAPG_AVERAGE_START = 4_000
SAPG_STEP_SCALE = 1.0
SAPG_STEP_OFFSET = 200.0
SAPG_STEP_POWER = 1.0
SAPG_BOUND_FACTOR = 1e6

MAP_TOL = 1e-8
MAP_MAX_ITER = 20_000

POST_BURN = 4_000
POST_KEEP = 4_000
C_DELTA = 0.9

# Retained kappa_c=10 calculation.
KAPPA10 = 10.0
K10_REFERENCE_C_DELTA = 0.9
K10_PRECONDITIONED_C_DELTA = 0.45
K10_PRECONDITIONED_BURN = 8_000
K10_PRECONDITIONED_KEEP = 4_000
K10_PRECONDITIONED_THIN = 2

# Paired production streams.
DATA_SEED_BASE = 31_000_000
SAPG_SEED_BASE = 41_000_000
POST_SEED_BASE = 51_000_000
K10_REFERENCE_POST_SEED_BASE = 61_000_000
K10_PRECONDITIONED_POST_SEED_BASE = 71_000_000

# The timestep audit was run on one fixed diagnostic dataset.
AUDIT_MASTER_SEED = 2026090202
AUDIT_DATA_SEED = AUDIT_MASTER_SEED + 100_000_000
AUDIT_SAPG_SEED = AUDIT_MASTER_SEED + 110_000_000
AUDIT_BROWNIAN_SEED = AUDIT_MASTER_SEED + 140_000_000
AUDIT_C_DELTA = (0.900, 0.450, 0.225)

FIGURE_DIR = Path("figures")




# ============================================================
# ESS convention used in the original Study 2 calculation
# ============================================================

def autocorrelation_1d(x: np.ndarray) -> np.ndarray:
    """FFT autocorrelation with lag-specific covariance denominators."""
    x = np.asarray(x, dtype=float)
    n = x.size

    if n < 2:
        return np.ones(n, dtype=float)

    x = x - np.mean(x)
    variance = float(np.mean(x**2))

    if variance <= np.finfo(float).eps:
        return np.ones(n, dtype=float)

    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=n_fft)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=n_fft)[:n]
    acov = acov / np.arange(n, 0, -1, dtype=float)

    rho = acov / acov[0]
    rho[0] = 1.0
    return rho


def ess_1d(x: np.ndarray) -> float:
    """ESS with the initial-positive-pair rule used in the study notebook."""
    x = np.asarray(x, dtype=float)
    n = x.size

    if n < 3 or np.var(x) <= np.finfo(float).eps:
        return float(n)

    rho = autocorrelation_1d(x)
    pair_sum = []
    k = 1

    while k + 1 < n:
        pair = float(rho[k] + rho[k + 1])
        if pair <= 0.0:
            break
        pair_sum.append(pair)
        k += 2

    tau = max(1.0, 1.0 + 2.0 * float(np.sum(pair_sum)))
    return float(np.clip(n / tau, 1.0, float(n)))


# ============================================================
# Data and geometry
# ============================================================

def generate_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one iid-Gaussian design and response."""
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(N, P))
    norms = np.linalg.norm(X, axis=0)
    X = X * (np.sqrt(N) / norms)

    noise = rng.normal(scale=SIGMA, size=N)
    y = X @ BETA_TRUE + noise

    return X, y


def geometry(X: np.ndarray, kappa_c: float) -> dict:
    """Return the likelihood and affine-curvature quantities."""
    H0 = (X.T @ X) / SIGMA2
    L0 = float(np.linalg.eigvalsh(H0)[-1])

    if np.isclose(kappa_c, 0.0):
        tau_c = None
        Hf = H0.copy()
    else:
        tau_c = float(np.sqrt(P / (kappa_c * L0)))
        Hf = H0 + np.outer(AFFINE_A, AFFINE_A) / tau_c**2

    Lf = float(np.linalg.eigvalsh(Hf)[-1])

    # Tangent-space curvature. The projected matrix has one numerical
    # zero in the sum direction; the remaining eigenvectors are tangent.
    H_perp = P_PERP @ H0 @ P_PERP
    evals, evecs = np.linalg.eigh(H_perp)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    positive = np.flatnonzero(evals > 1e-8)
    weak_index = int(positive[0])
    strong_index = int(positive[-1])

    L_perp = float(evals[strong_index])

    return {
        "H0": H0,
        "Hf": Hf,
        "L0": L0,
        "Lf": Lf,
        "L_perp": L_perp,
        "tau_c": tau_c,
        "u_weak": evecs[:, weak_index],
        "u_strong": evecs[:, strong_index],
    }


# ============================================================
# Small helpers
# ============================================================

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def relative_l2(beta: np.ndarray) -> float:
    return float(np.linalg.norm(beta - BETA_TRUE) / BETA_TRUE_NORM)


def prediction_risk(X: np.ndarray, beta: np.ndarray) -> float:
    error = X @ (beta - BETA_TRUE)
    return float(np.mean(error**2))


def support_metrics(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    tp = int(np.count_nonzero(mask & TRUE_MASK))
    fp = int(np.count_nonzero(mask & ~TRUE_MASK))
    size = int(np.count_nonzero(mask))

    return {
        "size": size,
        "tp": tp,
        "fp": fp,
        "tpr": float(tp / TRUE_SIZE),
        "fdp": float(fp / max(size, 1)),
        "exact": int(np.array_equal(mask, TRUE_MASK)),
    }


def decision_distance(
    samples: np.ndarray,
    threshold: float,
    pi_star: float,
    map_eligible: np.ndarray,
) -> tuple[float, float]:
    """Return the smallest decision distance overall and among MAP-eligible coordinates."""
    indicators = (np.abs(samples) >= threshold).astype(float)
    pi_hat = indicators.mean(axis=0)
    distances = np.empty(P, dtype=float)

    for j in range(P):
        variance = float(pi_hat[j] * (1.0 - pi_hat[j]))

        if variance == 0.0:
            distances[j] = 0.0 if np.isclose(pi_hat[j], pi_star) else np.inf
            continue

        ess = ess_1d(indicators[:, j])
        mcse = np.sqrt(variance / max(ess, 1.0))
        distances[j] = abs(pi_hat[j] - pi_star) / mcse

    finite_all = distances[np.isfinite(distances)]
    eligible = distances[np.asarray(map_eligible, dtype=bool)]
    finite_eligible = eligible[np.isfinite(eligible)]

    min_all = float(np.min(finite_all)) if finite_all.size else np.inf
    min_eligible = (
        float(np.min(finite_eligible)) if finite_eligible.size else np.inf
    )

    return min_all, min_eligible


def posterior_diagnostics(
    samples: np.ndarray,
    u_weak: np.ndarray,
    u_strong: np.ndarray,
) -> dict:
    """Compute the posterior summaries used in the paper."""
    mean = samples.mean(axis=0)
    sd = posterior_sd(samples)

    coordinate_ess = np.array(
        [ess_1d(samples[:, j]) for j in range(P)],
        dtype=float,
    )

    weak_ess = ess_1d(samples @ u_weak)
    strong_ess = ess_1d(samples @ u_strong)
    sum_ess = ess_1d(samples @ SUM_DIRECTION)

    sum_residual = samples.sum(axis=1) - AFFINE_B
    negative_mass = np.maximum(-samples, 0.0).sum(axis=1)
    l1_mass = np.abs(samples).sum(axis=1)

    return {
        "posterior_mean": mean,
        "median_sd": float(np.median(sd)),
        "median_coordinate_ess": float(np.median(coordinate_ess)),
        "min_coordinate_ess": float(np.min(coordinate_ess)),
        "ess_perp_weak": float(weak_ess),
        "ess_perp_strong": float(strong_ess),
        "ess_parallel": float(sum_ess),
        "mean_budget_residual": float(np.mean(sum_residual)),
        "mean_abs_budget_residual": float(np.mean(np.abs(sum_residual))),
        "mean_negative_mass": float(np.mean(negative_mass)),
        "mean_l1": float(np.mean(l1_mass)),
    }


def evaluate_gate_grid(
    rep: int,
    branch: str,
    kappa_c: float,
    X: np.ndarray,
    beta_map: np.ndarray,
    samples: np.ndarray,
) -> dict:
    """Evaluate the full 3 x 3 decision grid."""
    posterior_mean = samples.mean(axis=0)
    rows = []
    nominal = None
    D_all = np.nan
    D_eligible = np.nan

    for k_gate in K_GRID:
        threshold, _ = posterior_scale_threshold(samples, k_gate)
        activation = activation_probabilities(samples, threshold)
        map_eligible = np.abs(beta_map) >= threshold

        for pi_star in PI_GRID:
            selected = map_eligible & (activation >= pi_star)
            metrics = support_metrics(selected)

            gated_map = beta_map * selected
            gated_postmean = posterior_mean * selected

            row = {
                "rep": rep,
                "branch": branch,
                "kappa_c": float(kappa_c),
                "k_gate": float(k_gate),
                "pi_star": float(pi_star),
                "tau_post": float(threshold),
                **metrics,
                "gated_map_rel_l2": relative_l2(gated_map),
                "gated_map_pred_risk": prediction_risk(X, gated_map),
                "gated_postmean_rel_l2": relative_l2(gated_postmean),
                "gated_postmean_pred_risk": prediction_risk(X, gated_postmean),
            }
            rows.append(row)

            if np.isclose(k_gate, K_REFERENCE) and np.isclose(
                pi_star, PI_REFERENCE
            ):
                nominal = row.copy()
                D_all, D_eligible = decision_distance(
                    samples,
                    threshold,
                    pi_star,
                    map_eligible,
                )

    if nominal is None:
        raise RuntimeError("Reference gate was not found.")

    return {
        "rows": rows,
        "nominal": nominal,
        "posterior_mean": posterior_mean,
        "D_all": D_all,
        "D_eligible": D_eligible,
    }


# ============================================================
# MYULA calculations
# ============================================================

def run_scalar_myula(
    X: np.ndarray,
    y: np.ndarray,
    theta: float,
    tau_c: float | None,
    lambda_my: float,
    delta: float,
    beta0: np.ndarray,
    innovations: np.ndarray,
) -> np.ndarray:
    """Run scalar-step MYULA with supplied Gaussian innovations."""
    H0 = (X.T @ X) / SIGMA2
    linear = (X.T @ y) / SIGMA2
    beta = np.asarray(beta0, dtype=float).copy()

    expected = POST_BURN + POST_KEEP
    if innovations.shape != (expected, P):
        raise ValueError("Unexpected scalar-MYULA innovation shape.")

    samples = np.empty((POST_KEEP, P), dtype=float)
    save = 0

    for iteration in range(expected):
        grad = H0 @ beta - linear

        if tau_c is not None:
            residual = float(AFFINE_A @ beta - AFFINE_B)
            grad = grad + AFFINE_A * residual / tau_c**2

        prox_beta = soft_threshold(beta, lambda_my * theta)
        grad = grad + (beta - prox_beta) / lambda_my

        beta = (
            beta
            - delta * grad
            + np.sqrt(2.0 * delta) * innovations[iteration]
        )

        if iteration >= POST_BURN:
            samples[save] = beta
            save += 1

    if save != POST_KEEP:
        raise RuntimeError("Scalar MYULA retained the wrong number of draws.")

    return samples


def apply_metric(v: np.ndarray, gamma: float) -> np.ndarray:
    """Apply M = P_perp + gamma P_parallel without forming M."""
    return v + (gamma - 1.0) * SUM_DIRECTION * np.dot(SUM_DIRECTION, v)


def apply_metric_sqrt(v: np.ndarray, gamma: float) -> np.ndarray:
    """Apply the positive square root of the rank-one metric."""
    sqrt_gamma = float(np.sqrt(gamma))
    return v + (sqrt_gamma - 1.0) * SUM_DIRECTION * np.dot(
        SUM_DIRECTION, v
    )


def run_preconditioned_myula(
    X: np.ndarray,
    y: np.ndarray,
    theta: float,
    tau_c: float,
    lambda_my: float,
    delta: float,
    gamma: float,
    beta0: np.ndarray,
    innovations: np.ndarray,
    *,
    burn: int,
    keep: int,
    thin: int,
) -> np.ndarray:
    """Run the retained rank-one preconditioned MYULA calculation."""
    H0 = (X.T @ X) / SIGMA2
    linear = (X.T @ y) / SIGMA2
    beta = np.asarray(beta0, dtype=float).copy()

    n_total = burn + keep * thin
    if innovations.shape != (n_total, P):
        raise ValueError("Unexpected preconditioned-MYULA innovation shape.")

    samples = np.empty((keep, P), dtype=float)
    save = 0

    for iteration in range(n_total):
        grad = H0 @ beta - linear
        residual = float(AFFINE_A @ beta - AFFINE_B)
        grad = grad + AFFINE_A * residual / tau_c**2

        prox_beta = soft_threshold(beta, lambda_my * theta)
        grad = grad + (beta - prox_beta) / lambda_my

        metric_grad = apply_metric(grad, gamma)
        metric_noise = apply_metric_sqrt(innovations[iteration], gamma)

        beta = (
            beta
            - delta * metric_grad
            + np.sqrt(2.0 * delta) * metric_noise
        )

        if iteration >= burn:
            offset = iteration - burn
            if offset % thin == 0:
                samples[save] = beta
                save += 1

    if save != keep:
        raise RuntimeError("Preconditioned MYULA retained the wrong number of draws.")

    return samples


# ============================================================
# One branch calculation
# ============================================================

def fit_common_branch(
    rep: int,
    X: np.ndarray,
    y: np.ndarray,
    beta_ols: np.ndarray,
    theta0: float,
    kappa_c: float,
    common_innovations: np.ndarray,
) -> tuple[dict, list[dict]]:
    """Run one scalar-step branch for kappa_c in {0, 0.1, 1}."""
    geom = geometry(X, kappa_c)
    tau_c = geom["tau_c"]
    Lf = geom["Lf"]
    lambda_my = 1.0 / Lf
    delta = C_DELTA / (Lf + 1.0 / lambda_my)

    if tau_c is None:
        constraint = "none"
        a_arg = None
        b_arg = None
    else:
        constraint = "soft_affine"
        a_arg = AFFINE_A
        b_arg = AFFINE_B

    rng_sapg = np.random.default_rng(SAPG_SEED_BASE + rep)
    sapg_result = sapg_laplace(
        y,
        X,
        SIGMA2,
        rng=rng_sapg,
        n_iter=SAPG_N_ITER,
        theta_init=theta0,
        beta_ref=beta_ols,
        beta0=beta_ols,
        intrinsic_dim=None,
        constraint=constraint,
        a=a_arg,
        b=b_arg,
        tau_c=tau_c,
        theta_bounds=None,
        bound_factor=SAPG_BOUND_FACTOR,
        step_scale=SAPG_STEP_SCALE,
        step_offset=SAPG_STEP_OFFSET,
        step_power=SAPG_STEP_POWER,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=1,
        mcmc_warmup=0,
        lambda_my=lambda_my,
        step_size_myula=delta,
        c_delta=C_DELTA,
        store_state_path=False,
    )
    theta_hat = float(sapg_result.theta_hat)

    map_result = fista_map(
        y,
        X,
        SIGMA2,
        theta_hat,
        a=a_arg,
        b=b_arg,
        tau_c=tau_c,
        beta0=beta_ols,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    beta_map = np.asarray(map_result.beta, dtype=float)

    smooth_result = smoothed_map(
        y,
        X,
        SIGMA2,
        theta_hat,
        lambda_my,
        a=a_arg,
        b=b_arg,
        tau_c=tau_c,
        beta0=beta_map,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    smooth_shift = float(np.linalg.norm(smooth_result.beta - beta_map))

    samples = run_scalar_myula(
        X,
        y,
        theta_hat,
        tau_c,
        lambda_my,
        delta,
        beta_map,
        common_innovations,
    )
    post = posterior_diagnostics(samples, geom["u_weak"], geom["u_strong"])
    gates = evaluate_gate_grid(
        rep,
        "scalar",
        kappa_c,
        X,
        beta_map,
        samples,
    )
    nominal = gates["nominal"]

    summary = {
        "rep": rep,
        "branch": "scalar",
        "kappa_c": float(kappa_c),
        "theta0": theta0,
        "theta_hat": theta_hat,
        "tau_c": np.nan if tau_c is None else float(tau_c),
        "L0": geom["L0"],
        "Lf": Lf,
        "L_perp": geom["L_perp"],
        "lambda_my": lambda_my,
        "delta": delta,
        "smooth_map_shift_l2": smooth_shift,
        "map_rel_l2": relative_l2(beta_map),
        "postmean_rel_l2": relative_l2(post["posterior_mean"]),
        "gated_map_rel_l2": nominal["gated_map_rel_l2"],
        "gated_postmean_rel_l2": nominal["gated_postmean_rel_l2"],
        "mean_abs_budget_residual": post["mean_abs_budget_residual"],
        "median_posterior_sd": post["median_sd"],
        "median_coordinate_ess": post["median_coordinate_ess"],
        "min_coordinate_ess": post["min_coordinate_ess"],
        "ess_perp_weak": post["ess_perp_weak"],
        "ess_perp_strong": post["ess_perp_strong"],
        "ess_parallel": post["ess_parallel"],
        "nominal_size": nominal["size"],
        "nominal_tpr": nominal["tpr"],
        "nominal_fdp": nominal["fdp"],
        "nominal_exact": nominal["exact"],
        "nominal_D_min_map_eligible": gates["D_eligible"],
    }

    return summary, gates["rows"]


def fit_kappa10(
    rep: int,
    X: np.ndarray,
    y: np.ndarray,
    beta_ols: np.ndarray,
    theta0: float,
) -> tuple[list[dict], list[dict], dict]:
    """Run the scalar and retained preconditioned kappa_c=10 branches."""
    geom = geometry(X, KAPPA10)
    tau_c = float(geom["tau_c"])
    Lf = float(geom["Lf"])
    L_perp = float(geom["L_perp"])

    lambda_ref = 1.0 / Lf
    delta_ref = K10_REFERENCE_C_DELTA / (Lf + 1.0 / lambda_ref)

    lambda_pre = 1.0 / L_perp
    gamma = L_perp / Lf
    sqrt_gamma = np.sqrt(gamma)
    M_half = P_PERP + sqrt_gamma * P_PARALLEL
    Lf_metric = float(np.linalg.eigvalsh(M_half @ geom["Hf"] @ M_half)[-1])
    Ltotal_pre = Lf_metric + 1.0 / lambda_pre
    delta_pre = K10_PRECONDITIONED_C_DELTA / Ltotal_pre

    eigvals, eigvecs = np.linalg.eigh(geom["Hf"])
    top_alignment_sq = float(np.dot(eigvecs[:, -1], SUM_DIRECTION) ** 2)

    rng_sapg = np.random.default_rng(SAPG_SEED_BASE + rep)
    sapg_result = sapg_laplace(
        y,
        X,
        SIGMA2,
        rng=rng_sapg,
        n_iter=SAPG_N_ITER,
        theta_init=theta0,
        beta_ref=beta_ols,
        beta0=beta_ols,
        intrinsic_dim=None,
        constraint="soft_affine",
        a=AFFINE_A,
        b=AFFINE_B,
        tau_c=tau_c,
        theta_bounds=None,
        bound_factor=SAPG_BOUND_FACTOR,
        step_scale=SAPG_STEP_SCALE,
        step_offset=SAPG_STEP_OFFSET,
        step_power=SAPG_STEP_POWER,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=1,
        mcmc_warmup=0,
        lambda_my=lambda_ref,
        step_size_myula=delta_ref,
        c_delta=K10_REFERENCE_C_DELTA,
        store_state_path=False,
    )
    theta_hat = float(sapg_result.theta_hat)

    map_result = fista_map(
        y,
        X,
        SIGMA2,
        theta_hat,
        a=AFFINE_A,
        b=AFFINE_B,
        tau_c=tau_c,
        beta0=beta_ols,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    beta_map = np.asarray(map_result.beta, dtype=float)

    smooth_ref = smoothed_map(
        y,
        X,
        SIGMA2,
        theta_hat,
        lambda_ref,
        a=AFFINE_A,
        b=AFFINE_B,
        tau_c=tau_c,
        beta0=beta_map,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    smooth_pre = smoothed_map(
        y,
        X,
        SIGMA2,
        theta_hat,
        lambda_pre,
        a=AFFINE_A,
        b=AFFINE_B,
        tau_c=tau_c,
        beta0=beta_map,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )

    rng_ref = np.random.default_rng(K10_REFERENCE_POST_SEED_BASE + rep)
    innovations_ref = rng_ref.normal(size=(POST_BURN + POST_KEEP, P))
    samples_ref = run_scalar_myula(
        X,
        y,
        theta_hat,
        tau_c,
        lambda_ref,
        delta_ref,
        beta_map,
        innovations_ref,
    )

    n_pre = K10_PRECONDITIONED_BURN + K10_PRECONDITIONED_KEEP * K10_PRECONDITIONED_THIN
    rng_pre = np.random.default_rng(K10_PRECONDITIONED_POST_SEED_BASE + rep)
    innovations_pre = rng_pre.normal(size=(n_pre, P))
    samples_pre = run_preconditioned_myula(
        X,
        y,
        theta_hat,
        tau_c,
        lambda_pre,
        delta_pre,
        gamma,
        beta_map,
        innovations_pre,
        burn=K10_PRECONDITIONED_BURN,
        keep=K10_PRECONDITIONED_KEEP,
        thin=K10_PRECONDITIONED_THIN,
    )

    summaries = []
    gate_rows = []

    for branch, samples, lambda_my, delta, c_delta, smooth in (
        (
            "scalar",
            samples_ref,
            lambda_ref,
            delta_ref,
            K10_REFERENCE_C_DELTA,
            smooth_ref,
        ),
        (
            "preconditioned",
            samples_pre,
            lambda_pre,
            delta_pre,
            K10_PRECONDITIONED_C_DELTA,
            smooth_pre,
        ),
    ):
        post = posterior_diagnostics(samples, geom["u_weak"], geom["u_strong"])
        gates = evaluate_gate_grid(
            rep,
            branch,
            KAPPA10,
            X,
            beta_map,
            samples,
        )
        nominal = gates["nominal"]
        gate_rows.extend(gates["rows"])

        summaries.append(
            {
                "rep": rep,
                "branch": branch,
                "kappa_c": KAPPA10,
                "theta0": theta0,
                "theta_hat": theta_hat,
                "tau_c": tau_c,
                "L0": geom["L0"],
                "Lf": Lf,
                "L_perp": L_perp,
                "lambda_my": lambda_my,
                "c_delta": c_delta,
                "delta": delta,
                "gamma": 1.0 if branch == "scalar" else gamma,
                "smooth_map_shift_l2": float(np.linalg.norm(smooth.beta - beta_map)),
                "map_rel_l2": relative_l2(beta_map),
                "postmean_rel_l2": relative_l2(post["posterior_mean"]),
                "gated_map_rel_l2": nominal["gated_map_rel_l2"],
                "gated_postmean_rel_l2": nominal["gated_postmean_rel_l2"],
                "mean_abs_budget_residual": post["mean_abs_budget_residual"],
                "median_posterior_sd": post["median_sd"],
                "median_coordinate_ess": post["median_coordinate_ess"],
                "min_coordinate_ess": post["min_coordinate_ess"],
                "ess_perp_weak": post["ess_perp_weak"],
                "ess_perp_strong": post["ess_perp_strong"],
                "ess_parallel": post["ess_parallel"],
                "nominal_size": nominal["size"],
                "nominal_tpr": nominal["tpr"],
                "nominal_fdp": nominal["fdp"],
                "nominal_exact": nominal["exact"],
                "nominal_D_min_map_eligible": gates["D_eligible"],
            }
        )

    geom_row = {
        "rep": rep,
        "Lf_over_Lperp": Lf / L_perp,
        "top_alignment_sq": top_alignment_sq,
        "gamma": gamma,
        "Lf_metric_over_Lperp": Lf_metric / L_perp,
        "lambda_pre_over_lambda_ref": lambda_pre / lambda_ref,
    }

    return summaries, gate_rows, geom_row


# ============================================================
# Reported timestep audit
# ============================================================

def aggregate_normals(xi_fine: np.ndarray, block_size: int) -> np.ndarray:
    if block_size == 1:
        return xi_fine.copy()

    n, p = xi_fine.shape
    if n % block_size != 0:
        raise ValueError("Fine innovation length is not divisible by block size.")

    return (
        xi_fine.reshape(n // block_size, block_size, p).sum(axis=1)
        / np.sqrt(block_size)
    )


def run_timestep_audit() -> pd.DataFrame:
    """Reproduce the matched-time kappa_c=10 timestep audit in the supplement."""
    X, y = generate_dataset(AUDIT_DATA_SEED)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    theta0 = float(P / np.sum(np.abs(beta_ols)))

    geom = geometry(X, KAPPA10)
    tau_c = float(geom["tau_c"])
    Lf = float(geom["Lf"])
    L_perp = float(geom["L_perp"])

    lambda_ref = 1.0 / Lf
    delta_ref = 0.9 / (Lf + 1.0 / lambda_ref)

    rng_sapg = np.random.default_rng(AUDIT_SAPG_SEED)
    sapg_result = sapg_laplace(
        y,
        X,
        SIGMA2,
        rng=rng_sapg,
        n_iter=SAPG_N_ITER,
        theta_init=theta0,
        beta0=beta_ols,
        intrinsic_dim=None,
        constraint="soft_affine",
        a=AFFINE_A,
        b=AFFINE_B,
        tau_c=tau_c,
        theta_bounds=None,
        bound_factor=SAPG_BOUND_FACTOR,
        step_scale=SAPG_STEP_SCALE,
        step_offset=SAPG_STEP_OFFSET,
        step_power=SAPG_STEP_POWER,
        average_start=SAPG_AVERAGE_START,
        mcmc_steps=1,
        mcmc_warmup=0,
        lambda_my=lambda_ref,
        step_size_myula=delta_ref,
        c_delta=0.9,
        store_state_path=False,
    )
    theta_hat = float(sapg_result.theta_hat)

    map_result = fista_map(
        y,
        X,
        SIGMA2,
        theta_hat,
        a=AFFINE_A,
        b=AFFINE_B,
        tau_c=tau_c,
        beta0=beta_ols,
        tol=MAP_TOL,
        max_iter=MAP_MAX_ITER,
        min_iter=5,
    )
    beta_map = np.asarray(map_result.beta, dtype=float)

    lambda_pre = 1.0 / L_perp
    gamma = L_perp / Lf
    M_half = P_PERP + np.sqrt(gamma) * P_PARALLEL
    Lf_metric = float(np.linalg.eigvalsh(M_half @ geom["Hf"] @ M_half)[-1])
    Ltotal = Lf_metric + 1.0 / lambda_pre

    base_burn = 4_000
    keep = 4_000
    max_factor = 4
    n_fine = (base_burn + keep) * max_factor

    rng = np.random.default_rng(AUDIT_BROWNIAN_SEED)
    xi_fine = rng.normal(size=(n_fine, P))

    rows = []
    for c_delta, factor, block_size in (
        (0.900, 1, 4),
        (0.450, 2, 2),
        (0.225, 4, 1),
    ):
        innovations = aggregate_normals(xi_fine, block_size)
        delta = c_delta / Ltotal

        samples = run_preconditioned_myula(
            X,
            y,
            theta_hat,
            tau_c,
            lambda_pre,
            delta,
            gamma,
            beta_map,
            innovations,
            burn=base_burn * factor,
            keep=keep,
            thin=factor,
        )

        post = posterior_diagnostics(samples, geom["u_weak"], geom["u_strong"])
        threshold, _ = posterior_scale_threshold(samples, K_REFERENCE)
        activation = activation_probabilities(samples, threshold)
        selected = (np.abs(beta_map) >= threshold) & (activation >= PI_REFERENCE)
        metrics = support_metrics(selected)

        rows.append(
            {
                "c_delta": c_delta,
                "median_ess": post["median_coordinate_ess"],
                "weak_tangent_ess": post["ess_perp_weak"],
                "median_sd": post["median_sd"],
                "mean_abs_budget_residual": post["mean_abs_budget_residual"],
                "mean_l1": post["mean_l1"],
                "selected_size": metrics["size"],
                "tpr": metrics["tpr"],
                "fdp": metrics["fdp"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Summaries and figures
# ============================================================

def aggregate_reference_summary(
    common: pd.DataFrame,
    k10: pd.DataFrame,
) -> pd.DataFrame:
    """Build the four-row Study 2 reference summary."""
    final = pd.concat(
        [
            common,
            k10.loc[k10["branch"].eq("preconditioned")],
        ],
        ignore_index=True,
    )

    return (
        final.groupby("kappa_c", as_index=False)
        .agg(
            theta_hat=("theta_hat", "mean"),
            map_rel_l2=("map_rel_l2", "mean"),
            postmean_rel_l2=("postmean_rel_l2", "mean"),
            gated_map_rel_l2=("gated_map_rel_l2", "mean"),
            gated_postmean_rel_l2=("gated_postmean_rel_l2", "mean"),
            mean_abs_budget=("mean_abs_budget_residual", "mean"),
            median_ess=("median_coordinate_ess", "median"),
            weak_tangent_ess=("ess_perp_weak", "median"),
            mean_size=("nominal_size", "mean"),
            mean_tpr=("nominal_tpr", "mean"),
            mean_fdp=("nominal_fdp", "mean"),
            exact_rate=("nominal_exact", "mean"),
            median_D_eligible=("nominal_D_min_map_eligible", "median"),
        )
        .sort_values("kappa_c")
        .reset_index(drop=True)
    )


def aggregate_gate_grid(
    common_gate: pd.DataFrame,
    k10_gate: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the full decision grid using the retained kappa_c=10 branch."""
    final_gate = pd.concat(
        [
            common_gate,
            k10_gate.loc[k10_gate["branch"].eq("preconditioned")],
        ],
        ignore_index=True,
    )

    return (
        final_gate.groupby(["kappa_c", "k_gate", "pi_star"], as_index=False)
        .agg(
            mean_size=("size", "mean"),
            mean_tpr=("tpr", "mean"),
            mean_fdp=("fdp", "mean"),
            exact_rate=("exact", "mean"),
            gated_map_rel_l2=("gated_map_rel_l2", "mean"),
            gated_postmean_rel_l2=("gated_postmean_rel_l2", "mean"),
        )
        .sort_values(["kappa_c", "k_gate", "pi_star"])
        .reset_index(drop=True)
    )


def print_k10_comparison(k10: pd.DataFrame, geom: pd.DataFrame) -> None:
    comparison = (
        k10.groupby("branch", as_index=False)
        .agg(
            median_sd=("median_posterior_sd", "median"),
            median_ess=("median_coordinate_ess", "median"),
            min_ess=("min_coordinate_ess", "median"),
            weak_tangent_ess=("ess_perp_weak", "median"),
            sum_direction_ess=("ess_parallel", "median"),
            mean_abs_budget=("mean_abs_budget_residual", "mean"),
            mean_size=("nominal_size", "mean"),
            mean_tpr=("nominal_tpr", "mean"),
            mean_fdp=("nominal_fdp", "mean"),
            median_D_eligible=("nominal_D_min_map_eligible", "median"),
        )
    )

    print("\nkappa_c=10 scalar vs preconditioned posterior calculations")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.5g}"))

    print("\nkappa_c=10 geometry")
    print(
        f"mean Lf/L_perp              : {geom['Lf_over_Lperp'].mean():.4f}\n"
        f"mean top alignment squared  : {geom['top_alignment_sq'].mean():.6f}\n"
        f"mean gamma                  : {geom['gamma'].mean():.6f}\n"
        f"mean Lf,M/L_perp            : {geom['Lf_metric_over_Lperp'].mean():.5f}"
    )


def make_figures(common: pd.DataFrame, k10: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Supplementary two-panel figure: affine adherence and ESS.
    # --------------------------------------------------------
    adherence_rows = []
    mixing_rows = []

    for kappa in COMMON_KAPPAS:
        values = common.loc[np.isclose(common["kappa_c"], kappa)]
        residual = values["mean_abs_budget_residual"].to_numpy(float)
        ess = values["median_coordinate_ess"].to_numpy(float)

        adherence_rows.append(
            (
                kappa,
                residual.mean(),
                1.96 * residual.std(ddof=1) / np.sqrt(residual.size),
            )
        )
        mixing_rows.append(
            (
                kappa,
                np.median(ess),
                np.quantile(ess, 0.25),
                np.quantile(ess, 0.75),
            )
        )

    baseline10 = k10.loc[k10["branch"].eq("scalar")]
    retained10 = k10.loc[k10["branch"].eq("preconditioned")]

    for frame, target in ((baseline10, "baseline"), (retained10, "retained")):
        residual = frame["mean_abs_budget_residual"].to_numpy(float)
        ess = frame["median_coordinate_ess"].to_numpy(float)

        row_a = (
            10.0,
            residual.mean(),
            1.96 * residual.std(ddof=1) / np.sqrt(residual.size),
        )
        row_m = (
            10.0,
            np.median(ess),
            np.quantile(ess, 0.25),
            np.quantile(ess, 0.75),
        )

        if target == "baseline":
            adherence_rows.append(row_a)
            mixing_rows.append(row_m)
        else:
            retained_adherence = row_a
            retained_mixing = row_m

    ordinary_adherence = np.asarray(adherence_rows, dtype=float)
    ordinary_mixing = np.asarray(mixing_rows, dtype=float)
    xpos = np.arange(4)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(
        xpos,
        ordinary_adherence[:, 1],
        yerr=ordinary_adherence[:, 2],
        marker="o",
        capsize=4,
        label="Ordinary scalar-step",
    )
    ax.errorbar(
        [xpos[-1]],
        [retained_adherence[1]],
        yerr=[[retained_adherence[2]], [retained_adherence[2]]],
        marker="s",
        markersize=7,
        capsize=4,
        linestyle="none",
        label=r"Geometry-aware, $\kappa_c=10$",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["0", "0.1", "1", "10"])
    ax.set_xlabel(r"Affine-curvature ratio $\kappa_c$")
    ax.set_ylabel(r"Mean posterior $|\mathbf{1}^{\top}\beta - 1|$")
    ax.set_title("Affine adherence")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    y = ordinary_mixing[:, 1]
    yerr = np.vstack((y - ordinary_mixing[:, 2], ordinary_mixing[:, 3] - y))
    ax.errorbar(
        xpos,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
        label="Ordinary scalar-step",
    )
    retained_y = retained_mixing[1]
    retained_yerr = np.array(
        [
            [retained_y - retained_mixing[2]],
            [retained_mixing[3] - retained_y],
        ]
    )
    ax.errorbar(
        [xpos[-1]],
        [retained_y],
        yerr=retained_yerr,
        marker="s",
        markersize=7,
        capsize=4,
        linestyle="none",
        label=r"Geometry-aware, $\kappa_c=10$",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["0", "0.1", "1", "10"])
    ax.set_xlabel(r"Affine-curvature ratio $\kappa_c$")
    ax.set_ylabel("Median coordinate ESS")
    ax.set_title("Low-rank stiffness and mixing recovery")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"fig_study2_affine_and_mixing.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    # --------------------------------------------------------
    # Main-paper directional ESS figure at kappa_c=10.
    # --------------------------------------------------------
    directions = (
        ("Weak tangent", "ess_perp_weak"),
        ("Median coordinate ESS", "median_coordinate_ess"),
        ("Strong tangent", "ess_perp_strong"),
        ("Sum direction", "ess_parallel"),
    )
    positions = {
        "Weak tangent": (1.0, 1.35),
        "Median coordinate ESS": (2.2, 2.55),
        "Strong tangent": (3.4, 3.75),
        "Sum direction": (4.9, 5.25),
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for label, column in directions:
        for i, branch in enumerate(("scalar", "preconditioned")):
            values = k10.loc[k10["branch"].eq(branch), column].to_numpy(float)
            colour = "tab:blue" if branch == "scalar" else "tab:orange"
            bp = ax.boxplot(
                values,
                positions=[positions[label][i]],
                widths=0.24,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            bp["boxes"][0].set_facecolor(colour)
            bp["boxes"][0].set_alpha(0.65)

    centres = [np.mean(positions[label]) for label, _ in directions]
    ax.set_xticks(centres)
    ax.set_xticklabels([label for label, _ in directions])
    ax.set_ylabel("Effective sample size")
    ax.set_yscale("log")
    ax.set_title("Directional ESS: baseline vs geometry-aware")
    ax.axvline(4.35, linestyle="--", linewidth=0.9, alpha=0.5)

    # The legend is built from invisible boxplot-style rectangles.
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor="tab:blue",
                alpha=0.65,
                label="Without preconditioning",
            ),
            Patch(
                facecolor="tab:orange",
                alpha=0.65,
                label="With preconditioning",
            ),
        ],
        loc="upper left",
        frameon=True,
    )

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"fig_study2_directional_ess.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# ============================================================
# Main study
# ============================================================

def main() -> None:
    start = time.perf_counter()

    common_rows: list[dict] = []
    common_gate_rows: list[dict] = []
    k10_rows: list[dict] = []
    k10_gate_rows: list[dict] = []
    k10_geometry_rows: list[dict] = []

    print("Study 2: soft affine information")
    print("=" * 44)
    print(f"R={R}, n={N}, p={P}, sigma={SIGMA:g}")
    print(f"kappa_c grid: {KAPPA_GRID}")

    for rep in range(R):
        rep_start = time.perf_counter()

        X, y = generate_dataset(DATA_SEED_BASE + rep)
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        theta0 = float(P / max(np.sum(np.abs(beta_ols)), 1e-12))

        rng_post = np.random.default_rng(POST_SEED_BASE + rep)
        common_innovations = rng_post.normal(size=(POST_BURN + POST_KEEP, P))

        for kappa_c in COMMON_KAPPAS:
            summary, gates = fit_common_branch(
                rep,
                X,
                y,
                beta_ols,
                theta0,
                kappa_c,
                common_innovations,
            )
            common_rows.append(summary)
            common_gate_rows.extend(gates)

        summaries10, gates10, geom10 = fit_kappa10(
            rep,
            X,
            y,
            beta_ols,
            theta0,
        )
        k10_rows.extend(summaries10)
        k10_gate_rows.extend(gates10)
        k10_geometry_rows.append(geom10)

        print(
            f"rep={rep:03d} complete in {time.perf_counter() - rep_start:.2f}s"
        )

    common = pd.DataFrame(common_rows)
    common_gate = pd.DataFrame(common_gate_rows)
    k10 = pd.DataFrame(k10_rows)
    k10_gate = pd.DataFrame(k10_gate_rows)
    k10_geom = pd.DataFrame(k10_geometry_rows)

    reference = aggregate_reference_summary(common, k10)
    gate_grid = aggregate_gate_grid(common_gate, k10_gate)

    print("\nReference decision summary")
    print(reference.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print_k10_comparison(k10, k10_geom)

    print("\nComplete decision grid")
    print(gate_grid.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\nReported timestep audit")
    audit = run_timestep_audit()
    print(audit.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    make_figures(common, k10)

    elapsed = time.perf_counter() - start
    print(f"\nTotal runtime: {elapsed / 60.0:.2f} minutes")
    print(f"Figures written to: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
