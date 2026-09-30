
"""
fista.py

MAP solvers for the computational-statistics sparse-regression
pipeline.

Primary inferential point estimate
----------------------------------
The reported MAP is the MAP of the ORIGINAL, unsmoothed
posterior,

    U(beta) = f(beta) + theta ||beta||_1,

computed using FISTA.

Smoothed-MAP diagnostic
-----------------------
We also compute the mode of

    U_lambda(beta)
      = f(beta) + (theta ||.||_1)^lambda(beta),

because MYULA samples from an approximation based on this
smoothed potential.

The smoothed MAP is a numerical diagnostic; it is not the
primary reported estimator.

Hard sum-zero model
-------------------
For

    C = {beta : 1^T beta = 0},

the algorithms operate directly in C. The l1 proximal step is

    prox^C_{t ||.||_1},

implemented in prox.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prox import (
    soft_threshold,
    project_sum_zero,
    prox_l1_sum_zero,
)

from model import (
    gaussian_nll,
    gaussian_grad,
    soft_affine_nll,
    soft_affine_grad,
    smooth_lipschitz,
    sum_zero_lipschitz,
    laplace_penalty,
    laplace_moreau_value,
    laplace_moreau_grad,
    posterior_objective,
    smoothed_posterior_objective,
)


__all__ = [
    "MapResult",
    "fista_map",
    "fista_map_sum_zero",
    "smoothed_map",
    "smoothed_map_sum_zero",
]


@dataclass(frozen=True)
class MapResult:
    beta: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    lipschitz: float
    step_size: float
    stationarity: float
    objective_path: np.ndarray
    stationarity_path: np.ndarray
    lambda_my: float | None
    constraint: str


# ============================================================
# Validation helpers
# ============================================================

def _validate_common(y, X, sigma2, theta, beta0):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if X.shape[0] != y.size:
        raise ValueError(
            "The number of rows of X must equal len(y)."
        )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have non-zero dimensions.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")

    sigma2 = float(sigma2)
    theta = float(theta)

    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError(
            "sigma2 must be finite and strictly positive."
        )

    if not np.isfinite(theta) or theta < 0.0:
        raise ValueError(
            "theta must be finite and nonnegative."
        )

    p = X.shape[1]

    if beta0 is None:
        beta0 = np.zeros(p, dtype=float)
    else:
        beta0 = np.asarray(beta0, dtype=float)

        if beta0.ndim != 1 or beta0.size != p:
            raise ValueError(
                "beta0 must be one-dimensional with length p."
            )

        if not np.all(np.isfinite(beta0)):
            raise ValueError(
                "beta0 contains non-finite values."
            )

        beta0 = beta0.copy()

    return y, X, sigma2, theta, beta0


def _validate_solver_options(tol, max_iter, min_iter):
    tol = float(tol)

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            "tol must be finite and strictly positive."
        )

    if (
        not isinstance(max_iter, (int, np.integer))
        or max_iter <= 0
    ):
        raise ValueError(
            "max_iter must be a positive integer."
        )

    if (
        not isinstance(min_iter, (int, np.integer))
        or min_iter < 0
    ):
        raise ValueError(
            "min_iter must be a nonnegative integer."
        )

    if min_iter > max_iter:
        raise ValueError(
            "min_iter cannot exceed max_iter."
        )

    return tol


def _parse_affine(a, b, tau_c):
    no_affine = (
        a is None
        and b is None
        and tau_c is None
    )

    if no_affine:
        return False

    if a is None or b is None or tau_c is None:
        raise ValueError(
            "For a soft affine model, a, b, and tau_c "
            "must all be supplied."
        )

    return True


# ============================================================
# Smooth likelihood helpers
# ============================================================

def _smooth_value_grad(
    beta,
    y,
    X,
    sigma2,
    *,
    a=None,
    b=None,
    tau_c=None,
):
    use_affine = _parse_affine(a, b, tau_c)

    if not use_affine:
        value = gaussian_nll(
            beta, y, X, sigma2
        )

        grad = gaussian_grad(
            beta, y, X, sigma2
        )

    else:
        value = soft_affine_nll(
            beta,
            y,
            X,
            sigma2,
            a,
            b,
            tau_c,
        )

        grad = soft_affine_grad(
            beta,
            y,
            X,
            sigma2,
            a,
            b,
            tau_c,
        )

    return value, grad


# ============================================================
# Unsmoothed MAP: unconstrained / soft affine
# ============================================================

def fista_map(
    y,
    X,
    sigma2,
    theta,
    *,
    a=None,
    b=None,
    tau_c=None,
    beta0=None,
    tol=1e-8,
    max_iter=20000,
    min_iter=5,
):
    r"""
    FISTA for the unsmoothed MAP

        min_beta f(beta) + theta ||beta||_1.

    f may be either the Gaussian negative log-likelihood or the
    Gaussian likelihood augmented with a soft affine
    pseudo-observation.

    The step size is

        gamma = 1 / L_f.

    Convergence is monitored using the proximal-gradient mapping

        G_gamma(beta)
          =
          [beta
           - prox_{gamma theta ||.||_1}
             (beta - gamma grad f(beta))]
          / gamma.

    Returns
    -------
    MapResult
    """
    y, X, sigma2, theta, beta = _validate_common(
        y, X, sigma2, theta, beta0
    )

    tol = _validate_solver_options(
        tol, max_iter, min_iter
    )

    use_affine = _parse_affine(
        a, b, tau_c
    )

    if use_affine:
        Lf = smooth_lipschitz(
            X,
            sigma2,
            a=a,
            tau_c=tau_c,
        )

        constraint_name = "soft_affine"

    else:
        Lf = smooth_lipschitz(
            X,
            sigma2,
        )

        constraint_name = "none"

    if not np.isfinite(Lf) or Lf <= 0.0:
        raise ValueError(
            "The smooth Lipschitz constant must be positive."
        )

    step = 1.0 / Lf

    x = beta.copy()
    z = beta.copy()
    t = 1.0

    objective_path = []
    stationarity_path = []

    def objective(v):
        return posterior_objective(
            v,
            y,
            X,
            sigma2,
            theta,
            a=a,
            b=b,
            tau_c=tau_c,
        )

    def gradient(v):
        return _smooth_value_grad(
            v,
            y,
            X,
            sigma2,
            a=a,
            b=b,
            tau_c=tau_c,
        )[1]

    def stationarity(v):
        grad_v = gradient(v)

        prox_v = soft_threshold(
            v - step * grad_v,
            step * theta,
        )

        mapping = (v - prox_v) / step

        return float(
            np.linalg.norm(mapping)
            / max(1.0, np.linalg.norm(v))
        )

    objective_path.append(
        float(objective(x))
    )

    stationarity_path.append(
        stationarity(x)
    )

    converged = (
        stationarity_path[-1] <= tol
        and min_iter == 0
    )

    n_iter = 0

    for k in range(1, max_iter + 1):
        grad_z = gradient(z)

        x_new = soft_threshold(
            z - step * grad_z,
            step * theta,
        )

        t_new = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * t**2)
        )

        z_new = (
            x_new
            + ((t - 1.0) / t_new)
            * (x_new - x)
        )

        obj_new = float(
            objective(x_new)
        )

        stat_new = stationarity(
            x_new
        )

        objective_path.append(obj_new)
        stationarity_path.append(stat_new)

        x = x_new
        z = z_new
        t = t_new
        n_iter = k

        if k >= min_iter and stat_new <= tol:
            converged = True
            break

    return MapResult(
        beta=np.asarray(x, dtype=float),
        objective=float(objective_path[-1]),
        n_iter=int(n_iter),
        converged=bool(converged),
        lipschitz=float(Lf),
        step_size=float(step),
        stationarity=float(
            stationarity_path[-1]
        ),
        objective_path=np.asarray(
            objective_path,
            dtype=float,
        ),
        stationarity_path=np.asarray(
            stationarity_path,
            dtype=float,
        ),
        lambda_my=None,
        constraint=constraint_name,
    )


# ============================================================
# Unsmoothed MAP: hard sum-zero
# ============================================================

def fista_map_sum_zero(
    y,
    X,
    sigma2,
    theta,
    *,
    beta0=None,
    tol=1e-8,
    max_iter=20000,
    min_iter=5,
):
    r"""
    FISTA for

        min_beta
            ||y-X beta||^2/(2 sigma2)
            + theta ||beta||_1

        subject to
            1^T beta = 0.

    The algorithm operates directly in the sum-zero subspace.

    The smooth gradient is restricted using

        P_C grad f,

    and the backward step uses

        prox^C_{gamma theta ||.||_1}.
    """
    y, X, sigma2, theta, beta = _validate_common(
        y, X, sigma2, theta, beta0
    )

    tol = _validate_solver_options(
        tol, max_iter, min_iter
    )

    beta = project_sum_zero(beta)

    Lf = sum_zero_lipschitz(
        X,
        sigma2,
    )

    if not np.isfinite(Lf) or Lf <= 0.0:
        raise ValueError(
            "The restricted Lipschitz constant must be positive."
        )

    step = 1.0 / Lf

    x = beta.copy()
    z = beta.copy()
    t = 1.0

    objective_path = []
    stationarity_path = []

    def objective(v):
        return (
            gaussian_nll(
                v,
                y,
                X,
                sigma2,
            )
            + laplace_penalty(
                v,
                theta,
            )
        )

    def projected_gradient(v):
        return project_sum_zero(
            gaussian_grad(
                v,
                y,
                X,
                sigma2,
            )
        )

    def stationarity(v):
        grad_v = projected_gradient(v)

        prox_v = prox_l1_sum_zero(
            v - step * grad_v,
            step * theta,
        )

        mapping = (
            v - prox_v
        ) / step

        return float(
            np.linalg.norm(mapping)
            / max(1.0, np.linalg.norm(v))
        )

    objective_path.append(
        float(objective(x))
    )

    stationarity_path.append(
        stationarity(x)
    )

    converged = (
        stationarity_path[-1] <= tol
        and min_iter == 0
    )

    n_iter = 0

    for k in range(1, max_iter + 1):
        grad_z = projected_gradient(z)

        x_new = prox_l1_sum_zero(
            z - step * grad_z,
            step * theta,
        )

        # Clean only floating-point-level drift.
        if abs(np.sum(x_new)) > 1e-10:
            raise RuntimeError(
                "Hard sum-zero FISTA left the constraint "
                "subspace unexpectedly."
            )

        t_new = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * t**2)
        )

        z_new = (
            x_new
            + ((t - 1.0) / t_new)
            * (x_new - x)
        )

        # Linear combinations of feasible vectors should remain
        # feasible; this removes only floating-point drift.
        z_new = project_sum_zero(
            z_new
        )

        obj_new = float(
            objective(x_new)
        )

        stat_new = stationarity(
            x_new
        )

        objective_path.append(obj_new)
        stationarity_path.append(stat_new)

        x = x_new
        z = z_new
        t = t_new
        n_iter = k

        if k >= min_iter and stat_new <= tol:
            converged = True
            break

    return MapResult(
        beta=np.asarray(x, dtype=float),
        objective=float(objective_path[-1]),
        n_iter=int(n_iter),
        converged=bool(converged),
        lipschitz=float(Lf),
        step_size=float(step),
        stationarity=float(
            stationarity_path[-1]
        ),
        objective_path=np.asarray(
            objective_path,
            dtype=float,
        ),
        stationarity_path=np.asarray(
            stationarity_path,
            dtype=float,
        ),
        lambda_my=None,
        constraint="sum_zero",
    )


# ============================================================
# Smoothed posterior MAP diagnostic:
# unconstrained / soft affine
# ============================================================

def smoothed_map(
    y,
    X,
    sigma2,
    theta,
    lambda_my,
    *,
    a=None,
    b=None,
    tau_c=None,
    beta0=None,
    tol=1e-8,
    max_iter=20000,
    min_iter=5,
):
    r"""
    Mode of the Moreau-smoothed posterior

        min_beta
            f(beta)
            + (theta ||.||_1)^lambda(beta).

    This is a DIAGNOSTIC, not the primary reported MAP.

    Since the objective is smooth, an accelerated gradient
    iteration is used with the conservative curvature bound

        L_total = L_f + 1/lambda_my.
    """
    y, X, sigma2, theta, beta = _validate_common(
        y, X, sigma2, theta, beta0
    )

    tol = _validate_solver_options(
        tol, max_iter, min_iter
    )

    lambda_my = float(lambda_my)

    if (
        not np.isfinite(lambda_my)
        or lambda_my <= 0.0
    ):
        raise ValueError(
            "lambda_my must be finite and strictly positive."
        )

    use_affine = _parse_affine(
        a, b, tau_c
    )

    if use_affine:
        Lf = smooth_lipschitz(
            X,
            sigma2,
            a=a,
            tau_c=tau_c,
        )

        constraint_name = "soft_affine"

    else:
        Lf = smooth_lipschitz(
            X,
            sigma2,
        )

        constraint_name = "none"

    Ltotal = Lf + 1.0 / lambda_my
    step = 1.0 / Ltotal

    x = beta.copy()
    z = beta.copy()
    t = 1.0

    objective_path = []
    stationarity_path = []

    def objective(v):
        return smoothed_posterior_objective(
            v,
            y,
            X,
            sigma2,
            theta,
            lambda_my,
            a=a,
            b=b,
            tau_c=tau_c,
        )

    def total_gradient(v):
        grad_f = _smooth_value_grad(
            v,
            y,
            X,
            sigma2,
            a=a,
            b=b,
            tau_c=tau_c,
        )[1]

        grad_h = laplace_moreau_grad(
            v,
            theta,
            lambda_my,
        )

        return grad_f + grad_h

    def stationarity(v):
        grad = total_gradient(v)

        return float(
            np.linalg.norm(grad)
            / max(1.0, np.linalg.norm(v))
        )

    objective_path.append(
        float(objective(x))
    )

    stationarity_path.append(
        stationarity(x)
    )

    converged = (
        stationarity_path[-1] <= tol
        and min_iter == 0
    )

    n_iter = 0

    for k in range(1, max_iter + 1):
        grad_z = total_gradient(z)

        x_new = z - step * grad_z

        t_new = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * t**2)
        )

        z_new = (
            x_new
            + ((t - 1.0) / t_new)
            * (x_new - x)
        )

        obj_new = float(
            objective(x_new)
        )

        stat_new = stationarity(
            x_new
        )

        objective_path.append(obj_new)
        stationarity_path.append(stat_new)

        x = x_new
        z = z_new
        t = t_new
        n_iter = k

        if k >= min_iter and stat_new <= tol:
            converged = True
            break

    return MapResult(
        beta=np.asarray(x, dtype=float),
        objective=float(objective_path[-1]),
        n_iter=int(n_iter),
        converged=bool(converged),
        lipschitz=float(Ltotal),
        step_size=float(step),
        stationarity=float(
            stationarity_path[-1]
        ),
        objective_path=np.asarray(
            objective_path,
            dtype=float,
        ),
        stationarity_path=np.asarray(
            stationarity_path,
            dtype=float,
        ),
        lambda_my=float(lambda_my),
        constraint=constraint_name,
    )


# ============================================================
# Smoothed posterior MAP diagnostic:
# hard sum-zero
# ============================================================

def smoothed_map_sum_zero(
    y,
    X,
    sigma2,
    theta,
    lambda_my,
    *,
    beta0=None,
    tol=1e-8,
    max_iter=20000,
    min_iter=5,
):
    r"""
    Mode of the Moreau-smoothed posterior restricted to

        C = {beta : 1^T beta = 0}.

    The Moreau envelope is taken IN THE CONSTRAINT SUBSPACE:

        h_C^lambda(beta)
          =
          min_{u in C}
          theta ||u||_1
          + ||u-beta||^2/(2 lambda).

    Hence

        grad h_C^lambda(beta)
          =
          [beta
           - prox^C_{lambda theta ||.||_1}(beta)]
          / lambda.

    This is a diagnostic mode corresponding to the geometry
    that will later be used by hard-constrained MYULA.
    """
    y, X, sigma2, theta, beta = _validate_common(
        y, X, sigma2, theta, beta0
    )

    tol = _validate_solver_options(
        tol, max_iter, min_iter
    )

    lambda_my = float(lambda_my)

    if (
        not np.isfinite(lambda_my)
        or lambda_my <= 0.0
    ):
        raise ValueError(
            "lambda_my must be finite and strictly positive."
        )

    beta = project_sum_zero(
        beta
    )

    Lf = sum_zero_lipschitz(
        X,
        sigma2,
    )

    Ltotal = Lf + 1.0 / lambda_my
    step = 1.0 / Ltotal

    x = beta.copy()
    z = beta.copy()
    t = 1.0

    objective_path = []
    stationarity_path = []

    def constrained_moreau(v):
        prox_v = prox_l1_sum_zero(
            v,
            lambda_my * theta,
        )

        diff = prox_v - v

        value = (
            theta
            * np.sum(np.abs(prox_v))
            + 0.5
            * (diff @ diff)
            / lambda_my
        )

        grad = (
            v - prox_v
        ) / lambda_my

        return float(value), grad

    def objective(v):
        h_value, _ = constrained_moreau(v)

        return (
            gaussian_nll(
                v,
                y,
                X,
                sigma2,
            )
            + h_value
        )

    def total_gradient(v):
        grad_f = project_sum_zero(
            gaussian_grad(
                v,
                y,
                X,
                sigma2,
            )
        )

        _, grad_h = constrained_moreau(
            v
        )

        # Both terms are in C.
        return grad_f + grad_h

    def stationarity(v):
        grad = total_gradient(v)

        return float(
            np.linalg.norm(grad)
            / max(1.0, np.linalg.norm(v))
        )

    objective_path.append(
        float(objective(x))
    )

    stationarity_path.append(
        stationarity(x)
    )

    converged = (
        stationarity_path[-1] <= tol
        and min_iter == 0
    )

    n_iter = 0

    for k in range(1, max_iter + 1):
        grad_z = total_gradient(z)

        x_new = z - step * grad_z

        # Floating-point cleanup only.
        x_new = project_sum_zero(
            x_new
        )

        t_new = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * t**2)
        )

        z_new = (
            x_new
            + ((t - 1.0) / t_new)
            * (x_new - x)
        )

        z_new = project_sum_zero(
            z_new
        )

        obj_new = float(
            objective(x_new)
        )

        stat_new = stationarity(
            x_new
        )

        objective_path.append(obj_new)
        stationarity_path.append(stat_new)

        x = x_new
        z = z_new
        t = t_new
        n_iter = k

        if k >= min_iter and stat_new <= tol:
            converged = True
            break

    return MapResult(
        beta=np.asarray(x, dtype=float),
        objective=float(objective_path[-1]),
        n_iter=int(n_iter),
        converged=bool(converged),
        lipschitz=float(Ltotal),
        step_size=float(step),
        stationarity=float(
            stationarity_path[-1]
        ),
        objective_path=np.asarray(
            objective_path,
            dtype=float,
        ),
        stationarity_path=np.asarray(
            stationarity_path,
            dtype=float,
        ),
        lambda_my=float(lambda_my),
        constraint="sum_zero",
    )
