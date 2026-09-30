
"""
sapg.py

Stochastic-approximation proximal-gradient (SAPG) empirical-Bayes
calibration for the Laplace scale parameter theta.

Statistical target
------------------
For

    pi(beta | theta) proportional to
        exp[-theta ||beta||_1],

the exact log-theta marginal-likelihood score is

    d/d eta log p(y | theta)
      =
      d - theta E[||beta||_1 | y, theta],

where

    eta = log(theta)

and d is the intrinsic dimension of the prior support.

Hence:
    - unconstrained model:        d = p,
    - soft affine constraint:     d = p,
    - hard homogeneous rank-r
      constraint:                 d = p-r.

For the hard sum-to-zero case,

    d = p-1.

MYULA role
----------
The score above belongs to the ORIGINAL nonsmoothed model.
MYULA is used only as a numerical approximation to the posterior
expectation appearing in that score.

Thus the implemented stochastic score is

    Delta_hat_k
      =
      d - theta_k * gbar_k,

where gbar_k is estimated from one or more MYULA transitions.

The SA recursion is

    eta_{k+1}
      =
      Projection[
        eta_k + rho_k Delta_hat_k
      ],

with

    rho_k
      =
      step_scale / (k + step_offset)^step_power.

The default step_power=1 satisfies the usual Robbins-Monro
summability conditions. Other powers in (1/2,1] are allowed.

Averaging
---------
Because the recursion is carried out in eta=log(theta), the
primary averaged estimate is

    eta_hat = mean(eta_k)

after `average_start`, and

    theta_hat = exp(eta_hat).

The arithmetic mean of the theta iterates is also returned.

Randomness
----------
A numpy.random.Generator must be supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from prox import (
    project_sum_zero,
    prox_l1_sum_zero,
)

from model import (
    gaussian_grad,
    soft_affine_grad,
    smooth_lipschitz,
    sum_zero_lipschitz,
    laplace_moreau_grad,
)


__all__ = [
    "SapgResult",
    "sapg_laplace",
]


@dataclass(frozen=True)
class SapgResult:
    theta_hat: float
    eta_hat: float
    theta_mean_arithmetic: float

    theta_final: float
    eta_final: float
    beta_final: np.ndarray

    theta_init: float
    intrinsic_dim: int

    n_iter: int
    average_start: int
    mcmc_steps: int
    mcmc_warmup: int
    n_mcmc_transitions: int

    lambda_my: float
    step_size_myula: float
    c_delta: float | None
    lipschitz_smooth: float
    lipschitz_total_bound: float

    step_scale: float
    step_offset: float
    step_power: float

    theta_bounds: tuple[float, float]

    theta_path: np.ndarray
    eta_path: np.ndarray
    score_path: np.ndarray
    g_path: np.ndarray
    rho_path: np.ndarray
    bound_hit_path: np.ndarray

    state_path: np.ndarray

    constraint: str


# ============================================================
# Input helpers
# ============================================================

def _validate_y_X(y, X):
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
        raise ValueError(
            "X must have non-zero dimensions."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y contains non-finite values."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains non-finite values."
        )

    return y, X


def _validate_positive_int(name, value, lower=1):
    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer."
        )

    if value < lower:
        raise ValueError(
            f"{name} must be >= {lower}."
        )

    return int(value)


def _validate_geometry(
    constraint,
    X,
    a,
    b,
    tau_c,
    intrinsic_dim,
):
    p = X.shape[1]

    if constraint not in {
        "none",
        "soft_affine",
        "sum_zero",
    }:
        raise ValueError(
            "constraint must be one of "
            "{'none', 'soft_affine', 'sum_zero'}."
        )

    if constraint == "none":
        if (
            a is not None
            or b is not None
            or tau_c is not None
        ):
            raise ValueError(
                "a, b and tau_c must be omitted when "
                "constraint='none'."
            )

        default_dim = p

    elif constraint == "soft_affine":
        if (
            a is None
            or b is None
            or tau_c is None
        ):
            raise ValueError(
                "a, b and tau_c are required when "
                "constraint='soft_affine'."
            )

        a = np.asarray(a, dtype=float)

        if a.ndim != 1 or a.size != p:
            raise ValueError(
                "a must be one-dimensional with length p."
            )

        if not np.all(np.isfinite(a)):
            raise ValueError(
                "a contains non-finite values."
            )

        b = float(b)
        tau_c = float(tau_c)

        if not np.isfinite(b):
            raise ValueError(
                "b must be finite."
            )

        if (
            not np.isfinite(tau_c)
            or tau_c <= 0.0
        ):
            raise ValueError(
                "tau_c must be finite and strictly positive."
            )

        default_dim = p

    else:
        # Hard homogeneous sum-zero model.
        if (
            a is not None
            or b is not None
            or tau_c is not None
        ):
            raise ValueError(
                "a, b and tau_c must be omitted when "
                "constraint='sum_zero'."
            )

        if p < 2:
            raise ValueError(
                "A sum-zero model requires p >= 2."
            )

        default_dim = p - 1

    if intrinsic_dim is None:
        intrinsic_dim = default_dim

    else:
        intrinsic_dim = _validate_positive_int(
            "intrinsic_dim",
            intrinsic_dim,
            lower=1,
        )

        if intrinsic_dim != default_dim:
            warnings.warn(
                "intrinsic_dim differs from the geometric "
                f"default ({default_dim}). This is allowed "
                "because deliberate dimension overrides are "
                "needed for diagnostic experiments, but it "
                "changes the SAPG score.",
                RuntimeWarning,
                stacklevel=3,
            )

    return (
        int(intrinsic_dim),
        int(default_dim),
        a,
        b,
        tau_c,
    )


def _resolve_theta_initialisation(
    intrinsic_dim,
    p,
    theta_init,
    beta_ref,
):
    if theta_init is not None:
        theta_init = float(theta_init)

        if (
            not np.isfinite(theta_init)
            or theta_init <= 0.0
        ):
            raise ValueError(
                "theta_init must be finite and strictly positive."
            )

        return theta_init

    if beta_ref is None:
        raise ValueError(
            "Supply either theta_init or beta_ref. "
            "If beta_ref is supplied, the moment initialisation "
            "theta_init = intrinsic_dim / ||beta_ref||_1 is used."
        )

    beta_ref = np.asarray(
        beta_ref,
        dtype=float,
    )

    if (
        beta_ref.ndim != 1
        or beta_ref.size != p
    ):
        raise ValueError(
            "beta_ref must be one-dimensional with length p."
        )

    if not np.all(np.isfinite(beta_ref)):
        raise ValueError(
            "beta_ref contains non-finite values."
        )

    g_ref = float(
        np.sum(np.abs(beta_ref))
    )

    scale = max(
        1.0,
        float(np.linalg.norm(beta_ref)),
    )

    if g_ref <= 100.0 * np.finfo(float).eps * scale:
        raise ValueError(
            "beta_ref has essentially zero l1 norm, so "
            "theta_init = d / ||beta_ref||_1 is not usable."
        )

    return float(
        intrinsic_dim / g_ref
    )


def _resolve_theta_bounds(
    theta_init,
    theta_bounds,
    bound_factor,
):
    if theta_bounds is None:
        bound_factor = float(
            bound_factor
        )

        if (
            not np.isfinite(bound_factor)
            or bound_factor <= 1.0
        ):
            raise ValueError(
                "bound_factor must be finite and greater than 1."
            )

        lower = theta_init / bound_factor
        upper = theta_init * bound_factor

    else:
        if len(theta_bounds) != 2:
            raise ValueError(
                "theta_bounds must contain exactly "
                "(lower, upper)."
            )

        lower = float(
            theta_bounds[0]
        )

        upper = float(
            theta_bounds[1]
        )

        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or upper <= lower
        ):
            raise ValueError(
                "theta_bounds must satisfy "
                "0 < lower < upper < infinity."
            )

    if not (
        lower <= theta_init <= upper
    ):
        raise ValueError(
            "theta_init must lie inside theta_bounds."
        )

    return (
        float(lower),
        float(upper),
    )


def _resolve_myula_scales(
    Lf,
    lambda_my,
    step_size_myula,
    c_delta,
):
    if (
        not np.isfinite(Lf)
        or Lf <= 0.0
    ):
        raise ValueError(
            "L_f must be finite and strictly positive."
        )

    if lambda_my is None:
        lambda_my = 1.0 / Lf
    else:
        lambda_my = float(
            lambda_my
        )

        if (
            not np.isfinite(lambda_my)
            or lambda_my <= 0.0
        ):
            raise ValueError(
                "lambda_my must be finite and strictly positive."
            )

    Ltotal = (
        Lf + 1.0 / lambda_my
    )

    if step_size_myula is None:
        c_delta = float(
            c_delta
        )

        if (
            not np.isfinite(c_delta)
            or c_delta <= 0.0
            or c_delta >= 2.0
        ):
            raise ValueError(
                "c_delta must lie strictly between 0 and 2."
            )

        step_size_myula = (
            c_delta / Ltotal
        )

        c_delta_used = c_delta

    else:
        step_size_myula = float(
            step_size_myula
        )

        if (
            not np.isfinite(step_size_myula)
            or step_size_myula <= 0.0
        ):
            raise ValueError(
                "step_size_myula must be finite and positive."
            )

        if (
            step_size_myula
            >= 2.0 / Ltotal
        ):
            raise ValueError(
                "step_size_myula must satisfy "
                "delta < 2/L_total_bound."
            )

        c_delta_used = None

    return (
        float(lambda_my),
        float(step_size_myula),
        float(Ltotal),
        c_delta_used,
    )


# ============================================================
# Main SAPG routine
# ============================================================

def sapg_laplace(
    y,
    X,
    sigma2,
    *,
    rng,
    n_iter,
    theta_init=None,
    beta_ref=None,
    beta0=None,
    intrinsic_dim=None,
    constraint="none",
    a=None,
    b=None,
    tau_c=None,
    theta_bounds=None,
    bound_factor=10.0,
    step_scale=1.0,
    step_offset=200.0,
    step_power=1.0,
    average_start=None,
    mcmc_steps=1,
    mcmc_warmup=0,
    lambda_my=None,
    step_size_myula=None,
    c_delta=0.9,
    store_state_path=False,
):
    r"""
    SAPG empirical-Bayes calibration of the Laplace scale theta.

    Parameters
    ----------
    y, X, sigma2
        Fixed likelihood inputs.

    rng : numpy.random.Generator
        Explicit RNG used by the MYULA transitions.

    n_iter : int
        Number of stochastic-approximation updates.

    theta_init : float or None
        Initial theta. If omitted, beta_ref must be supplied and

            theta_init = intrinsic_dim / ||beta_ref||_1.

    beta_ref : array_like or None
        Reference vector used only for moment initialisation.

    beta0 : array_like or None
        Initial MCMC state. If None and beta_ref is supplied,
        beta_ref is used. Otherwise zero is used.

    intrinsic_dim : int or None
        Dimension d appearing in

            d - theta E||beta||_1.

        Defaults to p for unconstrained/soft-affine models
        and p-1 for the hard sum-zero model.

        An explicit override is allowed for diagnostic studies.

    constraint : {'none', 'soft_affine', 'sum_zero'}

    theta_bounds : tuple or None
        Projection interval for theta. If None,

            [theta_init/bound_factor,
             theta_init*bound_factor]

        is used.

    step_scale, step_offset, step_power
        Robbins-Monro schedule

            rho_k
              =
              step_scale
              / (k + step_offset)^step_power.

        step_power must lie in (1/2, 1].

    average_start : int or None
        Number of completed SA updates discarded before averaging
        eta. If None, n_iter//2 is used.

    mcmc_steps : int
        Number of MYULA transitions used at each fixed theta_k.
        The score uses the arithmetic mean of ||beta||_1 over
        these transitions.

    mcmc_warmup : int
        Optional MYULA transitions at theta_init before the first
        SA update.

    lambda_my : float or None
        Moreau parameter. Default: 1/L_f.

    step_size_myula : float or None
        MYULA timestep. Default:
            c_delta / (L_f + 1/lambda_my).

    store_state_path : bool
        If True, retain the MCMC state after each SA iteration.

    Returns
    -------
    SapgResult
    """
    y, X = _validate_y_X(
        y, X
    )

    sigma2 = float(
        sigma2
    )

    if (
        not np.isfinite(sigma2)
        or sigma2 <= 0.0
    ):
        raise ValueError(
            "sigma2 must be finite and strictly positive."
        )

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator."
        )

    n_iter = _validate_positive_int(
        "n_iter",
        n_iter,
        lower=1,
    )

    mcmc_steps = _validate_positive_int(
        "mcmc_steps",
        mcmc_steps,
        lower=1,
    )

    mcmc_warmup = _validate_positive_int(
        "mcmc_warmup",
        mcmc_warmup,
        lower=0,
    )

    p = X.shape[1]

    (
        intrinsic_dim,
        default_dim,
        a,
        b,
        tau_c,
    ) = _validate_geometry(
        constraint,
        X,
        a,
        b,
        tau_c,
        intrinsic_dim,
    )

    # --------------------------------------------------------
    # Reference and initial MCMC state
    # --------------------------------------------------------

    beta_ref_arr = None

    if beta_ref is not None:
        beta_ref_arr = np.asarray(
            beta_ref,
            dtype=float,
        )

        if (
            beta_ref_arr.ndim != 1
            or beta_ref_arr.size != p
        ):
            raise ValueError(
                "beta_ref must be one-dimensional with length p."
            )

        if not np.all(
            np.isfinite(beta_ref_arr)
        ):
            raise ValueError(
                "beta_ref contains non-finite values."
            )

        if (
            constraint == "sum_zero"
            and abs(np.sum(beta_ref_arr)) > 1e-10
        ):
            raise ValueError(
                "For hard sum-zero SAPG, beta_ref must itself "
                "satisfy the sum-zero constraint."
            )

    theta_init = (
        _resolve_theta_initialisation(
            intrinsic_dim,
            p,
            theta_init,
            beta_ref_arr,
        )
    )

    if beta0 is None:
        if beta_ref_arr is not None:
            beta = beta_ref_arr.copy()
        else:
            beta = np.zeros(
                p,
                dtype=float,
            )
    else:
        beta = np.asarray(
            beta0,
            dtype=float,
        )

        if (
            beta.ndim != 1
            or beta.size != p
        ):
            raise ValueError(
                "beta0 must be one-dimensional with length p."
            )

        if not np.all(
            np.isfinite(beta)
        ):
            raise ValueError(
                "beta0 contains non-finite values."
            )

        beta = beta.copy()

    if constraint == "sum_zero":
        beta = project_sum_zero(
            beta
        )

    # --------------------------------------------------------
    # Theta projection interval
    # --------------------------------------------------------

    lower_theta, upper_theta = (
        _resolve_theta_bounds(
            theta_init,
            theta_bounds,
            bound_factor,
        )
    )

    eta_lower = float(
        np.log(lower_theta)
    )

    eta_upper = float(
        np.log(upper_theta)
    )

    # --------------------------------------------------------
    # SA schedule
    # --------------------------------------------------------

    step_scale = float(
        step_scale
    )

    step_offset = float(
        step_offset
    )

    step_power = float(
        step_power
    )

    if (
        not np.isfinite(step_scale)
        or step_scale <= 0.0
    ):
        raise ValueError(
            "step_scale must be finite and strictly positive."
        )

    if (
        not np.isfinite(step_offset)
        or step_offset < 0.0
    ):
        raise ValueError(
            "step_offset must be finite and nonnegative."
        )

    if (
        not np.isfinite(step_power)
        or step_power <= 0.5
        or step_power > 1.0
    ):
        raise ValueError(
            "step_power must lie in (0.5, 1]."
        )

    if average_start is None:
        average_start = (
            n_iter // 2
        )
    else:
        average_start = _validate_positive_int(
            "average_start",
            average_start,
            lower=0,
        )

        if average_start > n_iter:
            raise ValueError(
                "average_start cannot exceed n_iter."
            )

    # --------------------------------------------------------
    # Geometry-specific smooth curvature
    # --------------------------------------------------------

    if constraint == "none":
        Lf = smooth_lipschitz(
            X,
            sigma2,
        )

    elif constraint == "soft_affine":
        Lf = smooth_lipschitz(
            X,
            sigma2,
            a=a,
            tau_c=tau_c,
        )

    else:
        Lf = sum_zero_lipschitz(
            X,
            sigma2,
        )

    (
        lambda_my,
        step_size_myula,
        Ltotal,
        c_delta_used,
    ) = _resolve_myula_scales(
        Lf,
        lambda_my,
        step_size_myula,
        c_delta,
    )

    noise_scale = float(
        np.sqrt(
            2.0 * step_size_myula
        )
    )

    # --------------------------------------------------------
    # One MYULA transition at fixed theta
    # --------------------------------------------------------

    def transition(current_beta, theta):
        if constraint == "none":

            grad_f = gaussian_grad(
                current_beta,
                y,
                X,
                sigma2,
            )

            grad_h = laplace_moreau_grad(
                current_beta,
                theta,
                lambda_my,
            )

            innovation = rng.normal(
                size=p
            )

            new_beta = (
                current_beta
                - step_size_myula
                * (grad_f + grad_h)
                + noise_scale
                * innovation
            )

        elif constraint == "soft_affine":

            grad_f = soft_affine_grad(
                current_beta,
                y,
                X,
                sigma2,
                a,
                b,
                tau_c,
            )

            grad_h = laplace_moreau_grad(
                current_beta,
                theta,
                lambda_my,
            )

            innovation = rng.normal(
                size=p
            )

            new_beta = (
                current_beta
                - step_size_myula
                * (grad_f + grad_h)
                + noise_scale
                * innovation
            )

        else:

            grad_f = project_sum_zero(
                gaussian_grad(
                    current_beta,
                    y,
                    X,
                    sigma2,
                )
            )

            prox_beta = prox_l1_sum_zero(
                current_beta,
                lambda_my * theta,
            )

            grad_h = (
                current_beta
                - prox_beta
            ) / lambda_my

            innovation = project_sum_zero(
                rng.normal(size=p)
            )

            new_beta = (
                current_beta
                - step_size_myula
                * (grad_f + grad_h)
                + noise_scale
                * innovation
            )

            # Floating-point cleanup only.
            new_beta = project_sum_zero(
                new_beta
            )

        if not np.all(
            np.isfinite(new_beta)
        ):
            raise RuntimeError(
                "SAPG/MYULA produced a non-finite state."
            )

        if (
            constraint == "sum_zero"
            and abs(np.sum(new_beta)) > 1e-10
        ):
            raise RuntimeError(
                "Hard sum-zero SAPG left the constraint subspace."
            )

        return new_beta

    # --------------------------------------------------------
    # Optional fixed-theta chain warmup
    # --------------------------------------------------------

    for _ in range(
        mcmc_warmup
    ):
        beta = transition(
            beta,
            theta_init,
        )

    # --------------------------------------------------------
    # Allocate paths
    #
    # theta_path[k] and eta_path[k] are the parameter values
    # AFTER k completed SA updates.
    #
    # score_path[k] corresponds to theta_path[k], before update
    # k+1.
    # --------------------------------------------------------

    eta = float(
        np.log(theta_init)
    )

    theta_path = np.empty(
        n_iter + 1,
        dtype=float,
    )

    eta_path = np.empty(
        n_iter + 1,
        dtype=float,
    )

    score_path = np.empty(
        n_iter,
        dtype=float,
    )

    g_path = np.empty(
        n_iter,
        dtype=float,
    )

    rho_path = np.empty(
        n_iter,
        dtype=float,
    )

    # -1 = lower projection,
    #  0 = no projection,
    # +1 = upper projection.
    bound_hit_path = np.zeros(
        n_iter,
        dtype=np.int8,
    )

    if store_state_path:
        state_path = np.empty(
            (n_iter, p),
            dtype=float,
        )
    else:
        state_path = np.empty(
            (0, p),
            dtype=float,
        )

    eta_path[0] = eta
    theta_path[0] = theta_init

    # --------------------------------------------------------
    # SAPG iterations
    # --------------------------------------------------------

    for k in range(
        1,
        n_iter + 1,
    ):
        theta_k = float(
            np.exp(eta)
        )

        # Posterior expectation estimate from one or more
        # continuing MYULA transitions at fixed theta_k.
        g_sum = 0.0

        for _ in range(
            mcmc_steps
        ):
            beta = transition(
                beta,
                theta_k,
            )

            g_sum += float(
                np.sum(np.abs(beta))
            )

        g_bar = (
            g_sum / mcmc_steps
        )

        score = (
            intrinsic_dim
            - theta_k * g_bar
        )

        rho = (
            step_scale
            / (k + step_offset)
            ** step_power
        )

        eta_proposed = (
            eta + rho * score
        )

        if eta_proposed < eta_lower:
            eta_new = eta_lower
            bound_hit = -1

        elif eta_proposed > eta_upper:
            eta_new = eta_upper
            bound_hit = 1

        else:
            eta_new = eta_proposed
            bound_hit = 0

        score_path[k - 1] = score
        g_path[k - 1] = g_bar
        rho_path[k - 1] = rho
        bound_hit_path[k - 1] = bound_hit

        eta = float(
            eta_new
        )

        eta_path[k] = eta
        theta_path[k] = float(
            np.exp(eta)
        )

        if store_state_path:
            state_path[k - 1] = beta

    # --------------------------------------------------------
    # Averaged estimate
    #
    # average_start means: discard the first `average_start`
    # completed SA updates. Thus eta_path[average_start:] is
    # averaged.
    # --------------------------------------------------------

    eta_average_segment = (
        eta_path[
            average_start:
        ]
    )

    theta_average_segment = (
        theta_path[
            average_start:
        ]
    )

    eta_hat = float(
        np.mean(
            eta_average_segment
        )
    )

    theta_hat = float(
        np.exp(eta_hat)
    )

    theta_mean_arithmetic = float(
        np.mean(
            theta_average_segment
        )
    )

    return SapgResult(
        theta_hat=theta_hat,
        eta_hat=eta_hat,
        theta_mean_arithmetic=theta_mean_arithmetic,

        theta_final=float(
            theta_path[-1]
        ),
        eta_final=float(
            eta_path[-1]
        ),
        beta_final=np.asarray(
            beta,
            dtype=float,
        ).copy(),

        theta_init=float(
            theta_init
        ),
        intrinsic_dim=int(
            intrinsic_dim
        ),

        n_iter=int(
            n_iter
        ),
        average_start=int(
            average_start
        ),
        mcmc_steps=int(
            mcmc_steps
        ),
        mcmc_warmup=int(
            mcmc_warmup
        ),
        n_mcmc_transitions=int(
            mcmc_warmup
            + n_iter * mcmc_steps
        ),

        lambda_my=float(
            lambda_my
        ),
        step_size_myula=float(
            step_size_myula
        ),
        c_delta=c_delta_used,
        lipschitz_smooth=float(
            Lf
        ),
        lipschitz_total_bound=float(
            Ltotal
        ),

        step_scale=float(
            step_scale
        ),
        step_offset=float(
            step_offset
        ),
        step_power=float(
            step_power
        ),

        theta_bounds=(
            float(lower_theta),
            float(upper_theta),
        ),

        theta_path=theta_path,
        eta_path=eta_path,
        score_path=score_path,
        g_path=g_path,
        rho_path=rho_path,
        bound_hit_path=bound_hit_path,

        state_path=state_path,

        constraint=constraint,
    )
