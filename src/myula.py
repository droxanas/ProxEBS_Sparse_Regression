
"""
myula.py

Moreau-Yosida Unadjusted Langevin Algorithm (MYULA) for the
computational-statistics sparse-regression pipeline.

Target approximation
--------------------
The statistical target is the original nonsmooth posterior

    pi(beta) proportional to
        exp[-f(beta) - theta ||beta||_1].

MYULA instead uses the Moreau-smoothed potential

    U_lambda(beta)
      = f(beta)
        + (theta ||.||_1)^lambda(beta).

Thus MYULA provides samples from a finite-step approximation
to the smoothed posterior. Both lambda_my and step_size are
therefore numerical approximation parameters.

Default numerical scales
------------------------
If lambda_my is not supplied,

    lambda_my = 1 / L_f.

The conservative gradient-Lipschitz bound is then

    L_total = L_f + 1/lambda_my.

If step_size is not supplied,

    delta = c_delta / L_total,

with c_delta=0.9 by default.

Hard sum-zero geometry
----------------------
For

    C = {beta : 1^T beta = 0},

the Moreau envelope is taken within C:

    h_C^lambda(beta)
      =
      min_{u in C}
      theta ||u||_1
      + ||u-beta||^2/(2 lambda).

The drift and Gaussian innovations are both restricted to C.

Randomness
----------
A numpy.random.Generator must be supplied explicitly.
No global random state is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prox import (
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
    laplace_moreau_value,
    laplace_moreau_grad,
    smoothed_posterior_objective,
)


__all__ = [
    "MyulaResult",
    "myula",
    "myula_sum_zero",
]


@dataclass(frozen=True)
class MyulaResult:
    samples: np.ndarray
    final_state: np.ndarray
    n_samples: int
    burn_in: int
    thin: int
    n_total_steps: int
    lambda_my: float
    step_size: float
    c_delta: float | None
    lipschitz_smooth: float
    lipschitz_total_bound: float
    potential_path: np.ndarray
    constraint: str


# ============================================================
# Validation helpers
# ============================================================

def _validate_inputs(
    y,
    X,
    sigma2,
    theta,
    beta0,
    rng,
    n_samples,
    burn_in,
    thin,
):
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

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator."
        )

    for name, value, lower in (
        ("n_samples", n_samples, 1),
        ("burn_in", burn_in, 0),
        ("thin", thin, 1),
    ):
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")

        if value < lower:
            raise ValueError(
                f"{name} must be >= {lower}."
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

    return (
        y,
        X,
        sigma2,
        theta,
        beta0,
    )


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


def _resolve_numerical_scales(
    Lf,
    lambda_my,
    step_size,
    c_delta,
):
    if not np.isfinite(Lf) or Lf <= 0.0:
        raise ValueError(
            "L_f must be finite and strictly positive."
        )

    if lambda_my is None:
        lambda_my = 1.0 / Lf
    else:
        lambda_my = float(lambda_my)

        if (
            not np.isfinite(lambda_my)
            or lambda_my <= 0.0
        ):
            raise ValueError(
                "lambda_my must be finite and strictly positive."
            )

    Ltotal = Lf + 1.0 / lambda_my

    if step_size is None:
        c_delta = float(c_delta)

        if (
            not np.isfinite(c_delta)
            or c_delta <= 0.0
            or c_delta >= 2.0
        ):
            raise ValueError(
                "c_delta must lie strictly between 0 and 2."
            )

        step_size = c_delta / Ltotal

    else:
        step_size = float(step_size)

        if (
            not np.isfinite(step_size)
            or step_size <= 0.0
        ):
            raise ValueError(
                "step_size must be finite and strictly positive."
            )

        # A user-specified step size is allowed, but reject
        # obviously unstable choices relative to the supplied
        # conservative Lipschitz bound.
        if step_size >= 2.0 / Ltotal:
            raise ValueError(
                "step_size must satisfy "
                "step_size < 2 / L_total_bound."
            )

        c_delta = None

    return (
        float(lambda_my),
        float(step_size),
        float(Ltotal),
        c_delta,
    )


# ============================================================
# Unconstrained / soft-affine MYULA
# ============================================================

def myula(
    y,
    X,
    sigma2,
    theta,
    *,
    rng,
    n_samples,
    burn_in=0,
    thin=1,
    beta0=None,
    lambda_my=None,
    step_size=None,
    c_delta=0.9,
    a=None,
    b=None,
    tau_c=None,
    store_potential=True,
):
    r"""
    Run MYULA for the unconstrained or soft-affine model.

    Update:

        beta_{k+1}
          =
          beta_k
          - delta [
              grad f(beta_k)
              + grad h_theta^lambda(beta_k)
            ]
          + sqrt(2 delta) xi_k,

    xi_k ~ N(0, I).

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit RNG.

    n_samples : int
        Number of retained samples.

    burn_in : int
        Number of initial MYULA transitions discarded.

    thin : int
        Keep every `thin`-th transition after burn-in.

    lambda_my : float or None
        Moreau parameter. Default: 1/L_f.

    step_size : float or None
        MYULA timestep. Default:
            c_delta / (L_f + 1/lambda_my).

    store_potential : bool
        If True, store the smoothed potential at every retained
        sample. If False, potential_path is empty.

    Returns
    -------
    MyulaResult
    """
    (
        y,
        X,
        sigma2,
        theta,
        beta,
    ) = _validate_inputs(
        y,
        X,
        sigma2,
        theta,
        beta0,
        rng,
        n_samples,
        burn_in,
        thin,
    )

    use_affine = _parse_affine(
        a, b, tau_c
    )

    if use_affine:
        a = np.asarray(a, dtype=float)

        if a.ndim != 1 or a.size != X.shape[1]:
            raise ValueError(
                "a must be one-dimensional with length p."
            )

        b = float(b)
        tau_c = float(tau_c)

        if not np.all(np.isfinite(a)):
            raise ValueError("a contains non-finite values.")

        if not np.isfinite(b):
            raise ValueError("b must be finite.")

        if (
            not np.isfinite(tau_c)
            or tau_c <= 0.0
        ):
            raise ValueError(
                "tau_c must be finite and strictly positive."
            )

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

    (
        lambda_my,
        step_size,
        Ltotal,
        c_delta_used,
    ) = _resolve_numerical_scales(
        Lf,
        lambda_my,
        step_size,
        c_delta,
    )

    noise_scale = np.sqrt(
        2.0 * step_size
    )

    n_total_steps = (
        burn_in
        + n_samples * thin
    )

    p = X.shape[1]

    samples = np.empty(
        (n_samples, p),
        dtype=float,
    )

    if store_potential:
        potential_path = np.empty(
            n_samples,
            dtype=float,
        )
    else:
        potential_path = np.empty(
            0,
            dtype=float,
        )

    keep_index = 0

    for step_index in range(1, n_total_steps + 1):

        if use_affine:
            grad_f = soft_affine_grad(
                beta,
                y,
                X,
                sigma2,
                a,
                b,
                tau_c,
            )
        else:
            grad_f = gaussian_grad(
                beta,
                y,
                X,
                sigma2,
            )

        grad_h = laplace_moreau_grad(
            beta,
            theta,
            lambda_my,
        )

        innovation = rng.normal(
            size=p
        )

        beta = (
            beta
            - step_size * (grad_f + grad_h)
            + noise_scale * innovation
        )

        if not np.all(np.isfinite(beta)):
            raise RuntimeError(
                "MYULA produced non-finite state values. "
                "The timestep may be too large or the model "
                "may be numerically ill-conditioned."
            )

        if step_index > burn_in:
            after_burn = (
                step_index - burn_in
            )

            if after_burn % thin == 0:
                samples[keep_index] = beta

                if store_potential:
                    potential_path[
                        keep_index
                    ] = smoothed_posterior_objective(
                        beta,
                        y,
                        X,
                        sigma2,
                        theta,
                        lambda_my,
                        a=a,
                        b=b,
                        tau_c=tau_c,
                    )

                keep_index += 1

    if keep_index != n_samples:
        raise RuntimeError(
            "Internal sample-retention bookkeeping failed."
        )

    return MyulaResult(
        samples=samples,
        final_state=np.asarray(
            beta,
            dtype=float,
        ).copy(),
        n_samples=int(n_samples),
        burn_in=int(burn_in),
        thin=int(thin),
        n_total_steps=int(n_total_steps),
        lambda_my=float(lambda_my),
        step_size=float(step_size),
        c_delta=c_delta_used,
        lipschitz_smooth=float(Lf),
        lipschitz_total_bound=float(Ltotal),
        potential_path=potential_path,
        constraint=constraint_name,
    )


# ============================================================
# Hard sum-zero MYULA
# ============================================================

def myula_sum_zero(
    y,
    X,
    sigma2,
    theta,
    *,
    rng,
    n_samples,
    burn_in=0,
    thin=1,
    beta0=None,
    lambda_my=None,
    step_size=None,
    c_delta=0.9,
    store_potential=True,
):
    r"""
    MYULA directly on

        C = {beta : 1^T beta = 0}.

    The update is

        beta_{k+1}
          =
          beta_k
          - delta [
              P_C grad f(beta_k)
              + grad h_C^lambda(beta_k)
            ]
          + sqrt(2 delta) P_C xi_k,

    where

        grad h_C^lambda(beta)
          =
          [
            beta
            - prox^C_{lambda theta ||.||_1}(beta)
          ] / lambda.

    Every state therefore remains in C up to floating-point
    accuracy.
    """
    (
        y,
        X,
        sigma2,
        theta,
        beta,
    ) = _validate_inputs(
        y,
        X,
        sigma2,
        theta,
        beta0,
        rng,
        n_samples,
        burn_in,
        thin,
    )

    beta = project_sum_zero(
        beta
    )

    Lf = sum_zero_lipschitz(
        X,
        sigma2,
    )

    (
        lambda_my,
        step_size,
        Ltotal,
        c_delta_used,
    ) = _resolve_numerical_scales(
        Lf,
        lambda_my,
        step_size,
        c_delta,
    )

    noise_scale = np.sqrt(
        2.0 * step_size
    )

    n_total_steps = (
        burn_in
        + n_samples * thin
    )

    p = X.shape[1]

    samples = np.empty(
        (n_samples, p),
        dtype=float,
    )

    if store_potential:
        potential_path = np.empty(
            n_samples,
            dtype=float,
        )
    else:
        potential_path = np.empty(
            0,
            dtype=float,
        )

    def constrained_moreau_value_grad(v):
        prox_v = prox_l1_sum_zero(
            v,
            lambda_my * theta,
        )

        diff = prox_v - v

        value = (
            theta
            * np.sum(np.abs(prox_v))
            + 0.5
            * float(diff @ diff)
            / lambda_my
        )

        grad = (
            v - prox_v
        ) / lambda_my

        return float(value), grad

    keep_index = 0

    for step_index in range(1, n_total_steps + 1):

        grad_f = project_sum_zero(
            gaussian_grad(
                beta,
                y,
                X,
                sigma2,
            )
        )

        _, grad_h = (
            constrained_moreau_value_grad(
                beta
            )
        )

        innovation = project_sum_zero(
            rng.normal(size=p)
        )

        beta = (
            beta
            - step_size * (grad_f + grad_h)
            + noise_scale * innovation
        )

        # Only floating-point cleanup.
        beta = project_sum_zero(
            beta
        )

        if not np.all(np.isfinite(beta)):
            raise RuntimeError(
                "MYULA produced non-finite state values. "
                "The timestep may be too large or the model "
                "may be numerically ill-conditioned."
            )

        if abs(np.sum(beta)) > 1e-10:
            raise RuntimeError(
                "Hard sum-zero MYULA left the constraint "
                "subspace unexpectedly."
            )

        if step_index > burn_in:
            after_burn = (
                step_index - burn_in
            )

            if after_burn % thin == 0:
                samples[keep_index] = beta

                if store_potential:
                    h_value, _ = (
                        constrained_moreau_value_grad(
                            beta
                        )
                    )

                    potential_path[
                        keep_index
                    ] = (
                        gaussian_nll(
                            beta,
                            y,
                            X,
                            sigma2,
                        )
                        + h_value
                    )

                keep_index += 1

    if keep_index != n_samples:
        raise RuntimeError(
            "Internal sample-retention bookkeeping failed."
        )

    return MyulaResult(
        samples=samples,
        final_state=np.asarray(
            beta,
            dtype=float,
        ).copy(),
        n_samples=int(n_samples),
        burn_in=int(burn_in),
        thin=int(thin),
        n_total_steps=int(n_total_steps),
        lambda_my=float(lambda_my),
        step_size=float(step_size),
        c_delta=c_delta_used,
        lipschitz_smooth=float(Lf),
        lipschitz_total_bound=float(Ltotal),
        potential_path=potential_path,
        constraint="sum_zero",
    )
