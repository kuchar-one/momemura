"""Clamped-Gaussian reference optimum (the "G_N" line).

The analytic Gaussian limit (e.g. 2/3 for alpha=beta=1) is an INFIMUM reached
only as squeezing r -> infinity; the search operates under a squeezing clamp
(tanh scaling * r_scale), so the best *reachable* Gaussian value is strictly
above the limit.  Progress lines that compare only against the r->inf limit
("vs G") systematically overstate how close the search is to beating
Gaussianity: at r_scale ~ 1.9 the clamped Gaussian optimum sits ~0.07 above
the limit, exactly the "gap" every stagnating run reports.

This module computes G_N = min <psi(r, theta, beta)| O |psi(r, theta, beta)>
over pure single-mode Gaussians D(beta) S(r e^{i*theta}) |0> with |r| <= r_max
and |Re beta|, |Im beta| <= d_max, by multi-start Adam in the SAME Fock-space
conventions as the moment scorer (it reuses ``_herald_static``, so the state
construction is bit-identical: hbar=2, xp-ordering, vacuum cov = I).

Two references are reported:
  * G_N   at r_max = r_scale       (single clamped squeezer, the practical bar)
  * G_N2  at r_max = 2 * r_scale   (leaf + final squeezers can compose along
                                    one quadrature, so the reachable Gaussian
                                    set extends to ~2x the per-element clamp)
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp


def _gauss_expectation_factory(operator_L: jnp.ndarray, r_max: float,
                               d_max: float, L: int):
    """<O> of D(beta)S(r e^{i theta})|0> as a jax fn of 4 raw params."""
    from src.simulation.jax.moment_scorer import _herald_static, HBAR
    from src.simulation.jax.herald import (
        passive_unitary_to_symplectic, vacuum_covariance)

    def expectation(raw):
        r = jnp.tanh(raw[0]) * r_max
        theta = raw[1]                     # unbounded angle
        bx = jnp.tanh(raw[2]) * d_max
        by = jnp.tanh(raw[3]) * d_max

        # single-mode squeezed cov in xp-ordering (hbar=2): R diag(e^-2r, e^2r) R^T
        c, s = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
        R = jnp.array([[c, -s], [s, c]])
        cov1 = R @ jnp.diag(jnp.array([jnp.exp(-2.0 * r),
                                       jnp.exp(2.0 * r)])) @ R.T
        mu1 = jnp.array([2.0 * bx, 2.0 * by])   # hbar=2: <x> = 2 Re(beta)

        # embed as mode 0 of a 3-mode system (modes 1,2 = decoupled vacuum) and
        # reuse the production herald with no fired detectors: bit-identical
        # Fock amplitudes to the search's own states.
        N = 3
        cov = jnp.eye(2 * N)
        idx = jnp.asarray([0, N])
        cov = cov.at[jnp.ix_(idx, idx)].set(cov1)
        mu = jnp.zeros(2 * N).at[idx].set(mu1)
        eff = jnp.zeros(2)
        psi, _prob = _herald_static(cov, mu, eff, 2, N, 2, 4, L)
        return jnp.real(jnp.vdot(psi, operator_L[:psi.shape[0], :psi.shape[0]] @ psi))

    return expectation


def clamped_gaussian_optimum(operator_L, r_max: float, d_max: float,
                             n_starts: int = 256, iters: int = 400,
                             lr: float = 0.05, seed: int = 0
                             ) -> Tuple[float, np.ndarray]:
    """Multi-start Adam minimisation of the clamped-Gaussian <O>.
    Returns (best_value, best_raw_params[4])."""
    import optax

    op = jnp.asarray(operator_L)
    L = int(op.shape[0])
    expectation = _gauss_expectation_factory(op, float(r_max), float(d_max), L)

    key = jax.random.PRNGKey(seed)
    raw0 = jax.random.uniform(key, (n_starts, 4), minval=-2.0, maxval=2.0)
    # include canonical starts: vacuum, +/- max squeezing at theta = 0, pi/2
    canon = jnp.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0],
        [3.0, np.pi / 2, 0.0, 0.0], [-3.0, np.pi / 2, 0.0, 0.0],
    ])
    raw0 = jnp.concatenate([canon, raw0[:-canon.shape[0]]], axis=0)

    opt = optax.adam(lr)
    grad_fn = jax.vmap(jax.value_and_grad(expectation))

    @jax.jit
    def step(raws, opt_state):
        vals, grads = grad_fn(raws)
        updates, opt_state = opt.update(grads, opt_state, raws)
        return optax.apply_updates(raws, updates), opt_state, vals

    raws = raw0
    opt_state = opt.init(raws)
    vals = None
    for _ in range(iters):
        raws, opt_state, vals = step(raws, opt_state)
    vals, _ = grad_fn(raws)
    best = int(jnp.argmin(vals))
    return float(vals[best]), np.asarray(raws[best])


@lru_cache(maxsize=8)
def gaussian_reference_values(alpha_str: str, beta_str: str, L: int,
                              r_scale: float, d_scale: float
                              ) -> Tuple[float, float]:
    """(G_N, G_N2): clamped-Gaussian optima at r_max = r_scale and 2*r_scale.
    Cached per (target, L, clamps); costs a few seconds on GPU, once."""
    from src.simulation.jax.moment_scorer import moment_operator
    op = moment_operator(int(L), alpha_str, beta_str)
    gn, _ = clamped_gaussian_optimum(op, r_scale, d_scale)
    gn2, _ = clamped_gaussian_optimum(op, 2.0 * r_scale, d_scale)
    return gn, gn2
