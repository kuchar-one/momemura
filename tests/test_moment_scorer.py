"""Tests for the moment-space optimizer scorer (src/simulation/jax/moment_scorer.py).

Self-contained: random genotypes (no stored repertoires needed), validated
against the trusted numpy reference
``frontend.gaussian_decomposition.compute_equivalent_gaussian``.

Run:  JAX_ENABLE_X64=1 python -m pytest tests/test_moment_scorer.py -q
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.genotypes.genotypes import get_genotype_decoder
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from src.simulation.jax.moment_scorer import jax_equivalent_gaussian

CFG = dict(genotype="00B", depth=3, modes=3, pnr_max=15,
           r_scale=1.87, d_scale=2.24, hx_scale=1.37, window=0.0)
CUTOFF = 30


def _decoder():
    return get_genotype_decoder("00B", depth=3, config=CFG)


def _random_genotypes(n, seed=0):
    dec = _decoder()
    D = dec.get_length(3)
    rng = np.random.default_rng(seed)
    return dec, rng.uniform(-3.0, 3.0, size=(n, D)).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_equivalent_gaussian_matches_numpy(seed):
    """cov, mu, densities and signal/control/pnr bookkeeping agree to ~1e-9."""
    dec, gens = _random_genotypes(6, seed=seed)
    for g in gens:
        params = dec.decode(jnp.asarray(g), CUTOFF)
        pnp = {k: (np.asarray(v) if hasattr(v, "shape") else v)
               for k, v in params.items()}
        eq_np = compute_equivalent_gaussian(pnp, light=True)
        eq_jx = jax_equivalent_gaussian(params)

        cov_np, cov_jx = np.asarray(eq_np["cov"]), np.asarray(eq_jx["cov"])
        mu_np, mu_jx = np.asarray(eq_np["mu"]), np.asarray(eq_jx["mu"])
        assert cov_np.shape == cov_jx.shape, "covariance shape mismatch"
        assert np.allclose(cov_np, cov_jx, atol=1e-7), \
            f"cov max|Δ|={np.max(np.abs(cov_np - cov_jx)):.2e}"
        assert np.allclose(mu_np, mu_jx, atol=1e-7), \
            f"mu max|Δ|={np.max(np.abs(mu_np - mu_jx)):.2e}"
        assert eq_np["signal_idx"] == eq_jx["signal_idx"]
        assert list(eq_np["control_idx"]) == list(eq_jx["control_idx"])
        assert list(eq_np["pnr_outcomes"]) == list(eq_jx["pnr_outcomes"])

        d_np = np.asarray(eq_np.get("homodyne_densities", []), float)
        d_jx = np.asarray([float(x) for x in eq_jx.get("homodyne_densities", [])])
        if d_np.size:
            assert np.allclose(d_np, d_jx, atol=1e-7), "homodyne density mismatch"


def test_equivalent_gaussian_is_differentiable():
    """Gradient flows through .at/ix_/gather; matches finite differences."""
    dec, gens = _random_genotypes(1, seed=7)
    g0 = jnp.asarray(gens[0].astype(np.float64))

    def scalar(g):
        eq = jax_equivalent_gaussian(dec.decode(g, CUTOFF))
        return jnp.real(jnp.sum(eq["cov"])) + jnp.real(jnp.sum(eq["mu"]))

    grad = np.asarray(jax.grad(scalar)(g0))
    assert np.all(np.isfinite(grad))
    assert np.sum(np.abs(grad) > 1e-9) > 0, "no gradient flowed"

    i = int(np.argmax(np.abs(grad)))
    h = 1e-5
    fd = (float(scalar(g0.at[i].add(h))) - float(scalar(g0.at[i].add(-h)))) / (2 * h)
    assert abs(grad[i] - fd) / (abs(fd) + 1e-12) < 1e-5
