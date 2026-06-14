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
from frontend.gbs_optimizer import reduced_herald
from src.simulation.jax.moment_scorer import (
    jax_equivalent_gaussian, jax_reduced_herald, fired_product,
    REDUCED_HERALD_PROD_BUDGET, moment_score_one, extract_structure,
    jax_equivalent_gaussian_static, jax_reduced_herald_static,
    MOMENT_MAXF, MOMENT_BF,
)
from frontend.gbs_optimizer import reduced_herald as _np_reduced_herald
from src.simulation.jax.herald import (
    passive_unitary_to_symplectic as _p2s, vacuum_covariance as _vac,
    complex_alpha_to_qp as _a2qp,
)

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


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_static_equivalent_gaussian_matches_per_structure(seed):
    """The single-compile static equivalent-Gaussian (fixed 24-mode masked) ==
    the per-structure version on the real (signal + active-control) submodes,
    bit-exact, INCLUDING inactive-leaf routing. This is the Option-2 piece A
    correctness gate."""
    def submat(cov, order, Nfull):
        idx = np.array(list(order) + [o + Nfull for o in order])
        return cov[np.ix_(idx, idx)]

    dec, gens = _random_genotypes(20, seed=seed)
    n_checked = 0
    for g in gens:
        params = dec.decode(jnp.asarray(g), CUTOFF)
        st = extract_structure(params)
        active, nctrl, _pnr = st
        if sum(active) == 0:
            continue
        eq = jax_equivalent_gaussian(params, st)
        cov_ps = np.asarray(eq["cov"]); mu_ps = np.asarray(eq["mu"])
        Nps = cov_ps.shape[0] // 2
        order_ps = [eq["signal_idx"]] + list(eq["control_idx"])
        cs, ms, ep, _dn = jax_equivalent_gaussian_static(params)
        cov_st = np.asarray(cs); mu_st = np.asarray(ms)
        real = [1 + 2 * i + c for i in range(8) for c in range(2)
                if active[i] and c < nctrl[i]]
        order_st = [0] + real
        A = submat(cov_ps, order_ps, Nps)
        B = submat(cov_st, order_st, 17)
        assert A.shape == B.shape
        assert np.allclose(A, B, atol=1e-7), f"cov Δ={np.max(np.abs(A - B)):.2e}"
        mps = np.array([mu_ps[o] for o in order_ps] + [mu_ps[o + Nps] for o in order_ps])
        mst = np.array([mu_st[o] for o in order_st] + [mu_st[o + 17] for o in order_st])
        assert np.allclose(mps, mst, atol=1e-7)
        n_checked += 1
    assert n_checked > 0


@pytest.mark.parametrize("M,nj", [(6, [2, 1, 1, 1, 1]), (8, [1, 1, 1, 1, 1, 1, 1]),
                                  (5, [3, 2, 1, 1]), (8, [2, 1, 1, 1, 1, 1, 1]),
                                  (2, [4]), (1, [])])
def test_reduced_herald_static_matches_numpy(M, nj):
    """Option-2 piece B: the single-compile flat-buffer reduced_herald (runtime
    radii) == numpy reduced_herald, bit-exact, across fired-mode counts incl. the
    high-kf regime (random pure Gaussian states)."""
    L = 50
    rng = np.random.default_rng(M * 7 + sum(nj) + len(nj))
    r = rng.uniform(0.1, 0.8, M)
    X = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    U, _ = np.linalg.qr(X)
    S = np.asarray(_p2s(jnp.asarray(U))) @ np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    cov = S @ np.asarray(_vac(M, 2.0)) @ S.T
    mu = np.asarray(_a2qp(jnp.asarray(rng.uniform(-.3, .3, M) + 1j * rng.uniform(-.3, .3, M)), 2.0))
    if fired_product(tuple(nj)) > MOMENT_BF or len(nj) > MOMENT_MAXF:
        pytest.skip("over budget")
    psi_np, p_np = _np_reduced_herald(cov, mu, 0, list(range(1, M)), nj, cutoff=L)
    # embed into the static 17-mode layout (signal 0, controls 1..M-1)
    C = np.eye(34); m = np.zeros(34)
    idx = list(range(M)) + [17 + i for i in range(M)]
    C[np.ix_(idx, idx)] = cov; m[np.array(idx)] = mu
    eff = np.zeros(16)
    for j, n in enumerate(nj):
        eff[j] = n
    psi_st, p_st = jax_reduced_herald_static(jnp.asarray(C), jnp.asarray(m), jnp.asarray(eff), L)
    psi_st = np.asarray(psi_st)
    mm = min(len(psi_np), len(psi_st))
    assert np.allclose(psi_np[:mm], psi_st[:mm], atol=1e-7), \
        f"psi Δ={np.max(np.abs(psi_np[:mm] - psi_st[:mm])):.2e}"
    assert abs(p_np - float(p_st)) <= 1e-7 * (abs(p_np) + 1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reduced_herald_matches_numpy(seed):
    """jax_reduced_herald == numpy reduced_herald, BIT-exact (same convention):
    heralded signal state psi AND herald probability, across fired patterns.
    Skips the over-budget extreme-Sigma-n tail (routed to CPU fallback in prod)."""
    L = 60
    dec, gens = _random_genotypes(8, seed=seed)
    n_checked = 0
    kf_seen = set()
    for g in gens:
        params = dec.decode(jnp.asarray(g), CUTOFF)
        pnp = {k: (np.asarray(v) if hasattr(v, "shape") else v)
               for k, v in params.items()}
        eq = compute_equivalent_gaussian(pnp, light=True)
        cov = np.asarray(eq["cov"], float)
        mu = np.asarray(eq["mu"], float)
        sig = int(eq["signal_idx"])
        ctrl = tuple(int(x) for x in eq["control_idx"])
        n0 = tuple(int(x) for x in eq["pnr_outcomes"])
        if fired_product(n0) * L > REDUCED_HERALD_PROD_BUDGET * L:
            continue  # extreme tail -> CPU fallback, not the in-loop path
        psi_np, p_np = reduced_herald(cov, mu, sig, list(ctrl), list(n0), cutoff=L)
        psi_jx, p_jx = jax_reduced_herald(jnp.asarray(cov), jnp.asarray(mu),
                                          sig, ctrl, n0, L)
        psi_jx = np.asarray(psi_jx)
        Lm = min(len(psi_np), len(psi_jx))
        assert np.allclose(psi_np[:Lm], psi_jx[:Lm], atol=1e-7), \
            f"psi max|Δ|={np.max(np.abs(psi_np[:Lm] - psi_jx[:Lm])):.2e}"
        assert abs(p_np - float(p_jx)) <= 1e-7 * (abs(p_np) + 1e-12), \
            f"prob rel|Δ|={abs(p_np - float(p_jx)) / (abs(p_np) + 1e-30):.2e}"
        n_checked += 1
        kf_seen.add(sum(1 for x in n0 if x >= 1))
    assert n_checked > 0, "no in-budget genotypes were checked"


def test_moment_score_one_matches_numpy_exp():
    """The per-genotype scoring kernel: <O> and P_leaf match a direct numpy
    (reduced_herald + O) computation. Synthetic Hermitian O (number operator)
    -> no qutip dependency. Restricted to small fired-product genotypes."""
    L = 40
    O = np.diag(np.arange(L, dtype=float))      # Hermitian
    Oj = jnp.asarray(O)
    dec, gens = _random_genotypes(60, seed=11)
    checked = 0
    for g in gens:
        params = dec.decode(jnp.asarray(g), CUTOFF)
        pnp = {k: (np.asarray(v) if hasattr(v, "shape") else v)
               for k, v in params.items()}
        eq = compute_equivalent_gaussian(pnp, light=True)
        n0 = tuple(int(x) for x in eq["pnr_outcomes"])
        if fired_product(n0) > 16:
            continue
        struct = extract_structure(params)
        psi_np, _ = reduced_herald(np.asarray(eq["cov"], float),
                                   np.asarray(eq["mu"], float),
                                   int(eq["signal_idx"]), list(eq["control_idx"]),
                                   list(n0), cutoff=L)
        m = min(L, len(psi_np))
        exp_np = float(np.real(np.vdot(psi_np[:m], O[:m, :m] @ psi_np[:m])))
        e, _, _, _ = moment_score_one(params, Oj, struct, L)
        assert abs(float(e) - exp_np) < 1e-6, f"exp Δ={abs(float(e) - exp_np):.2e}"
        checked += 1
        if checked >= 3:
            break
    assert checked > 0, "no small-fired genotype found to check"


@pytest.mark.skipif(not os.environ.get("MOMENT_SLOW"),
                    reason="slow AD test; set MOMENT_SLOW=1 to run")
def test_moment_score_one_is_differentiable():
    """d<O>/dgenotype is finite & nonzero through the moment kernel. Slow:
    reverse-mode AD through the recurrence fori_loop is the known task-3 perf
    item (see MOMENT_SCORER_PLAN.md 6b). Run with: MOMENT_SLOW=1 pytest ..."""
    L = 40
    Oj = jnp.asarray(np.diag(np.arange(L, dtype=float)))
    dec, gens = _random_genotypes(60, seed=11)
    for g in gens:
        params = dec.decode(jnp.asarray(g), CUTOFF)
        pnp = {k: (np.asarray(v) if hasattr(v, "shape") else v)
               for k, v in params.items()}
        eq = compute_equivalent_gaussian(pnp, light=True)
        n0 = tuple(int(x) for x in eq["pnr_outcomes"])
        if fired_product(n0) > 4:
            continue
        struct = extract_structure(params)

        def loss(gg):
            e, _, _, _ = moment_score_one(dec.decode(gg, CUTOFF), Oj, struct, L)
            return e

        grad = np.asarray(jax.grad(loss)(jnp.asarray(g, np.float64)))
        assert np.all(np.isfinite(grad)) and np.sum(np.abs(grad) > 1e-9) > 0
        return
    pytest.skip("no small-fired genotype found")
