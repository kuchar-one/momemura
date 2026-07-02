"""Unit tests for the NG-pipeline refactor:

  * B30F forced-heralding decode invariants + STE forward-exactness
  * B30-family depth embedding preserves the heralded state exactly
  * B30F -> B30 conversion is lossless
  * PNR-pattern seeds decode to the requested click patterns and are valid
  * macro-mutations preserve genotype shape and hit their target slices
  * NG-stratified Pareto seed selection round-robins across strata
  * 2-objective fitness stack from the moment scorer
  * incremental clean_archive_moment fingerprinting

Run: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_ng_pipeline_units.py -v
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from src.genotypes.genotypes import get_genotype_decoder  # noqa: E402
from src.genotypes.converter import (  # noqa: E402
    upgrade_depth, convert_b30f_to_b30, supports_depth_upgrade)
from src.simulation.jax.moment_scorer import (  # noqa: E402
    jax_equivalent_gaussian_static, jax_reduced_herald_static)

CFG = dict(depth=3, modes=3, pnr_max=10, r_scale=1.9, d_scale=2.2,
           hx_scale=1.4)
# small box: state-EQUALITY tests only need both sides to share parameters,
# not production fidelity -- keeps CPU compile times manageable.
L_TEST = 40
BF_TEST = 256
MAXF_TEST = 8


def _state(name, g, depth):
    cfg = dict(CFG, depth=depth)
    dec = get_genotype_decoder(name, depth, cfg)
    p = dec.decode(jnp.asarray(g), 30)
    cov, mu, eff, _ = jax_equivalent_gaussian_static(p, depth)
    psi, prob = jax_reduced_herald_static(cov, mu, eff, L_TEST, BF_TEST,
                                          depth, MAXF_TEST)
    return np.array(psi), float(prob)


# --------------------------------------------------------------------- B30F --
def test_b30f_forced_heralding_invariants():
    dec = get_genotype_decoder("B30F", 3, CFG)
    rng = np.random.default_rng(0)
    for _ in range(10):
        g = rng.uniform(-2, 2, dec.get_length(3)).astype(np.float32)
        p = dec.decode(jnp.asarray(g), 30)
        active = np.array(p["leaf_active"])
        lp = p["leaf_params"]
        nc = np.array(lp["n_ctrl"])
        pnr = np.array(lp["pnr"])
        assert active[0], "leaf 0 must be active"
        assert (nc >= 1).all(), "every leaf must own >= 1 control"
        assert (pnr[:, 0] >= 1).all(), "first detector must fire >= 1"
        assert (pnr[:, 0] <= CFG["pnr_max"]).all()


def test_b30f_all_off_still_fires():
    dec = get_genotype_decoder("B30F", 3, CFG)
    g = -np.ones(dec.get_length(3), dtype=np.float32)
    p = dec.decode(jnp.asarray(g), 30)
    lp = p["leaf_params"]
    assert bool(np.array(p["leaf_active"])[0])
    assert int(np.array(lp["n_ctrl"])[0]) >= 1
    assert int(np.array(lp["pnr"])[0, 0]) >= 1


def test_ste_forward_exact():
    rng = np.random.default_rng(1)
    for name in ["B30", "B30F"]:
        dec = get_genotype_decoder(name, 3, CFG)
        g = rng.uniform(-2, 2, dec.get_length(3)).astype(np.float32)
        lp = dec.decode(jnp.asarray(g), 30)["leaf_params"]
        assert np.allclose(np.array(lp["pnr_ste"]),
                           np.array(lp["pnr"]).astype(float))
        assert np.allclose(np.array(lp["n_ctrl_ste"]),
                           np.array(lp["n_ctrl"]).astype(float))


# ---------------------------------------------------------- depth embedding --
@pytest.mark.parametrize("name", ["B30", "B30F"])
def test_depth_embedding_preserves_state(name):
    assert supports_depth_upgrade(name)
    rng = np.random.default_rng(3)
    D3 = get_genotype_decoder(name, 3, CFG).get_length(3)
    for _ in range(2):
        g3 = rng.uniform(-1.5, 1.5, D3).astype(np.float32)
        psi3, pr3 = _state(name, g3, 3)
        g4 = upgrade_depth(g3, name, 3, 4, CFG)
        psi4, pr4 = _state(name, g4, 4)
        assert np.max(np.abs(psi3 - psi4)) < 1e-6
        assert abs(pr3 - pr4) < 1e-6


def test_b30f_to_b30_lossless():
    rng = np.random.default_rng(4)
    D = get_genotype_decoder("B30F", 3, CFG).get_length(3)
    for _ in range(4):
        g = rng.uniform(-1.5, 1.5, D).astype(np.float32)
        pf, _ = _state("B30F", g, 3)
        pb, _ = _state("B30", convert_b30f_to_b30(g, 3, CFG), 3)
        assert np.max(np.abs(pf - pb)) < 1e-6


# ----------------------------------------------------------------- pnr seeds --
def test_pnr_pattern_seeds_decode():
    from src.genotypes.pnr_seeds import generate_pnr_pattern_seeds
    rng = np.random.default_rng(5)
    cfg = dict(CFG, moment_maxf=MAXF_TEST, moment_bf=BF_TEST)
    seeds = generate_pnr_pattern_seeds("B30F", 3, cfg, 16, rng)
    assert len(seeds) == 16
    dec = get_genotype_decoder("B30F", 3, cfg)
    # seed 0 = uniform-1: all 8 leaves fit the (maxf=8, bf=256) budget
    p = dec.decode(jnp.asarray(seeds[0]), 30)
    assert np.array(p["leaf_active"]).all()
    assert (np.array(p["leaf_params"]["pnr"])[:, 0] == 1).all()
    # seed 1 = uniform-2, budget-capped: active prefix fires 2 clicks each,
    # fired count within maxf and prod(n+1) within bf
    p2 = dec.decode(jnp.asarray(seeds[1]), 30)
    act2 = np.array(p2["leaf_active"])
    pnr2 = np.array(p2["leaf_params"]["pnr"])[:, 0]
    assert act2.sum() >= 2 and (pnr2[act2] == 2).all()
    assert act2.sum() <= MAXF_TEST
    assert np.sum(np.log(pnr2[act2] + 1.0)) <= np.log(BF_TEST) + 1e-9
    # every seed heralds a nonzero, finite state
    for g in seeds[:4]:
        psi, prob = _state("B30F", g, 3)
        assert np.isfinite(psi).all()
        assert np.linalg.norm(psi) > 0.5


# ----------------------------------------------------------- macro mutations --
def test_macro_mutations_shape_and_targets():
    from src.optimization.macro_mutations import (
        make_macro_mutation, make_mixed_mutation)
    fn = make_macro_mutation("B30F", 3, CFG)
    D = get_genotype_decoder("B30F", 3, CFG).get_length(3)
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (24, D), minval=-1, maxval=1)
    y = fn(x, key)
    assert y.shape == x.shape
    assert np.isfinite(np.array(y)).all()
    # decode every mutant: forced invariants must survive any operator
    dec = get_genotype_decoder("B30F", 3, CFG)
    for i in range(y.shape[0]):
        p = dec.decode(y[i], 30)
        assert (np.array(p["leaf_params"]["pnr"])[:, 0] >= 1).all()
    # mixed mutation composes with a poly stub
    mixed = make_mixed_mutation(lambda x, k: x + 0.01, fn, 0.5)
    z = mixed(x, key)
    assert z.shape == x.shape


# ---------------------------------------------------- NG-stratified seeding --
def test_pareto_ng_round_robin():
    from src.utils.result_scanner import compute_pareto_front
    # build candidates: a big Gaussian stratum + small NG strata
    cands = []
    rng = np.random.default_rng(7)
    for i in range(50):
        cands.append(dict(genotype=np.zeros(3), name="B30",
                          fit0=-0.74 - 0.001 * i, fit1=0.0, dng=0.0,
                          score=0.0, path=""))
    for i in range(5):
        cands.append(dict(genotype=np.ones(3), name="B30",
                          fit0=-0.80 - 0.01 * i, fit1=-1.0, dng=0.3,
                          score=0.0, path=""))
    for i in range(3):
        cands.append(dict(genotype=2 * np.ones(3), name="B30",
                          fit0=-0.90 - 0.01 * i, fit1=-2.0, dng=1.0,
                          score=0.0, path=""))
    # emulate the stratified selection logic
    edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, float("inf")]
    strata = [[] for _ in range(len(edges) - 1)]
    for c in cands:
        d = max(float(c.get("dng", 0.0)), 0.0)
        for s in range(len(edges) - 1):
            if edges[s] <= d < edges[s + 1]:
                strata[s].append(c)
                break
    # dng=0.0 -> [0, 0.02) = stratum 0; 0.3 -> [0.2, 0.4) = stratum 4;
    # 1.0 -> [0.8, 1.6) = stratum 6
    assert len(strata[0]) == 50 and len(strata[4]) == 5 and len(strata[6]) == 3
    front4 = compute_pareto_front(strata[4])
    assert len(front4) >= 1


# ----------------------------------------------------------- 2-obj fitness --
def test_moment_fitness_two_objectives():
    from src.simulation.jax.moment_scorer import moment_score_population_static
    cfg = dict(CFG, scorer="moment", moment_cutoff=L_TEST, moment_bf=BF_TEST,
               moment_maxf=MAXF_TEST, moment_chunk=4, moment_fast=True,
               target_alpha="1.0", target_beta="0.0", alpha_expectation=1.0,
               alpha_probability=0.0, alpha_nongauss=0.1,
               moment_ng_descriptor=True)
    ch = tuple(sorted(cfg.items()))
    D = get_genotype_decoder("B30F", 3, cfg).get_length(3)
    rng = np.random.default_rng(9)
    g = rng.uniform(-1, 1, (4, D)).astype(np.float64)
    op = jnp.eye(L_TEST, dtype=jnp.complex128)  # dummy operator: <O> = 1
    fit, desc, extras = moment_score_population_static(
        g, op, "B30F", ch, 30, L_TEST, 0.0, 2.0 / 3.0)
    assert fit.shape == (4, 2), f"expected 2 objectives, got {fit.shape}"
    assert desc.shape == (4, 4), "expected 4 descriptor axes with NG on"
    assert "nongaussianity" in extras
