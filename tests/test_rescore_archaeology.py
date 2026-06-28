"""Tests for scripts/rescore_all_experiments.py (PROMPT s7).

Run only these:  pytest tests/ -k rescore

Covers: the target resolver + empirical |beta|->beta cross-validation, round-trip
consistency of the exact re-score against a stored non-fast moment run,
independent-fidelity (moment scorer vs thewalrus reduced_herald), the Hudson /
Wigner-negativity sentinels and the fake_subgaussian gate, robustness (corrupt
pickle / missing target_beta / wrong-length genotype are logged & skipped, never
crash), and determinism of all_solutions across two runs.
"""
import os
import sys
import glob
import json
import pickle

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax = pytest.importorskip("jax")
pytest.importorskip("thewalrus")
jax.config.update("jax_enable_x64", True)

import rescore_all_experiments as R  # noqa: E402
from src.utils.result_manager import SimpleRepertoire  # noqa: E402


# --------------------------------------------------------------------------- #
# shared empirical maps learned from the real archive (built once)             #
# --------------------------------------------------------------------------- #
def _all_configs_by_group():
    from collections import defaultdict
    cbg = defaultdict(list)
    for root in R.DEFAULT_ROOTS:
        rabs = os.path.join(REPO, root)
        for cf in glob.glob(os.path.join(rabs, "*", "*", "config.json")):
            group = os.path.basename(os.path.dirname(os.path.dirname(cf)))
            try:
                cbg[group].append(json.load(open(cf)))
            except Exception:
                pass
    return cbg


@pytest.fixture(scope="module")
def empirical_maps():
    cbg = _all_configs_by_group()
    if not cbg:
        pytest.skip("no experiment configs found")
    beta_map, alpha_map, contradictions = R.build_empirical_maps(cbg)
    return beta_map, alpha_map, contradictions, cbg


# --------------------------------------------------------------------------- #
# 1. target resolver + empirical map cross-validation                          #
# --------------------------------------------------------------------------- #
def test_rescore_as_complex_parser():
    assert R.as_complex("(1+1j)") == complex(1, 1)
    assert R.as_complex("1+1i") == complex(1, 1)      # i -> j
    assert R.as_complex(" 0.0 ") == 0
    assert R.as_complex(2.0) == complex(2, 0)
    assert R.as_complex(None) is None
    assert R.as_complex("") is None
    assert R.as_complex("garbage") is None


def test_rescore_group_name_to_target(empirical_maps):
    beta_map, alpha_map, _contra, _cbg = empirical_maps
    a, b, reason = R.target_from_group("B30_c30_a2p73_b1p41", beta_map, alpha_map)
    assert reason is None
    assert abs(a - 2.7320508) < 1e-4 and b == complex(1, 1)

    a, b, reason = R.target_from_group("0_c4_a2p00_b0p00", beta_map, alpha_map)
    assert reason is None
    assert abs(a - 2.0) < 1e-9 and b == 0          # b0p00 -> 0 is legitimate

    a, b, reason = R.target_from_group("00B_c30_a1p00_b1p00", beta_map, alpha_map)
    assert reason is None
    assert abs(a - 1.0) < 1e-9 and b == complex(1, 0)


def test_rescore_empirical_beta_map_no_contradictions(empirical_maps):
    beta_map, alpha_map, contradictions, cbg = empirical_maps
    # the learned map must agree with EVERY config that carries an explicit target
    assert not contradictions, f"target-map contradictions: {contradictions}"
    n_checked = 0
    for group, cfgs in cbg.items():
        info = R.parse_group_name(group)
        if not info:
            continue
        for cfg in cfgs:
            b = R.as_complex(cfg.get("target_beta"))
            if b is not None:
                assert abs(beta_map[info["b_str"]] - b) < 1e-6, (group, b)
                n_checked += 1
            a = R.as_complex(cfg.get("target_alpha"))
            if a is not None:
                assert abs(alpha_map[info["a_str"]] - a) < 1e-6, (group, a)
    assert n_checked > 100          # sanity: we actually cross-checked many configs


def test_rescore_unmapped_beta_is_unresolved():
    # a b-code with no example anywhere and not zero -> unresolved (never guessed)
    a, b, reason = R.target_from_group("X_c30_a1p00_b9p99", {}, {"1p00": complex(1, 0)})
    assert a is None and reason and reason.startswith("target_unresolved")


# --------------------------------------------------------------------------- #
# helpers for round-trip / fidelity (one real non-fast moment run)             #
# --------------------------------------------------------------------------- #
def _find_moment_nonfast_run():
    for root in R.DEFAULT_ROOTS:
        for cf in sorted(glob.glob(os.path.join(REPO, root, "*", "*", "config.json"))):
            try:
                c = json.load(open(cf))
            except Exception:
                continue
            if c.get("scorer") == "moment" and not c.get("moment_fast"):
                return os.path.dirname(cf), c
    return None, None


@pytest.fixture(scope="module")
def moment_run(empirical_maps):
    beta_map, alpha_map, _c, _cbg = empirical_maps
    rundir, cfg = _find_moment_nonfast_run()
    if rundir is None:
        pytest.skip("no non-fast moment-scored run found")
    group = os.path.basename(os.path.dirname(rundir))
    a, b, _src, reason = R.resolve_target(cfg, group, beta_map, alpha_map)
    assert reason is None, f"could not resolve target for {group}: {reason}"
    gen, fit, des = R.load_run_arrays(os.path.join(rundir, "results.pkl"))
    design = cfg.get("genotype")
    depth = int(cfg.get("depth") or 3)
    maxf = int(cfg.get("moment_maxf") or 8)
    eng = R.RescoreEngine(design, depth, maxf, cfg)
    return dict(eng=eng, gen=gen, fit=fit, des=des, a=a, b=b, depth=depth,
                L_hi=120, L_lo=50, cfg=cfg)


# --------------------------------------------------------------------------- #
# 2. round-trip consistency: exact re-score reproduces the stored <O>          #
#    The optimizer stored <O> computed at the run's OWN moment_cutoff, so the   #
#    faithful reproducibility check re-scores at THAT L (the moment scorer is   #
#    bit-identical at equal L).  Comparing at a different L would instead probe  #
#    Fock convergence -- which is exactly what the l_truncation filter is for,  #
#    and which legitimately fails for hard high-alpha targets.                  #
# --------------------------------------------------------------------------- #
def test_rescore_roundtrip_matches_stored(moment_run):
    m = moment_run
    eng, gen, fit, cfg = m["eng"], m["gen"], m["fit"], m["cfg"]
    L_run = int(cfg.get("moment_cutoff") or m["L_hi"])
    O = R.O_at(m["a"], m["b"], L_run)
    bf = int(cfg.get("moment_bf") or 8192)
    valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0]
    order = valid[np.argsort(fit[valid, 0])[::-1]][:24]   # best stored cells
    psi, _, _, _ = eng.score(gen[order].astype(np.float32), L_run, bf)
    n_ok = 0
    worst = 0.0
    for j, k in enumerate(order):
        exp = R.expectation(psi[j], O)
        stored = -fit[k, 0]
        d = abs(exp - stored)
        worst = max(worst, d)
        # exact re-score at the run's own cutoff must reproduce the stored value
        assert d < 2e-3, (k, exp, stored, d)
        n_ok += 1
    assert n_ok >= 5, f"too few cells to validate round-trip ({n_ok})"


# --------------------------------------------------------------------------- #
# 3. independent-fidelity: moment scorer vs thewalrus reduced_herald           #
# --------------------------------------------------------------------------- #
def test_rescore_independent_fidelity(moment_run):
    from frontend.gbs_optimizer import reduced_herald
    m = moment_run
    eng, gen, fit = m["eng"], m["gen"], m["fit"]
    valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0]
    order = valid[np.argsort(fit[valid, 0])[::-1]][:12]
    checked = 0
    for k in order:
        g = gen[k].astype(np.float32)
        psi_m, _, _, _ = eng.score(g[None], m["L_hi"], 8192)
        psi_m = psi_m[0]
        cov, mu, eff = eng.equivalent_gaussian(g)
        ncontrol = eff.shape[0]
        nvec = [int(round(x)) for x in eff]
        try:
            psi_i, _ = reduced_herald(cov, mu, 0, list(range(1, 1 + ncontrol)),
                                      nvec, cutoff=m["L_hi"])
        except Exception:
            continue            # over-budget herald -> skip (handled in the sweep too)
        psi_i = np.asarray(psi_i).ravel()
        L = min(len(psi_i), len(psi_m))
        fid = abs(np.vdot(psi_i[:L], psi_m[:L])) ** 2
        if np.isfinite(fid) and fid > 1e-6:        # well-defined herald
            assert fid >= 0.999, (k, fid)
            checked += 1
    assert checked >= 3, f"too few well-defined heralds to check ({checked})"


# --------------------------------------------------------------------------- #
# 4. Hudson sentinels + fake_subgaussian gate                                  #
# --------------------------------------------------------------------------- #
def _fock(n, dim=40):
    v = np.zeros(dim, complex); v[n] = 1.0
    return v


def _squeezed_vacuum(r, dim=80):
    from math import factorial, cosh, tanh, sqrt
    v = np.zeros(dim, complex)
    for n in range(dim // 2):
        v[2 * n] = ((-tanh(r)) ** n) * sqrt(factorial(2 * n)) / (
            2 ** n * factorial(n)) / sqrt(cosh(r))
    return v / np.linalg.norm(v)


def _coherent(alpha, dim=80):
    from math import factorial, exp, sqrt
    v = np.array([exp(-abs(alpha) ** 2 / 2) * (alpha ** n) / sqrt(factorial(n))
                  for n in range(dim)], complex)
    return v / np.linalg.norm(v)


def test_rescore_wigner_sentinels():
    # Gaussian pure states -> (essentially) non-negative Wigner
    nv_sq = R.wigner_negative_volume(_squeezed_vacuum(0.6), grid=25, span=5.0)
    nv_coh = R.wigner_negative_volume(_coherent(1.0 + 0.5j), grid=25, span=5.0)
    assert nv_sq < 1e-2, nv_sq
    assert nv_coh < 1e-2, nv_coh
    # Fock |1> -> clearly Wigner-negative
    nv_f1 = R.wigner_negative_volume(_fock(1), grid=25, span=5.0)
    assert nv_f1 > 0.1, nv_f1
    # and the negativity ordering is unambiguous
    assert nv_f1 > 10 * max(nv_sq, nv_coh, 1e-6)


def test_rescore_wigner_gaussian_meets_limit():
    # a Gaussian pure state cannot beat the Gaussian limit: <O> >= gaussian_limit
    a, b = complex(1, 0), complex(1, 1)
    glim, _gs, O = R.target_refs(a, b, 80)
    psi = _squeezed_vacuum(0.5, dim=80)
    exp = R.expectation(psi, O)
    assert exp >= glim - 1e-6, (exp, glim)


def test_rescore_fake_subgaussian_gate():
    glim = 1.0
    # hand-made "below the limit but Wigner-positive" -> IMPOSSIBLE -> flagged
    is_art, reasons = R.classify_artifact(
        exp_lo=0.8, exp_hi=0.8, P=1e-3, herald_norm=1.0, fired_modes=1,
        fp_budget=4, indep_fidelity=1.0, wigner_negvol=0.0, gaussian_limit=glim)
    assert is_art and "fake_subgaussian" in reasons
    # genuine sub-Gaussian (below limit AND Wigner-negative) -> NOT flagged
    is_art, reasons = R.classify_artifact(
        exp_lo=0.8, exp_hi=0.8, P=1e-3, herald_norm=1.0, fired_modes=1,
        fp_budget=4, indep_fidelity=1.0, wigner_negvol=0.5, gaussian_limit=glim)
    assert "fake_subgaussian" not in reasons
    # above the limit, Wigner-positive -> NOT a fake_subgaussian (not claiming advantage)
    is_art, reasons = R.classify_artifact(
        exp_lo=1.2, exp_hi=1.2, P=1e-3, herald_norm=1.0, fired_modes=1,
        fp_budget=4, indep_fidelity=1.0, wigner_negvol=0.0, gaussian_limit=glim)
    assert "fake_subgaussian" not in reasons


def test_rescore_classify_other_filters():
    base = dict(exp_lo=1.2, exp_hi=1.2, P=1e-3, herald_norm=1.0, fired_modes=1,
                fp_budget=4, indep_fidelity=1.0, wigner_negvol=np.nan, gaussian_limit=1.0)
    assert R.classify_artifact(**base)[0] is False
    assert "bad_prob" in R.classify_artifact(**{**base, "P": 0.0})[1]
    assert "over_budget" in R.classify_artifact(**{**base, "fired_modes": 99})[1]
    assert "over_budget" in R.classify_artifact(**{**base, "fp_budget": 1 << 20})[1]
    assert "unnormalized" in R.classify_artifact(**{**base, "herald_norm": 0.5})[1]
    assert "l_truncation" in R.classify_artifact(**{**base, "exp_lo": 0.5})[1]
    assert "scorer_mismatch" in R.classify_artifact(
        **{**base, "indep_fidelity": 0.5, "has_fidelity": True})[1]
    # fidelity ignored when not on the checked subsample
    assert "scorer_mismatch" not in R.classify_artifact(
        **{**base, "indep_fidelity": 0.5, "has_fidelity": False})[1]


# --------------------------------------------------------------------------- #
# 5/6. robustness + determinism (end-to-end on a crafted tiny tree)            #
# --------------------------------------------------------------------------- #
def _write_run(root, group, run, gen, fit, des, cfg, corrupt=False):
    d = os.path.join(root, group, run)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    if corrupt:
        with open(os.path.join(d, "results.pkl"), "wb") as f:
            f.write(b"\x80\x04 not a real pickle \xff\xff garbage")
    else:
        rep = SimpleRepertoire(np.asarray(gen, np.float32), np.asarray(fit, float),
                               np.asarray(des, float))
        with open(os.path.join(d, "results.pkl"), "wb") as f:
            pickle.dump({"repertoire": rep, "history": {}, "centroids": None,
                         "timestamp": "t", "history_fronts": []}, f)


def _crafted_tree(tmp):
    """A tiny roots tree with: a good 00B run, a good run whose config DROPS
    target_beta, a corrupt pickle, and a wrong-length genotype."""
    root = os.path.join(tmp, "experiments")
    cfg00b = dict(genotype="00B", depth=3, modes=3, pnr_max=15, cutoff=30,
                  r_scale=1.868551121099462, d_scale=2.23606797749979,
                  hx_scale=1.3693063937629153, window=0.1,
                  target_alpha="1.0", target_beta="(1+1j)")
    glen = R.make_decoder("00B", 3, cfg00b).get_length(3)        # 83
    rng = np.random.default_rng(0)
    gen = rng.standard_normal((6, glen))
    fit = np.column_stack([-rng.uniform(1.0, 1.5, 6), -rng.uniform(1, 6, 6),
                           -rng.uniform(1, 4, 6), -rng.uniform(0, 8, 6)])
    des = np.column_stack([rng.integers(1, 4, 6), rng.integers(0, 4, 6),
                           rng.integers(0, 8, 6)]).astype(float)
    group = "00B_c30_a1p00_b1p41"
    _write_run(root, group, "20260101-000001_p8_i10", gen, fit, des, cfg00b)
    # (b) same group, config MISSING target_beta -> resolved via empirical/folder map
    cfg_nob = {k: v for k, v in cfg00b.items() if k != "target_beta"}
    _write_run(root, group, "20260101-000002_p8_i10", gen, fit, des, cfg_nob)
    # (a) corrupt pickle -> unloadable
    _write_run(root, group, "20260101-000003_p8_i10", gen, fit, des, cfg00b, corrupt=True)
    # (c) wrong-length genotype -> undecodable(length_mismatch)
    gen_bad = rng.standard_normal((4, glen - 10))
    fit_bad = np.column_stack([-rng.uniform(1, 1.5, 4), -rng.uniform(1, 6, 4),
                               -rng.uniform(1, 4, 4), -rng.uniform(0, 8, 4)])
    des_bad = np.zeros((4, 3))
    _write_run(root, group, "20260101-000004_p8_i10", gen_bad, fit_bad, des_bad, cfg00b)
    return root


def _make_args(roots, out):
    args = R.build_argparser().parse_args(
        ["--roots", *roots, "--out", out, "--per-run-cap", "6",
         "--fidelity-subsample", "1", "--l-high", "60", "--l-search", "40",
         "--wigner-grid", "13", "--progress-every", "100"])
    return args


def test_rescore_robustness_and_accounting(tmp_path):
    root = _crafted_tree(str(tmp_path))
    out = os.path.join(str(tmp_path), "out")
    df, summary = R.run_sweep(_make_args([root], out))
    # never crashed; accounting adds up over all 4 discovered runs
    assert summary["n_runs"] == 4
    assert (summary["n_rescored"] + summary["n_undecodable"]
            + summary["n_unloadable"]) == 4
    assert summary["n_rescored"] == 2            # the two good runs
    assert summary["n_unloadable"] == 1          # corrupt pickle
    assert summary["n_undecodable"] == 1         # wrong-length genotype
    import pandas as pd
    undec = pd.read_csv(os.path.join(out, "undecodable.csv"))
    reasons = " ".join(undec["reason"].astype(str))
    assert "length_mismatch" in reasons
    assert any(t in reasons for t in ("UnpicklingError", "no repertoire", "Error"))
    assert os.path.exists(os.path.join(out, "REPORT.md"))


def test_rescore_determinism(tmp_path):
    root = _crafted_tree(str(tmp_path))
    out1 = os.path.join(str(tmp_path), "o1")
    out2 = os.path.join(str(tmp_path), "o2")
    df1, _ = R.run_sweep(_make_args([root], out1))
    df2, _ = R.run_sweep(_make_args([root], out2))
    cols = ["root", "group", "run", "cell_idx", "exp_hi", "exp_lo", "prob",
            "herald_norm", "is_artifact", "artifact_reason"]
    a = df1[cols].sort_values(["group", "run", "cell_idx"]).reset_index(drop=True)
    b = df2[cols].sort_values(["group", "run", "cell_idx"]).reset_index(drop=True)
    import pandas as pd
    pd.testing.assert_frame_equal(a, b)
