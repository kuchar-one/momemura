#!/usr/bin/env python3
"""rescore_all_experiments.py -- data archaeology over EVERY past optimization run.

The repo holds thousands of optimization runs (under ``experiments/`` and the
``output_old*/experiments/`` archives), many polluted by Fock-truncation
artifacts, ``--moment-fast`` prob=1 placeholders, or dropped ``target_beta``
metadata.  This script re-decodes and **re-scores every recoverable result with
the current, exact (truncation-free) moment pipeline**, applies physical-validity
filters (including a Hudson / Wigner-negativity gate), throws out the artifacts,
and surfaces the genuine artifact-free optima per target group.

It generalizes ``scripts/validate_moment_archive.py`` (the single-run reference
validator) to the whole archive: discover -> load -> resolve target -> select
cells -> re-score at L_lo & L_hi (exact) -> filter -> aggregate per target group.

Always 64-bit (the moment recurrence underflows in complex64):

    JAX_ENABLE_X64=1 python scripts/rescore_all_experiments.py \
        --groups '*_a1p00_b1p41' --per-run-cap 64 --limit 40 --out recompute/

Full run (every selected cell of every run):

    JAX_ENABLE_X64=1 python scripts/rescore_all_experiments.py --per-run-cap 0

Deliverables land under ``--out`` (default ``recompute/``): ``all_solutions.parquet``
(+ ``.csv`` sample), per-group artifact-free Pareto fronts, best genuine states,
``REPORT.md``, ``undecodable.csv``, and plots.  Read-only w.r.t. the run data.
"""
from __future__ import annotations

# --- x64 BEFORE jax is imported anywhere (the scorer is only correct in 64-bit) ---
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", ""))  # respect GPU if present

import sys
import re
import glob
import json
import time
import pickle
import argparse
import functools
import traceback
from collections import defaultdict, Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Documented run roots (PROMPT s1).  Each is a <root>/<group>/<run>/results.pkl tree.
DEFAULT_ROOTS = [
    "experiments",
    "output_old/experiments",
    "output_oldold/experiments",
    "output/experiments",
]

GROUP_RE = re.compile(r"^(?P<design>.+)_c(?P<cutoff>\d+)_a(?P<a>[0-9p]+)_b(?P<b>[0-9p]+)$")


# --------------------------------------------------------------------------- #
# target parsing / resolution (PROMPT s3.1)                                    #
# --------------------------------------------------------------------------- #
def as_complex(v):
    """Robustly parse a config/CLI target into a Python complex, or None if
    absent/unparseable.  Handles bare numbers, parenthesised reprs like
    '(1+1j)', 'i'->'j', and stray spaces (mirror validate_moment_archive)."""
    if v is None:
        return None
    if isinstance(v, (int, float, complex)):
        return complex(v)
    s = str(v).strip().replace(" ", "").replace("i", "j")
    if s == "" or s.lower() == "none":
        return None
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    try:
        return complex(s)
    except ValueError:
        return None


def num_from_pstr(s):
    """'2p73' -> 2.73 ; '1p00' -> 1.0 ; '0p00' -> 0.0  (folder-name decimal code)."""
    try:
        return float(str(s).replace("p", "."))
    except ValueError:
        return None


def parse_group_name(group):
    """Split '<design>_c<cutoff>_a<astr>_b<bstr>' -> dict or None."""
    m = GROUP_RE.match(group)
    if not m:
        return None
    return {
        "design": m["design"],
        "cutoff": int(m["cutoff"]),
        "a_str": m["a"],
        "b_str": m["b"],
    }


def build_empirical_maps(configs_by_group):
    """Learn the folder b-string -> complex-beta (and a-string -> complex-alpha)
    map empirically from EVERY config that DOES carry an explicit target
    (PROMPT s3.1).  Returns (beta_map, alpha_map, contradictions).

    ``configs_by_group``: dict group_name -> list of config dicts.
    Contradictions (one folder code mapping to two different explicit values)
    are reported so the resolver/test can flag them rather than guess.
    """
    beta_votes = defaultdict(Counter)
    alpha_votes = defaultdict(Counter)
    for group, cfgs in configs_by_group.items():
        info = parse_group_name(group)
        if not info:
            continue
        for cfg in cfgs:
            b = as_complex(cfg.get("target_beta"))
            if b is not None:
                beta_votes[info["b_str"]][b] += 1
            a = as_complex(cfg.get("target_alpha"))
            if a is not None:
                alpha_votes[info["a_str"]][a] += 1

    contradictions = []
    beta_map, alpha_map = {}, {}
    for bstr, votes in beta_votes.items():
        # contradiction = two values that disagree beyond rounding (|beta| ties allowed
        # only if numerically equal). Pick the plurality value; record disagreements.
        distinct = list(votes)
        if len(distinct) > 1 and not _all_close(distinct):
            contradictions.append(("beta", bstr, dict(votes)))
        beta_map[bstr] = votes.most_common(1)[0][0]
    for astr, votes in alpha_votes.items():
        distinct = list(votes)
        if len(distinct) > 1 and not _all_close(distinct):
            contradictions.append(("alpha", astr, dict(votes)))
        alpha_map[astr] = votes.most_common(1)[0][0]
    return beta_map, alpha_map, contradictions


def _all_close(vals, tol=1e-6):
    vals = list(vals)
    return all(abs(v - vals[0]) <= tol for v in vals)


def resolve_target(cfg, group, beta_map, alpha_map):
    """Resolve (alpha, beta) for one run (PROMPT s3.1).  Returns
    (alpha, beta, source, reason): on success reason is None; on failure
    (alpha, beta) is (None, None) and reason is 'target_unresolved:<why>'.

    Priority: explicit config value (robust parse) > empirical folder-code map >
    direct folder-code decimal decode (alpha only; beta phase is NOT guessable
    from |beta| so an unmapped b-code is left unresolved)."""
    info = parse_group_name(group)
    src = []

    # alpha: config > empirical a-map > folder decimal
    a = as_complex(cfg.get("target_alpha"))
    if a is not None:
        src.append("alpha=config")
    elif info and info["a_str"] in alpha_map:
        a = alpha_map[info["a_str"]]; src.append("alpha=empirical")
    elif info:
        v = num_from_pstr(info["a_str"])
        if v is not None:
            a = complex(v); src.append("alpha=folder")

    # beta: config > empirical b-map (folder b-code).  An unmapped b-code -> unresolved
    # (|beta| alone loses the phase; b0p00 -> 0 is legitimate and IS in the map).
    b = as_complex(cfg.get("target_beta"))
    if b is not None:
        src.append("beta=config")
    elif info and info["b_str"] in beta_map:
        b = beta_map[info["b_str"]]; src.append("beta=empirical")
    elif info and num_from_pstr(info["b_str"]) == 0.0:
        b = 0j; src.append("beta=zero")  # b0p00 with no explicit example anywhere

    if a is None:
        return None, None, None, "target_unresolved:no_alpha"
    if b is None:
        bcode = info["b_str"] if info else "?"
        return None, None, None, f"target_unresolved:no_beta_for_b{bcode}"
    return a, b, "+".join(src), None


def target_from_group(group, beta_map, alpha_map):
    """Convenience: resolve a full group name '<d>_c<c>_a<astr>_b<bstr>' to
    (alpha, beta) using only the empirical maps / folder code (no config)."""
    a, b, _src, reason = resolve_target({}, group, beta_map, alpha_map)
    return a, b, reason


# --------------------------------------------------------------------------- #
# robust repertoire loading (PROMPT s3.4)                                       #
# --------------------------------------------------------------------------- #
class _GenericTypeUnpickler(pickle.Unpickler):
    """Map any unimportable class (e.g. qdax ``MapElitesRepertoire`` /
    ``MOMERepertoire`` when qdax isn't installed) to a fresh *type* stand-in, so
    NEWOBJ/BUILD still reconstruct the object's attribute dict.  This recovers
    the genotypes/fitnesses/descriptors arrays without qdax (jax IS required, to
    unpickle the array leaves)."""
    _stubs = {}

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            key = (module, name)
            if key not in self._stubs:
                self._stubs[key] = type(str(name), (object,), {})
            return self._stubs[key]


def _extract_arrays(rep):
    """Pull (genotypes, fitnesses, descriptors) off a repertoire-like object and
    flatten to 2-D (collapsing any MOME pareto-slot axis).  Raises on absence."""
    gen = getattr(rep, "genotypes", None)
    fit = getattr(rep, "fitnesses", None)
    des = getattr(rep, "descriptors", None)
    if gen is None or fit is None:
        raise ValueError("repertoire missing genotypes/fitnesses")
    gen = np.asarray(gen, dtype=np.float64)
    fit = np.asarray(fit, dtype=np.float64)
    gen = gen.reshape(-1, gen.shape[-1])
    fit = fit.reshape(-1, fit.shape[-1])
    if des is not None:
        des = np.asarray(des, dtype=np.float64).reshape(-1, np.asarray(des).shape[-1])
    else:
        des = np.full((gen.shape[0], 3), np.nan)
    if des.shape[0] != gen.shape[0]:
        des = np.full((gen.shape[0], max(des.shape[1], 3)), np.nan)
    return gen, fit, des


def load_run_arrays(pkl_path):
    """Return (gen2d, fit2d, des2d) for a run, trying the tolerant pareto_report
    loader first, then the generic-type unpickler (MOME without qdax).  Raises
    if neither recovers arrays (-> caller logs 'unloadable')."""
    import pareto_report as pr
    last = None
    try:
        rep = pr.load_repertoire(pkl_path)
        if rep is not None:
            return _extract_arrays(rep)
    except Exception as e:  # noqa: BLE001
        last = e
    # fallback: generic stand-in types
    with open(pkl_path, "rb") as f:
        data = _GenericTypeUnpickler(f).load()
    rep = data.get("repertoire") if isinstance(data, dict) else data
    if rep is None:
        raise ValueError(f"no repertoire in pickle (last={last!r})")
    return _extract_arrays(rep)


# --------------------------------------------------------------------------- #
# cell selection (PROMPT s4.3)                                                  #
# --------------------------------------------------------------------------- #
def select_cells(fit, des, per_run_cap):
    """Pick which cells to re-score: the run's non-dominated front over
    (<O>, logP) UNION the best-<O> cell per descriptor bin (MAP-Elites elites).
    Deterministic; returns (indices, n_valid, n_skipped).

    per_run_cap<=0 means 'no cap' (take all selected)."""
    n_obj = fit.shape[1]
    exp = -fit[:, 0]                       # <O>  (lower better)
    logp = fit[:, 1] if n_obj > 1 else np.zeros(len(fit))
    valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0]
    if valid.size == 0:
        return np.array([], dtype=int), 0, 0

    # (a) non-dominated front: minimize <O>, maximize logP  (== minimize -logP)
    objs = np.column_stack([exp[valid], -logp[valid]])
    nd = []
    for j in range(len(valid)):
        dom = np.all(objs <= objs[j], axis=1) & np.any(objs < objs[j], axis=1)
        if not np.any(dom):
            nd.append(valid[j])
    sel = set(int(i) for i in nd)

    # (b) best-<O> elite per descriptor bin
    bins = defaultdict(list)
    for i in valid:
        if des is not None and des.shape[1] >= 1 and np.all(np.isfinite(des[i])):
            key = tuple(np.round(des[i]).astype(int).tolist())
        else:
            key = ("nodesc",)
        bins[key].append(int(i))
    for _key, idxs in bins.items():
        best = min(idxs, key=lambda k: exp[k])
        sel.add(best)

    sel = np.array(sorted(sel), dtype=int)
    n_skipped = 0
    if per_run_cap and per_run_cap > 0 and sel.size > per_run_cap:
        # keep the best-<O> cells (deterministic), drop the rest
        order = sel[np.argsort(exp[sel], kind="stable")]
        n_skipped = int(sel.size - per_run_cap)
        sel = np.sort(order[:per_run_cap])
    return sel, int(valid.size), n_skipped


# --------------------------------------------------------------------------- #
# decoder / depth handling                                                      #
# --------------------------------------------------------------------------- #
def make_decoder(design, depth, cfg):
    from src.genotypes.genotypes import get_genotype_decoder
    return get_genotype_decoder(design, depth=depth, config=cfg)


def infer_depth(design, cfg, glen, candidates=range(1, 8)):
    """Find depth d whose decoder length == genotype length (PROMPT s3.3).
    Returns d or None."""
    for d in candidates:
        try:
            dec = make_decoder(design, d, cfg)
            if dec.get_length(d) == glen:
                return d
        except Exception:  # noqa: BLE001
            continue
    return None


# --------------------------------------------------------------------------- #
# batched exact re-scoring engine (one compiled graph per decoder bucket)       #
# --------------------------------------------------------------------------- #
class RescoreEngine:
    """Holds the jitted, vmapped exact scorer for one decoder bucket
    (design, depth, modes, pnr_max, scales, maxf, cutoff)."""

    def __init__(self, design, depth, maxf, cfg):
        import jax
        import jax.numpy as jnp
        from src.simulation.jax.moment_scorer import (
            jax_equivalent_gaussian_static, jax_reduced_herald_static,
            _leaf_prob_product_static)
        self.jax = jax
        self.jnp = jnp
        self.design = design
        self.depth = int(depth)
        self.maxf = int(maxf)
        self.cutoff = int(cfg.get("cutoff") or 30)
        self.dec = make_decoder(design, self.depth, cfg)
        self.glen = self.dec.get_length(self.depth)
        depthc = self.depth
        maxfc = self.maxf
        cutc = self.cutoff
        dec = self.dec

        def core(g, L, BF):
            p = dec.decode(g, cutc)
            cov, mu, eff, _ = jax_equivalent_gaussian_static(p, depthc)
            psi, prob = jax_reduced_herald_static(cov, mu, eff, L, BF, depthc, maxfc)
            n_active = jnp.sum(jnp.asarray(p["leaf_active"]).astype(jnp.float64))
            return psi, prob, eff, n_active

        @functools.partial(jax.jit, static_argnums=(1, 2))
        def batch(G, L, BF):
            return jax.vmap(lambda g: core(g, L, BF))(G)

        @functools.partial(jax.jit, static_argnums=(1,))
        def leafprob(g, L):
            return _leaf_prob_product_static(dec.decode(g, cutc), L, depthc)

        def eq_one(g):
            p = dec.decode(g, cutc)
            return jax_equivalent_gaussian_static(p, depthc)

        self._batch = batch
        self._leafprob = leafprob
        self._eq_one = eq_one

    def score(self, G, L, BF):
        """G: (B, glen) float array. Returns numpy (psi[B,L], prob[B], eff[B,*], n_active[B])."""
        jnp = self.jnp
        psi, prob, eff, na = self._batch(jnp.asarray(G), int(L), int(BF))
        return (np.asarray(psi), np.asarray(prob, dtype=float),
                np.asarray(eff, dtype=float), np.asarray(na, dtype=float))

    def leaf_prob(self, g, L):
        return float(np.real(self._leafprob(self.jnp.asarray(g), int(L))))

    def equivalent_gaussian(self, g):
        cov, mu, eff, _ = self._eq_one(self.jnp.asarray(g))
        return np.asarray(cov, float), np.asarray(mu, float), np.asarray(eff, float)


# --------------------------------------------------------------------------- #
# Wigner negative volume via displaced parity (PROMPT s5, no qutip)            #
# --------------------------------------------------------------------------- #
def wigner_negative_volume(psi, grid=25, span=5.0, pad=None, return_grid=False):
    """Negative volume of the single-mode pure state ``psi`` (Fock amplitudes)
    via the displaced-parity Wigner  W(z) = (2/pi) <psi|D(z) Pi D(-z)|psi>,
    Pi = diag((-1)^n),  z = (x + i p)/sqrt(2).
    negative_volume = sum_{W<0} |W| dx dp on a coarse grid.

    Uses scipy.sparse.linalg.expm_multiply (no full matrix exponential / qutip).
    A non-negative result on a Gaussian pure state and a positive result on a
    Fock state are checked by the Hudson sentinel tests."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import expm_multiply

    psi = np.asarray(psi, dtype=complex).ravel()
    nrm = np.linalg.norm(psi)
    if nrm == 0 or not np.isfinite(nrm):
        out = (0.0, None, None, None) if return_grid else 0.0
        return out
    psi = psi / nrm
    # working dim: state support + headroom for the largest displacement (|z|^2 ~ span^2)
    prob = np.abs(psi) ** 2
    csum = np.cumsum(prob)
    supp = int(np.searchsorted(csum, csum[-1] * (1 - 1e-10))) + 1
    if pad is None:
        pad = int(span * span) + 30
    dim = min(len(psi), supp + pad)
    psi = psi[:dim].copy()
    psi = psi / np.linalg.norm(psi)

    sq = np.sqrt(np.arange(1, dim))
    a = diags(sq, 1, format="csc")          # annihilation: a[n-1,n]=sqrt(n)
    adag = diags(sq, -1, format="csc")      # creation
    parity = (-1.0) ** np.arange(dim)

    xs = np.linspace(-span, span, grid)
    dx = xs[1] - xs[0]
    W = np.empty((grid, grid))
    for ix, x in enumerate(xs):
        for ip, p in enumerate(xs):
            z = (x + 1j * p) / np.sqrt(2.0)
            # phi = D(z)^dagger psi = exp(conj(z) a - z adag) psi
            M = (np.conj(z) * a - z * adag).tocsc()
            phi = expm_multiply(M, psi)
            W[ix, ip] = (2.0 / np.pi) * np.real(np.sum(parity * np.abs(phi) ** 2))
    negvol = float(np.sum(np.abs(W[W < 0])) * dx * dx)
    if return_grid:
        return negvol, xs, xs, W
    return negvol


# --------------------------------------------------------------------------- #
# artifact classification (PROMPT s5) -- pure & testable                        #
# --------------------------------------------------------------------------- #
def classify_artifact(*, exp_lo, exp_hi, P, herald_norm, fired_modes, fp_budget,
                      indep_fidelity, wigner_negvol, gaussian_limit,
                      tol=0.02, neg_tol=1e-3, margin=1e-3, bf_high=8192, maxf=8,
                      has_fidelity=False):
    """Apply the physical-validity filters to one re-scored cell.  Returns
    (is_artifact, reasons).  A cell is VALID only if every filter passes.

    Hudson gate: a pure heralded state with ⟨O⟩ below the Gaussian limit MUST be
    Wigner-negative (Hudson's theorem); a claimed sub-Gaussian state with
    non-negative Wigner is impossible -> ``fake_subgaussian``."""
    reasons = []
    if not (np.isfinite(P) and P > 1e-40 and P <= 1.0 + 1e-9):
        reasons.append("bad_prob")
    if fired_modes > maxf or fp_budget > bf_high:
        reasons.append("over_budget")
    if not (herald_norm >= 0.99):
        reasons.append("unnormalized")
    if abs(exp_hi - exp_lo) > tol:
        reasons.append("l_truncation")
    if has_fidelity and np.isfinite(indep_fidelity) and indep_fidelity < 0.999:
        reasons.append("scorer_mismatch")
    claims_subgaussian = exp_hi < (gaussian_limit - margin)
    if claims_subgaussian and np.isfinite(wigner_negvol) and wigner_negvol < neg_tol:
        reasons.append("fake_subgaussian")
    return (len(reasons) > 0, reasons)


# --------------------------------------------------------------------------- #
# per-target reference values (PROMPT s2)                                       #
# --------------------------------------------------------------------------- #
_TARGET_CACHE = {}


def O_at(a, b, L):
    """L x L target operator (numpy); moment_operator is itself lru-cached."""
    from src.simulation.jax.moment_scorer import moment_operator
    return np.asarray(moment_operator(int(L), complex(a), complex(b)))


def target_refs(a, b, L_hi):
    """(gaussian_limit, gs_eig, O_hi) for target (a,b); cached per (a,b,L_hi)."""
    key = (complex(a), complex(b), int(L_hi))
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]
    import jax.numpy as jnp
    from src.simulation.jax.moment_scorer import moment_operator
    from src.utils.gkp_operator import get_u_vec_from_alpha_beta, gaussian_limit
    ux, uy, uz = get_u_vec_from_alpha_beta(complex(a), complex(b))
    glim = float(gaussian_limit(ux, uy, uz))
    O_hi = moment_operator(int(L_hi), complex(a), complex(b))
    gs_eig = float(jnp.linalg.eigvalsh(O_hi)[0])
    res = (glim, gs_eig, np.asarray(O_hi))
    _TARGET_CACHE[key] = res
    return res


def expectation(psi, O):
    """<psi|O|psi> (real) with O sliced to len(psi)."""
    L = len(psi)
    Os = O[:L, :L]
    return float(np.real(np.vdot(psi, Os @ psi)))


# --------------------------------------------------------------------------- #
# discovery                                                                     #
# --------------------------------------------------------------------------- #
def discover_runs(roots, group_glob):
    """Yield (root, group, run, pkl_path) for every results.pkl, sorted."""
    import fnmatch
    runs = []
    for root in roots:
        rabs = root if os.path.isabs(root) else os.path.join(REPO, root)
        if not os.path.isdir(rabs):
            continue
        for pkl in glob.glob(os.path.join(rabs, "*", "*", "results.pkl")):
            run = os.path.basename(os.path.dirname(pkl))
            group = os.path.basename(os.path.dirname(os.path.dirname(pkl)))
            if group_glob and not fnmatch.fnmatch(group, group_glob):
                continue
            runs.append((root, group, run, pkl))
    runs.sort(key=lambda r: (r[0], r[1], r[2]))
    return runs


# --------------------------------------------------------------------------- #
# main sweep                                                                    #
# --------------------------------------------------------------------------- #
def run_sweep(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    import pandas as pd

    roots = args.roots
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    t_start = time.time()
    runs = discover_runs(roots, args.groups)
    if args.limit and args.limit > 0:
        runs = runs[: args.limit]
    print(f"[discover] {len(runs)} runs under roots={roots} groups={args.groups!r}"
          + (f" (limited to {args.limit})" if args.limit else ""))

    # pre-load every config (for the empirical target maps + per-run resolution)
    configs = {}
    configs_by_group = defaultdict(list)
    for (root, group, run, pkl) in runs:
        cfgp = os.path.join(os.path.dirname(pkl), "config.json")
        try:
            cfg = json.load(open(cfgp))
        except Exception:  # noqa: BLE001
            cfg = {}
        configs[pkl] = cfg
        configs_by_group[group].append(cfg)
    beta_map, alpha_map, contradictions = build_empirical_maps(configs_by_group)
    print(f"[targets] empirical beta_map={ {k:str(v) for k,v in beta_map.items()} }")
    print(f"[targets] empirical alpha_map={ {k:str(v) for k,v in alpha_map.items()} }")
    if contradictions:
        print(f"[targets] WARNING contradictions: {contradictions}")

    engines = {}        # bucket-key -> RescoreEngine
    rows = []           # per re-scored cell
    undecodable = []    # per non-rescored run
    run_status = []     # ledger: one row per run
    n_runs_rescored = n_runs_undecodable = n_runs_unloadable = 0

    for ri, (root, group, run, pkl) in enumerate(runs):
        cfg = configs[pkl]
        prov = dict(root=root, group=group, run=run)
        # --- load ---
        try:
            gen, fit, des = load_run_arrays(pkl)
        except Exception as e:  # noqa: BLE001
            undecodable.append({**prov, "status": "unloadable",
                                "reason": f"{type(e).__name__}:{str(e)[:80]}", "n_cells": 0})
            run_status.append({**prov, "status": "unloadable"})
            n_runs_unloadable += 1
            continue

        # --- resolve target ---
        a, b, tsrc, treason = resolve_target(cfg, group, beta_map, alpha_map)
        if a is None:
            undecodable.append({**prov, "status": "undecodable", "reason": treason,
                                "n_cells": int(gen.shape[0])})
            run_status.append({**prov, "status": "undecodable"})
            n_runs_undecodable += 1
            continue

        # --- decoder / depth ---
        design = cfg.get("genotype") or parse_group_name(group)["design"]
        depth = cfg.get("depth")
        glen = gen.shape[1]
        if depth is None:
            depth = infer_depth(design, cfg, glen)
        else:
            depth = int(depth)
            try:
                dec_probe = make_decoder(design, depth, cfg)
                if dec_probe.get_length(depth) != glen:
                    depth = infer_depth(design, cfg, glen)
            except Exception:  # noqa: BLE001
                depth = infer_depth(design, cfg, glen)
        if depth is None:
            undecodable.append({**prov, "status": "undecodable",
                                "reason": f"length_mismatch:glen={glen}", "n_cells": int(gen.shape[0])})
            run_status.append({**prov, "status": "undecodable"})
            n_runs_undecodable += 1
            continue

        maxf = int(cfg.get("moment_maxf") or args.maxf)
        bkey = (design, depth, int(cfg.get("modes") or 3), int(cfg.get("pnr_max") or 3),
                round(float(cfg.get("r_scale", 2.0)), 6), round(float(cfg.get("d_scale", 3.0)), 6),
                round(float(cfg.get("hx_scale", 4.0)), 6), round(float(cfg.get("window", 0.1)), 6),
                maxf, int(cfg.get("cutoff") or 30))
        try:
            if bkey not in engines:
                engines[bkey] = RescoreEngine(design, depth, maxf, cfg)
            eng = engines[bkey]
        except Exception as e:  # noqa: BLE001
            undecodable.append({**prov, "status": "undecodable",
                                "reason": f"engine_build:{type(e).__name__}:{str(e)[:60]}",
                                "n_cells": int(gen.shape[0])})
            run_status.append({**prov, "status": "undecodable"})
            n_runs_undecodable += 1
            continue
        if glen != eng.glen:
            undecodable.append({**prov, "status": "undecodable",
                                "reason": f"length_mismatch:glen={glen}!=dec={eng.glen}",
                                "n_cells": int(gen.shape[0])})
            run_status.append({**prov, "status": "undecodable"})
            n_runs_undecodable += 1
            continue

        # --- select cells ---
        sel, n_valid, n_skipped = select_cells(fit, des, args.per_run_cap)
        if sel.size == 0:
            undecodable.append({**prov, "status": "undecodable", "reason": "no_valid_cells",
                                "n_cells": int(gen.shape[0])})
            run_status.append({**prov, "status": "undecodable"})
            n_runs_undecodable += 1
            continue

        glim, gs_eig, O_hi = target_refs(a, b, args.l_high)
        O_lo = O_at(a, b, args.l_search)
        scorer_backend = cfg.get("scorer") or "breeding_fock"
        moment_fast = bool(cfg.get("moment_fast", False))

        try:
            run_rows = _rescore_run(
                eng, gen, fit, des, sel, args, a, b, glim, gs_eig, O_hi, O_lo,
                prov, depth, maxf, scorer_backend, moment_fast)
        except Exception as e:  # noqa: BLE001
            if args.debug:
                traceback.print_exc()
            undecodable.append({**prov, "status": "undecodable",
                                "reason": f"rescore_error:{type(e).__name__}:{str(e)[:60]}",
                                "n_cells": int(gen.shape[0])})
            run_status.append({**prov, "status": "undecodable"})
            n_runs_undecodable += 1
            continue

        for r in run_rows:
            r["target_source"] = tsrc
            r["n_skipped_cells"] = n_skipped
        rows.extend(run_rows)
        run_status.append({**prov, "status": "rescored", "n_rescored": len(run_rows)})
        n_runs_rescored += 1
        if (ri + 1) % max(1, args.progress_every) == 0 or ri + 1 == len(runs):
            print(f"[{ri+1}/{len(runs)}] {root}/{group}/{run}  "
                  f"valid={n_valid} scored={len(run_rows)}  rows so far={len(rows)}  "
                  f"({time.time()-t_start:.0f}s)")

    print(f"[sweep] rescored={n_runs_rescored} undecodable={n_runs_undecodable} "
          f"unloadable={n_runs_unloadable}  cells={len(rows)}  ({time.time()-t_start:.0f}s)")

    # --- write deliverables ---
    df = pd.DataFrame(rows)
    summary = _write_outputs(df, undecodable, run_status, runs, beta_map, alpha_map,
                             contradictions, args, out, roots)
    return df, summary


def _rescore_run(eng, gen, fit, des, sel, args, a, b, glim, gs_eig, O_hi, O_lo,
                 prov, depth, maxf, scorer_backend, moment_fast):
    """Re-score the selected cells of one run; return a list of row dicts."""
    G = gen[sel].astype(np.float32)
    # batch in chunks to bound memory
    psis_lo, probs_lo = [], []
    psis_hi, probs_hi, effs, nactive = [], [], [], []
    cb = max(1, args.score_batch)
    for s in range(0, len(G), cb):
        Gb = G[s:s + cb]
        pl, ql, _, _ = eng.score(Gb, args.l_search, args.bf_search)
        ph, qh, ef, na = eng.score(Gb, args.l_high, args.bf_high)
        psis_lo.append(pl); probs_lo.append(ql)
        psis_hi.append(ph); probs_hi.append(qh); effs.append(ef); nactive.append(na)
    psi_lo = np.concatenate(psis_lo); prob_lo = np.concatenate(probs_lo)
    psi_hi = np.concatenate(psis_hi); prob_hi = np.concatenate(probs_hi)
    eff = np.concatenate(effs); nact = np.concatenate(nactive)

    # independent-fidelity subsample (deterministic: first k of the selected set)
    nsub = min(args.fidelity_subsample, len(sel))
    sub_idx = set(range(nsub))

    rows = []
    for j, k in enumerate(sel):
        pl = psi_lo[j]; ph = psi_hi[j]
        exp_lo = expectation(pl, O_lo)
        exp_hi = expectation(ph, O_hi)
        # leaf probability (exact; recovers the real value when --moment-fast stored prob=1)
        P = eng.leaf_prob(gen[k].astype(np.float32), args.l_high)
        logP = float(np.log10(np.clip(P, 1e-45, 1.0)))
        # Completeness.  psi is already normalized, so <psi|psi>==1 is uninformative;
        # instead measure how much probability the cutoff actually captures.
        #   herald_norm_lo = p_fock(L_lo)/p_fock(L_hi)  (prob = p_vac*p_fock; p_vac cancels)
        #                    -> was the *search* cutoff L_lo enough?
        #   herald_norm    = 1 - (mass in the top few Fock levels of the normalized psi_hi)
        #                    -> "retry at higher L": is L_hi itself enough? (the gate)
        herald_norm_lo = float(prob_lo[j] / prob_hi[j]) if prob_hi[j] > 0 else 0.0
        tail_hi = float(np.sum(np.abs(ph[-4:]) ** 2))
        herald_norm = 1.0 - tail_hi

        eff_r = np.rint(np.maximum(eff[j], 0.0)).astype(int)
        fired = eff_r[eff_r >= 1]
        fired_modes = int(fired.size)
        max_pnr = int(fired.max()) if fired.size else 0
        total_photons = int(fired.sum())
        fp_budget = int(np.prod(fired + 1)) if fired.size else 1
        active_modes = int(round(float(nact[j])))

        # independent thewalrus cross-check on the subsample
        indep_fidelity = np.nan
        if j in sub_idx:
            try:
                cov, mu, effc = eng.equivalent_gaussian(gen[k].astype(np.float32))
                from frontend.gbs_optimizer import reduced_herald
                ncontrol = effc.shape[0]
                cidx = list(range(1, 1 + ncontrol))
                nvec = [int(round(x)) for x in effc]
                psi_ind, _ = reduced_herald(cov, mu, 0, cidx, nvec, cutoff=args.l_high)
                psi_ind = np.asarray(psi_ind).ravel()
                L = min(len(psi_ind), len(ph))
                ov = np.vdot(psi_ind[:L], ph[:L])
                indep_fidelity = float(abs(ov) ** 2)
            except Exception:  # noqa: BLE001
                indep_fidelity = np.nan

        vs_gs = exp_hi - gs_eig
        vs_gaussian = exp_hi - glim

        # ---- Hudson / Wigner gate: only suspicious sub-Gaussian rows need it ----
        wig = np.nan
        claims_subgaussian = exp_hi < (glim - args.margin)
        if claims_subgaussian:
            wig = wigner_negative_volume(ph, grid=args.wigner_grid, span=args.wigner_span)

        # ---- artifact filters (PROMPT s5) ----
        is_artifact, reasons = classify_artifact(
            exp_lo=exp_lo, exp_hi=exp_hi, P=P, herald_norm=herald_norm,
            fired_modes=fired_modes, fp_budget=fp_budget,
            indep_fidelity=indep_fidelity, wigner_negvol=wig, gaussian_limit=glim,
            tol=args.tol, neg_tol=args.neg_tol, margin=args.margin,
            bf_high=args.bf_high, maxf=maxf, has_fidelity=(j in sub_idx))

        rows.append(dict(
            **prov, cell_idx=int(k), design=eng.design, depth=int(depth),
            modes=int(eng.dec.n_modes), pnr_max=int(eng.dec.pnr_max),
            target_alpha=str(a), target_beta=str(b),
            exp_stored=float(-fit[k, 0]),
            exp_lo=exp_lo, exp_hi=exp_hi, herald_norm=herald_norm,
            herald_norm_lo=herald_norm_lo,
            prob=float(P), logP=logP,
            herald_prob_full=float(prob_hi[j]),
            active_modes=active_modes, max_pnr=max_pnr, total_photons=total_photons,
            fired_modes=fired_modes, fp_budget=fp_budget,
            indep_fidelity=indep_fidelity, wigner_negvol=float(wig) if np.isfinite(wig) else np.nan,
            gs_eig=gs_eig, gaussian_limit=glim, vs_gs=vs_gs, vs_gaussian=vs_gaussian,
            scorer_backend=scorer_backend, moment_fast=moment_fast,
            is_artifact=bool(is_artifact),
            artifact_reason=(";".join(reasons) if reasons else ""),
        ))
    return rows


# --------------------------------------------------------------------------- #
# outputs (PROMPT s6)                                                           #
# --------------------------------------------------------------------------- #
def pareto_front(sub):
    """Artifact-free non-dominated front over (exp_hi minimize, logP maximize)."""
    import pandas as pd
    v = sub[~sub["is_artifact"]].copy()
    if v.empty:
        return v
    objs = np.column_stack([v["exp_hi"].values, -v["logP"].values])
    keep = np.ones(len(v), dtype=bool)
    for i in range(len(v)):
        dom = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        if np.any(dom):
            keep[i] = False
    return v[keep].sort_values("exp_hi")


def _write_outputs(df, undecodable, run_status, runs, beta_map, alpha_map,
                   contradictions, args, out, roots):
    import pandas as pd
    # sort deterministically
    if not df.empty:
        df = df.sort_values(["root", "group", "run", "cell_idx"]).reset_index(drop=True)

    # 1. all_solutions.parquet (+ csv sample)
    pq = os.path.join(out, "all_solutions.parquet")
    if not df.empty:
        try:
            df.to_parquet(pq, index=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] parquet failed ({e!r}); writing csv only")
        df.head(args.csv_sample).to_csv(os.path.join(out, "all_solutions_sample.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(out, "all_solutions_sample.csv"), index=False)

    # 2. per-group artifact-free Pareto fronts
    pf_dir = os.path.join(out, "pareto_fronts")
    os.makedirs(pf_dir, exist_ok=True)
    front_summ = {}
    if not df.empty:
        for group, sub in df.groupby("group"):
            fr = pareto_front(sub)
            cols = ["root", "group", "run", "cell_idx", "exp_hi", "logP", "prob",
                    "total_photons", "fired_modes", "max_pnr", "vs_gaussian", "vs_gs",
                    "wigner_negvol", "target_alpha", "target_beta"]
            cols = [c for c in cols if c in fr.columns]
            fr[cols].to_csv(os.path.join(pf_dir, f"{group}.csv"), index=False)
            front_summ[group] = fr

    # 5. undecodable.csv (every run not re-scored)
    pd.DataFrame(undecodable).to_csv(os.path.join(out, "undecodable.csv"), index=False)
    pd.DataFrame(run_status).to_csv(os.path.join(out, "run_ledger.csv"), index=False)

    # 3 + 6. best states + plots
    best_per_group = _save_best_states_and_plots(df, out, args)

    # 4. REPORT.md
    summary = _write_report(df, undecodable, run_status, runs, beta_map, alpha_map,
                            contradictions, args, out, roots, front_summ, best_per_group)
    print(f"[outputs] wrote {out}/REPORT.md, all_solutions.parquet, "
          f"pareto_fronts/, best_states/, undecodable.csv, plots/")
    return summary


def _save_best_states_and_plots(df, out, args):
    import pandas as pd
    best_dir = os.path.join(out, "best_states")
    plot_dir = os.path.join(out, "plots")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    best_per_group = {}
    if df.empty:
        return best_per_group

    # artifact_reason histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        reasons = Counter()
        for r in df[df["is_artifact"]]["artifact_reason"]:
            for tok in str(r).split(";"):
                if tok:
                    reasons[tok] += 1
        if reasons:
            fig, ax = plt.subplots(figsize=(7, 4))
            ks = list(reasons); vs = [reasons[k] for k in ks]
            ax.bar(ks, vs, color="#b5651d")
            ax.set_ylabel("# cells"); ax.set_title("artifact reasons")
            plt.xticks(rotation=30, ha="right"); fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "artifact_reasons.png"), dpi=130)
            plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] reason histogram failed: {e!r}")

    valid = df[~df["is_artifact"]]
    for group, sub in df.groupby("group"):
        vsub = sub[~sub["is_artifact"]]
        if vsub.empty:
            continue
        best = vsub.loc[vsub["exp_hi"].idxmin()]
        best_per_group[group] = best
        gdir = os.path.join(best_dir, group)
        os.makedirs(gdir, exist_ok=True)
        # re-derive + save the genuine best state's raw artifacts
        try:
            _dump_best_state(best, gdir, args)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] dump best {group} failed: {e!r}")
        # per-group Pareto plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(sub["logP"], sub["exp_hi"], s=8, c="#cccccc", label="all rescored")
            ax.scatter(vsub["logP"], vsub["exp_hi"], s=14, c="#1f77b4", label="valid")
            fr = pareto_front(sub)
            if not fr.empty:
                ax.plot(fr["logP"], fr["exp_hi"], "-o", c="#d62728", ms=4, label="Pareto")
            ax.axhline(best["gaussian_limit"], ls="--", c="green", lw=1, label="Gaussian limit")
            ax.set_xlabel("logP"); ax.set_ylabel(r"$\langle O\rangle$")
            ax.set_title(group); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(plot_dir, f"pareto_{group}.png"), dpi=130)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] pareto plot {group} failed: {e!r}")
    return best_per_group


def _dump_best_state(best, gdir, args):
    """Save raw genotype, decoded params, psi_hi, scalars, and a Wigner PNG for
    one genuine best state."""
    import json as _json
    root = best["root"]
    base = root if os.path.isabs(root) else os.path.join(REPO, root)
    pkl = os.path.join(base, best["group"], best["run"], "results.pkl")
    gen, fit, des = load_run_arrays(pkl)
    g = gen[int(best["cell_idx"])].astype(np.float32)
    cfg = json.load(open(os.path.join(os.path.dirname(pkl), "config.json")))
    design = cfg.get("genotype") or parse_group_name(best["group"])["design"]
    depth = int(best["depth"])
    maxf = int(cfg.get("moment_maxf") or args.maxf)
    eng = RescoreEngine(design, depth, maxf, cfg)
    a = complex(best["target_alpha"]); b = complex(best["target_beta"])
    psi, prob, eff, _na = eng.score(g[None], args.l_high, args.bf_high)
    psi = psi[0]
    np.save(os.path.join(gdir, "genotype.npy"), g)
    np.save(os.path.join(gdir, "psi_hi.npy"), psi)
    params = eng.dec.decode(eng.jnp.asarray(g), eng.cutoff)
    params_np = {k: np.asarray(v).tolist() if hasattr(v, "shape") or isinstance(v, (list, tuple))
                 else (np.asarray(v).tolist() if hasattr(v, "__array__") else v)
                 for k, v in _flatten_params(params).items()}
    with open(os.path.join(gdir, "params.json"), "w") as f:
        _json.dump(params_np, f, indent=1, default=str)
    meta = {k: (float(best[k]) if isinstance(best[k], (int, float, np.floating, np.integer))
                else str(best[k]))
            for k in ["exp_hi", "exp_stored", "prob", "logP", "vs_gaussian", "vs_gs",
                      "total_photons", "fired_modes", "max_pnr", "wigner_negvol",
                      "indep_fidelity", "root", "run", "cell_idx", "target_alpha",
                      "target_beta"] if k in best}
    with open(os.path.join(gdir, "meta.json"), "w") as f:
        _json.dump(meta, f, indent=1)
    # Wigner PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        negvol, xs, ps, W = wigner_negative_volume(
            psi, grid=max(41, args.wigner_grid), span=args.wigner_span, return_grid=True)
        if W is not None:
            wlim = np.max(np.abs(W))
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.contourf(xs, ps, W.T, 60, cmap="RdBu_r", vmin=-wlim, vmax=wlim)
            ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("p")
            ax.set_title(f"{best['group']}  <O>={best['exp_hi']:.3f}  negvol={negvol:.2e}")
            fig.colorbar(im, ax=ax, shrink=0.85); fig.tight_layout()
            fig.savefig(os.path.join(gdir, "wigner.png"), dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] wigner png {best['group']} failed: {e!r}")


def _flatten_params(params, prefix=""):
    flat = {}
    for k, v in params.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_params(v, kk + "."))
        else:
            flat[kk] = v
    return flat


def _write_report(df, undecodable, run_status, runs, beta_map, alpha_map,
                  contradictions, args, out, roots, front_summ, best_per_group):
    import pandas as pd
    rl = pd.DataFrame(run_status)
    n_runs = len(runs)
    n_resc = int((rl["status"] == "rescored").sum()) if not rl.empty else 0
    n_undec = int((rl["status"] == "undecodable").sum()) if not rl.empty else 0
    n_unload = int((rl["status"] == "unloadable").sum()) if not rl.empty else 0
    n_cells = len(df)
    n_valid = int((~df["is_artifact"]).sum()) if not df.empty else 0
    n_art = n_cells - n_valid

    undec_reasons = Counter()
    for r in undecodable:
        undec_reasons[str(r.get("reason", "?")).split(":")[0]] += 1
    art_reasons = Counter()
    if not df.empty:
        for r in df[df["is_artifact"]]["artifact_reason"]:
            for tok in str(r).split(";"):
                if tok:
                    art_reasons[tok] += 1

    groups = sorted(df["group"].unique()) if not df.empty else []
    lines = []
    lines.append("# Re-score archaeology report\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                 f"L_search={args.l_search} L_high={args.l_high} bf_high={args.bf_high} "
                 f"maxf={args.maxf} tol={args.tol} neg_tol={args.neg_tol} "
                 f"per_run_cap={args.per_run_cap}"
                 + (f" limit={args.limit}" if args.limit else "") + "_\n")
    lines.append("## Totals\n")
    lines.append(f"- roots scanned: {len(roots)}  ({', '.join(roots)})")
    lines.append(f"- groups discovered: {len(set(g for _,g,_,_ in runs))}; "
                 f"runs discovered: {n_runs}")
    lines.append(f"- runs **re-scored**: {n_resc} | **undecodable**: {n_undec} | "
                 f"**unloadable**: {n_unload}  "
                 f"(accounting: {n_resc}+{n_undec}+{n_unload}={n_resc+n_undec+n_unload} == {n_runs})")
    lines.append(f"- cells re-scored: {n_cells} | **valid**: {n_valid} | "
                 f"**artifact**: {n_art}"
                 + (f" ({100*n_art/max(n_cells,1):.1f}%)" if n_cells else ""))
    if undec_reasons:
        lines.append(f"- undecodable/unloadable reasons: "
                     + ", ".join(f"{k}={v}" for k, v in undec_reasons.most_common()))
    if art_reasons:
        lines.append(f"- artifact reasons: "
                     + ", ".join(f"{k}={v}" for k, v in art_reasons.most_common()))
    lines.append("")
    lines.append("## Empirical target maps\n")
    lines.append("| folder b-code | resolved beta |   | folder a-code | resolved alpha |")
    lines.append("|---|---|---|---|---|")
    bk = sorted(beta_map); ak = sorted(alpha_map)
    for i in range(max(len(bk), len(ak))):
        bc = bk[i] if i < len(bk) else ""
        bv = str(beta_map[bc]) if bc else ""
        ac = ak[i] if i < len(ak) else ""
        av = str(alpha_map[ac]) if ac else ""
        lines.append(f"| {bc} | {bv} |   | {ac} | {av} |")
    if contradictions:
        lines.append(f"\n**WARNING** target-map contradictions: `{contradictions}`")
    lines.append("")

    lines.append("## Per target group: genuine best vs. originally-stored best\n")
    lines.append("| group | target (a,b) | best genuine ⟨O⟩ | vs_gaussian | vs_gs | prob | "
                 "photons | provenance | stored best ⟨O⟩ | artifact gap |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    group_target = {}
    if not df.empty:
        for group in groups:
            sub = df[df["group"] == group]
            tgt = f"({sub['target_alpha'].iloc[0]}, {sub['target_beta'].iloc[0]})"
            group_target[group] = tgt
            stored_best = float(sub["exp_stored"].min())
            vsub = sub[~sub["is_artifact"]]
            if vsub.empty:
                lines.append(f"| {group} | {tgt} | — (all artifact) | | | | | | "
                             f"{stored_best:.4f} | — |")
                continue
            best = vsub.loc[vsub["exp_hi"].idxmin()]
            gap = best["exp_hi"] - stored_best
            prov = f"{best['root']}/{best['run']}#{int(best['cell_idx'])}"
            lines.append(
                f"| {group} | {tgt} | **{best['exp_hi']:.4f}** | {best['vs_gaussian']:+.4f} | "
                f"{best['vs_gs']:+.4f} | {best['prob']:.2e} | {int(best['total_photons'])} | "
                f"{prov} | {stored_best:.4f} | {gap:+.4f} |")
    lines.append("")
    lines.append("_`vs_gaussian` < 0 means a genuine non-Gaussian advantage (and must show "
                 "Wigner negativity to survive the Hudson gate). `artifact gap` = genuine best "
                 "− stored best; large positive means much of the old 'record' was a "
                 "truncation/placeholder artifact._\n")

    # narrative
    lines.append("## Narrative\n")
    sub_g = df[(~df["is_artifact"]) & (df["vs_gaussian"] < 0)] if not df.empty else df
    n_subg = len(sub_g) if not df.empty else 0
    lines.append(f"- {n_resc}/{n_runs} runs ({100*n_resc/max(n_runs,1):.0f}%) decoded & "
                 f"re-scored with the exact pipeline; the rest are logged in `undecodable.csv`.")
    lines.append(f"- {n_valid}/{n_cells} re-scored cells survived all physical-validity filters.")
    lines.append(f"- genuine sub-Gaussian (⟨O⟩ < Gaussian limit, Wigner-negative) cells: {n_subg}.")
    fk = [r for r in (df["artifact_reason"] if not df.empty else []) if "fake_subgaussian" in str(r)]
    lines.append(f"- `fake_subgaussian` rejections (claimed sub-Gaussian but Wigner-positive → "
                 f"impossible by Hudson's theorem): {art_reasons.get('fake_subgaussian',0)}.")
    unresolved = [r for r in undecodable if str(r.get("reason","")).startswith("target_unresolved")]
    if unresolved:
        gs = sorted(set(r["group"] for r in unresolved))
        lines.append(f"- target_unresolved groups (excluded, not guessed): {gs}")
    lines.append("")
    lines.append("## Files\n")
    lines.append("- `all_solutions.parquet` — one row per re-scored cell (full schema).")
    lines.append("- `pareto_fronts/<group>.csv` — artifact-free ⟨O⟩-vs-logP front per group.")
    lines.append("- `best_states/<group>/` — genotype.npy, params.json, psi_hi.npy, wigner.png, meta.json.")
    lines.append("- `undecodable.csv` — every run not re-scored, with reason.")
    lines.append("- `run_ledger.csv` — status of every discovered run.")
    lines.append("- `plots/` — per-group Pareto fronts, artifact-reason histogram.")

    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return dict(n_runs=n_runs, n_rescored=n_resc, n_undecodable=n_undec,
                n_unloadable=n_unload, n_cells=n_cells, n_valid=n_valid,
                n_artifact=n_art, art_reasons=dict(art_reasons),
                undec_reasons=dict(undec_reasons))


# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS,
                    help="run roots (each a <root>/<group>/<run>/results.pkl tree)")
    ap.add_argument("--groups", default="*", help="group-name glob filter (fnmatch)")
    ap.add_argument("--l-search", type=int, default=50, dest="l_search")
    ap.add_argument("--bf-search", type=int, default=1024, dest="bf_search")
    ap.add_argument("--l-high", type=int, default=120, dest="l_high")
    ap.add_argument("--bf-high", type=int, default=8192, dest="bf_high")
    ap.add_argument("--maxf", type=int, default=8, help="default in-graph fired-mode cap (config overrides)")
    ap.add_argument("--tol", type=float, default=0.02, help="max |Δ⟨O⟩| (L_lo vs L_hi) before l_truncation")
    ap.add_argument("--neg-tol", type=float, default=1e-3, dest="neg_tol",
                    help="min Wigner negative volume for a claimed sub-Gaussian state")
    ap.add_argument("--margin", type=float, default=1e-3,
                    help="⟨O⟩ must beat (gaussian_limit - margin) to count as claiming sub-Gaussian")
    ap.add_argument("--per-run-cap", type=int, default=256, dest="per_run_cap",
                    help="max cells re-scored per run (0 = no cap = all selected)")
    ap.add_argument("--fidelity-subsample", type=int, default=4, dest="fidelity_subsample",
                    help="cells per run cross-checked against thewalrus reduced_herald")
    ap.add_argument("--score-batch", type=int, default=32, dest="score_batch",
                    help="genotypes per vmapped batch")
    ap.add_argument("--wigner-grid", type=int, default=25, dest="wigner_grid")
    ap.add_argument("--wigner-span", type=float, default=5.0, dest="wigner_span")
    ap.add_argument("--csv-sample", type=int, default=4000, dest="csv_sample")
    ap.add_argument("--limit", type=int, default=0, help="cap #runs processed (smoke runs)")
    ap.add_argument("--progress-every", type=int, default=10, dest="progress_every")
    ap.add_argument("--out", default="recompute/")
    ap.add_argument("--debug", action="store_true")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    run_sweep(args)


if __name__ == "__main__":
    main()
