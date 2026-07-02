"""PNR-pattern seed generation for the B30 family (B30 / B30B / B30F).

Physics-motivated initial population: instead of asking the GA to discover
firing structure from random continuous noise (measure-zero odds), we inject
genotypes that already HOLD canonical heralding patterns -- combinations of
1..pnr_max clicks across the leaves' detectors -- dressed with breeding-friendly
continuous defaults (balanced beamsplitters, x=0 homodyne, strong squeezing,
no displacement) plus jitter.

For small (depth, pnr_max) the canonical family is enumerated; the remainder
(or everything beyond the budget) is sampled with click counts biased toward
the literature range (1-3 photons per detector).
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.genotypes.genotypes import get_genotype_decoder

# raw value whose tanh()*pi/2 mapping gives theta = pi/4 (balanced BS)
_BALANCED_THETA_RAW = float(np.arctanh(0.5))


def b30_layout(genotype_name: str, depth: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Flat-array layout of the B30-family genotype:
    hom(nodes) | shared(SH) | unique(L*U) | mix(nodes*3) | final(5).
    Returns index bounds + decoder constants used by seeds and macro-mutations."""
    dec = get_genotype_decoder(genotype_name, depth=depth, config=config)
    L = 2 ** depth
    nodes = L - 1
    m = dec.n_modes
    SH, U, PN, F = dec.Sharedv, dec.Unique, dec.PN, dec.F
    hom0 = 0
    sh0 = nodes
    u0 = sh0 + SH
    mix0 = u0 + L * U
    fin0 = mix0 + nodes * PN
    total = fin0 + F
    assert total == dec.get_length(depth), (
        f"layout mismatch: {total} != {dec.get_length(depth)}")
    return {
        "L": L, "nodes": nodes, "n_modes": m,
        "n_control": dec.n_control, "pnr_len": dec.pnr_len,
        "pnr_max": dec.pnr_max, "U": U, "PN": PN, "F": F, "D": total,
        "hom": (hom0, nodes),
        "shared": (sh0, u0),
        "shared_r": (sh0, sh0 + m),
        "shared_phases": (sh0 + m, sh0 + m + m * m),
        "shared_disp": (sh0 + m + m * m, u0),
        "unique": (u0, mix0),
        "mix": (mix0, fin0),
        "final": (fin0, total),
    }


def _encode_pnr0(k: int, pnr_max: int, forced: bool) -> float:
    """Raw value for the FIRST detector's click count ``k`` (bin centre)."""
    if forced:  # B30F: k = 1 + round(clip(v,0,1)*(pm-1))
        return 0.5 if pnr_max <= 1 else (k - 1) / float(pnr_max - 1)
    return k / float(pnr_max)  # B30: k = round(clip(v,0,1)*pm)


def _encode_pnr_rest(k: int, pnr_max: int) -> float:
    return k / float(pnr_max)


def _encode_n_ctrl(n: int, n_control: int, forced: bool) -> float:
    if forced:  # B30F: n = 1 + round(clip((v+1)/2)*(nc-1))
        if n_control <= 1:
            return 0.0
        return 2.0 * (n - 1) / float(n_control - 1) - 1.0
    return 2.0 * n / float(n_control) - 1.0  # B30: n = round((v+1)/2*nc)


def _base_continuous(lay: Dict[str, Any], rng: np.random.Generator,
                     r_frac: float = 0.7) -> np.ndarray:
    """Breeding-friendly continuous defaults with jitter:
    hom x=0, balanced BS everywhere, strong squeezing on the signal mode,
    random small leaf-interferometer phases (nonzero signal-control coupling),
    zero displacement, identity final Gaussian."""
    g = np.zeros(lay["D"], dtype=np.float32)
    # hom: raw 0 -> x = 0 (jittered)
    h0, h1 = lay["hom"]
    g[h0:h1] = rng.normal(0.0, 0.05, h1 - h0)
    # shared squeezing: signal mode strong, controls moderate
    r0, r1 = lay["shared_r"]
    r_raw = np.full(r1 - r0, float(np.arctanh(min(r_frac * 0.7, 0.95))))
    r_raw[0] = float(np.arctanh(min(r_frac, 0.95)))
    g[r0:r1] = r_raw + rng.normal(0.0, 0.05, r1 - r0)
    # leaf interferometer phases: small random -> genuine signal-control coupling
    p0, p1 = lay["shared_phases"]
    g[p0:p1] = rng.uniform(-0.6, 0.6, p1 - p0)
    # displacement: zero (breeding uses none)
    d0, d1 = lay["shared_disp"]
    g[d0:d1] = rng.normal(0.0, 0.02, d1 - d0)
    # mixing: balanced theta = pi/4, phi = varphi = 0
    m0, m1 = lay["mix"]
    mix = np.zeros((lay["nodes"], lay["PN"]), dtype=np.float32)
    mix[:, 0] = _BALANCED_THETA_RAW
    g[m0:m1] = (mix + rng.normal(0.0, 0.05, mix.shape)).reshape(-1)
    # final Gaussian: identity (raw 0)
    f0, f1 = lay["final"]
    g[f0:f1] = rng.normal(0.0, 0.02, f1 - f0)
    return g


def _write_pattern(g: np.ndarray, lay: Dict[str, Any], forced: bool,
                   active: np.ndarray, n_ctrl: np.ndarray,
                   pnr: np.ndarray) -> np.ndarray:
    """Encode a discrete pattern (active[L], n_ctrl[L], pnr[L, pnr_len]) into
    the unique block of raw genotype ``g`` (in place; returns g)."""
    u0, _ = lay["unique"]
    U = lay["U"]
    pm = lay["pnr_max"]
    for i in range(lay["L"]):
        base = u0 + i * U
        g[base + 0] = 1.0 if active[i] else -1.0
        g[base + 1] = _encode_n_ctrl(int(n_ctrl[i]), lay["n_control"], forced)
        g[base + 2] = _encode_pnr0(int(pnr[i, 0]), pm, forced)
        for c in range(1, lay["pnr_len"]):
            g[base + 2 + c] = _encode_pnr_rest(int(pnr[i, c]), pm)
    return g


def generate_pnr_pattern_seeds(
    genotype_name: str,
    depth: int,
    config: Dict[str, Any],
    n_seeds: int,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate up to ``n_seeds`` raw genotypes carrying canonical PNR firing
    patterns.  Canonical family first (enumerated), then biased random samples.

    Canonical family (all leaves balanced-mixed, x=0 homodyne):
      * uniform-k:    all leaves active, every first detector fires k (k=1..pm)
      * m-active:     first m leaves active (m = 2, 4, ..., L), single click
      * alternating:  clicks alternate k=1/2 across leaves
      * two-detector: n_ctrl=2, both detectors fire once (when modes >= 3)
    """
    if genotype_name not in {"B30", "B30B", "B30F"}:
        raise ValueError(f"PNR pattern seeds support the B30 family, "
                         f"not {genotype_name!r}")
    forced = genotype_name == "B30F"
    lay = b30_layout(genotype_name, depth, config)
    rng = rng or np.random.default_rng(0)
    L, pm, pl = lay["L"], lay["pnr_max"], lay["pnr_len"]
    seeds: List[np.ndarray] = []

    # MOMENT-SCORER BUDGET: a pattern is only scoreable in-graph if the number
    # of FIRED detectors kf <= moment_maxf and prod(n_j + 1) <= moment_bf.
    # Patterns beyond the budget would be flagged invalid at insertion, so we
    # never emit them (at depth 5 that excludes 'all 32 leaves fire').
    maxf = int((config or {}).get("moment_maxf", 8))
    bf = int((config or {}).get("moment_bf", 1024))
    log_bf = np.log(float(bf))

    def _in_budget(active, n_ctrl, pnr) -> bool:
        ns = []
        for i in range(L):
            if not active[i]:
                continue
            for c in range(min(int(n_ctrl[i]), pl)):
                if pnr[i, c] >= 1:
                    ns.append(int(pnr[i, c]))
        if len(ns) > maxf:
            return False
        return float(np.sum(np.log(np.asarray(ns, float) + 1.0))) <= log_bf + 1e-9

    def add(active, n_ctrl, pnr):
        if len(seeds) >= n_seeds or not _in_budget(active, n_ctrl, pnr):
            return
        g = _base_continuous(lay, rng)
        seeds.append(_write_pattern(g, lay, forced, active, n_ctrl, pnr))

    ones = np.ones(L, dtype=int)

    def _first_m_active(m):
        act = np.zeros(L, bool)
        act[:max(1, min(m, L))] = True
        return act

    # 1) uniform-k on as many leaves as the budget allows
    for k in range(1, pm + 1):
        m_k = min(L, maxf, int(log_bf // np.log(k + 1.0)))
        if m_k < 1:
            continue
        pnr = np.zeros((L, pl), dtype=int)
        pnr[:, 0] = k
        add(_first_m_active(m_k), ones, pnr)

    # 2) m-active single-click (m = 2, 4, ..., budget-capped)
    m = 2
    m_max = min(L, maxf, int(log_bf // np.log(2.0)))
    while m <= m_max:
        pnr = np.zeros((L, pl), dtype=int)
        pnr[:, 0] = 1
        add(_first_m_active(m), ones, pnr)
        m *= 2

    # 3) alternating 1/2 clicks on the budget-capped prefix
    m_alt = min(L, maxf)
    pnr = np.zeros((L, pl), dtype=int)
    pnr[:, 0] = np.where(np.arange(L) % 2 == 0, 1, 2)
    while m_alt >= 2 and not _in_budget(_first_m_active(m_alt), ones, pnr):
        m_alt -= 1
    add(_first_m_active(max(m_alt, 1)), ones, pnr)

    # 4) two-detector (1,1) when a second control exists (kf = 2 per leaf)
    if lay["n_control"] >= 2 and pl >= 2:
        m_2d = min(L, maxf // 2, int(log_bf // (2.0 * np.log(2.0))))
        if m_2d >= 1:
            pnr = np.zeros((L, pl), dtype=int)
            pnr[:, 0] = 1
            pnr[:, 1] = 1
            add(_first_m_active(m_2d), ones * 2, pnr)

    # 5) biased random samples for the remainder (click counts favour 1-3),
    #    constructed under the budget: sample the fired-leaf count first, then
    #    clicks while the log-budget lasts.
    weights = np.array([0.45, 0.25, 0.15] + [0.15 / max(pm - 3, 1)] * max(pm - 3, 0))
    weights = weights[:pm] / weights[:pm].sum()
    attempts = 0
    while len(seeds) < n_seeds and attempts < 50 * n_seeds:
        attempts += 1
        m_fire = int(rng.integers(1, min(L, maxf) + 1))
        which = rng.choice(L, size=m_fire, replace=False)
        act = np.zeros(L, bool)
        act[which] = True
        if forced:
            act[0] = True
        n_ctrl = np.ones(L, dtype=int)
        pnr = np.zeros((L, pl), dtype=int)
        budget = log_bf
        fired = 0
        for i in np.flatnonzero(act):
            k = int(rng.choice(np.arange(1, pm + 1), p=weights))
            while k >= 1 and (np.log(k + 1.0) > budget or fired >= maxf):
                k -= 1
            if k < 1:
                if forced:
                    k = 1  # forced decode fires anyway; keep it minimal
                else:
                    act[i] = False
                    continue
            pnr[i, 0] = k
            budget -= np.log(k + 1.0)
            fired += 1
        if not act.any():
            act[0] = True
            pnr[0, 0] = 1
        add(act, n_ctrl, pnr)

    return seeds
