"""Moment-space optimizer scorer (JAX) -- exact, truncation-free.

This is the in-loop counterpart of the numpy rescoring path
(``frontend.gaussian_decomposition.compute_equivalent_gaussian`` ->
``frontend.gbs_optimizer.reduced_herald``): it composes the WHOLE breeding
circuit on a small covariance (symplectics + analytic homodyne/n=0
conditioning) and only ever materialises the final single-mode Fock state, so
it is exact at any squeezing -- no Fock breeding tree, no cutoff-30 truncation
artifacts to reap.  See ``MOMENT_SCORER_PLAN.md``.

Piece 2a (this file, so far): ``jax_equivalent_gaussian`` -- a JAX, differentiable
mirror of ``compute_equivalent_gaussian``.  The CONTINUOUS parameters (squeezing
r, interferometer phases, displacements, BS theta/phi, homodyne x, final
Gaussian) flow as traced arrays => exact gradients.  The DISCRETE structure
(leaf_active, n_ctrl per leaf, PNR pattern) is read concretely, so each distinct
structure traces/JITs to its own static graph -- the population is scored
bucketed by structure (``MOMENT_SCORER_PLAN.md`` Sec. 2b/3), which keeps every
covariance exactly sized and imposes no cap on the number of control modes.

Convention (matches thewalrus + the numpy reference): hbar = 2, xp-ordering
(x_0..x_{N-1}, p_0..p_{N-1}), vacuum covariance = Identity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Tuple

from src.simulation.jax.runner import jax_clements_unitary
from src.genotypes.genotypes import get_genotype_decoder
from src.simulation.jax.herald import (
    passive_unitary_to_symplectic,
    vacuum_covariance,
    complex_alpha_to_qp,
)

HBAR = 2.0


# --------------------------------------------------------------------------- #
# Leaf Gaussian moments (mirror frontend.independent_verifier._build_gaussian_moments
# and runner.jax_get_gaussian_moments -- identical convention, verified).
# --------------------------------------------------------------------------- #
def _leaf_moments(r: jnp.ndarray, phases: jnp.ndarray, disp: jnp.ndarray,
                  N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """(mu, cov) of an N-mode pure Gaussian leaf: squeezers -> Clements -> disp."""
    S_sq = jnp.diag(jnp.concatenate([jnp.exp(-r[:N]), jnp.exp(r[:N])]))
    U_pass = jax_clements_unitary(phases[: N * N], N)
    S = passive_unitary_to_symplectic(U_pass) @ S_sq
    mu = complex_alpha_to_qp(disp[:N], HBAR)
    cov = S @ vacuum_covariance(N, HBAR) @ S.T
    return mu, cov


# --------------------------------------------------------------------------- #
# Beam splitter (mirror gaussian_decomposition.get_bs_symplectic) and
# point-homodyne projection (mirror gaussian_decomposition.measure_homodyne).
# --------------------------------------------------------------------------- #
def _bs_symplectic(theta: jnp.ndarray, phi: jnp.ndarray, N: int,
                   mode_a: int, mode_b: int) -> jnp.ndarray:
    """2N x 2N BS symplectic, same Heisenberg convention as the JAX breeding sim
    (a_A -> t a_A + e^{i phi} r a_B; a_B -> -e^{-i phi} r a_A + t a_B)."""
    t = jnp.cos(theta)
    r = jnp.sin(theta)
    U = jnp.eye(N, dtype=jnp.complex128)
    U = U.at[mode_a, mode_a].set(t.astype(jnp.complex128))
    U = U.at[mode_a, mode_b].set(jnp.exp(1j * phi) * r)
    U = U.at[mode_b, mode_a].set(-jnp.exp(-1j * phi) * r)
    U = U.at[mode_b, mode_b].set(t.astype(jnp.complex128))
    return passive_unitary_to_symplectic(U)


def _measure_homodyne(V: jnp.ndarray, mu: jnp.ndarray, idx: int, N: int,
                      x_val: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project pure Gaussian onto |x=x_val> for mode ``idx``; drop that mode.
    Returns (V_new[2(N-1)], mu_new, gaussian_density p(x=x_val))."""
    B11 = V[idx, idx]
    keep = [i for i in range(N) if i != idx] + [i + N for i in range(N) if i != idx]
    keep = jnp.array(keep, dtype=jnp.int32)
    c = V[keep, idx]
    V_A = V[jnp.ix_(keep, keep)]
    V_new = V_A - jnp.outer(c, c) / B11
    mu_new = mu[keep] + (x_val - mu[idx]) / B11 * c
    density = (jnp.exp(-0.5 * (x_val - mu[idx]) ** 2 / B11)
              / jnp.sqrt(2.0 * jnp.pi * B11))
    return V_new, mu_new, density


def _apply_final_gaussian(V: jnp.ndarray, mu: jnp.ndarray, fg: Dict[str, Any],
                          idx: int, N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply S(r,phi) R(varphi) D(disp) on the root signal mode (mirror
    gaussian_decomposition.apply_final_gaussian_symplectic)."""
    r = fg["r"]; phi = fg["phi"]; varphi = fg["varphi"]; disp = fg["disp"]
    cos_v, sin_v = jnp.cos(varphi), jnp.sin(varphi)
    R_mat = jnp.array([[cos_v, -sin_v], [sin_v, cos_v]])
    ch, sh = jnp.cosh(r), jnp.sinh(r)
    c2, s2 = jnp.cos(2 * phi), jnp.sin(2 * phi)
    S_sq = jnp.array([[ch - sh * c2, -sh * s2], [-sh * s2, ch + sh * c2]])
    S2 = R_mat @ S_sq
    S = jnp.eye(2 * N)
    S = S.at[idx, idx].set(S2[0, 0])
    S = S.at[idx, idx + N].set(S2[0, 1])
    S = S.at[idx + N, idx].set(S2[1, 0])
    S = S.at[idx + N, idx + N].set(S2[1, 1])
    V_new = S @ V @ S.T
    mu_new = S @ mu
    scale = jnp.sqrt(2 * HBAR)
    disp = disp.astype(jnp.complex128) if jnp.iscomplexobj(disp) else disp + 0j
    mu_new = mu_new.at[idx].add(scale * jnp.real(disp))
    mu_new = mu_new.at[idx + N].add(scale * jnp.imag(disp))
    return V_new, mu_new


def _global_assemble(blocks: List[Tuple[int, jnp.ndarray]],
                     mblocks: List[Tuple[int, jnp.ndarray]], tot: int
                     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Block-place per-leaf (x..p)-ordered moments into the global (x_all,p_all)
    layout (mirror compute_equivalent_gaussian's assembly loops)."""
    V = jnp.zeros((2 * tot, 2 * tot))
    off = 0
    for N_leaf, V_leaf in blocks:
        xs = jnp.arange(off, off + N_leaf)
        xp_rows = jnp.concatenate([xs, xs + tot])
        local = jnp.concatenate([jnp.arange(N_leaf), jnp.arange(N_leaf) + N_leaf])
        V = V.at[jnp.ix_(xp_rows, xp_rows)].set(V_leaf[jnp.ix_(local, local)])
        off += N_leaf
    mu = jnp.zeros(2 * tot)
    off = 0
    for N_leaf, mu_leaf in mblocks:
        mu = mu.at[jnp.arange(off, off + N_leaf)].set(mu_leaf[:N_leaf])
        mu = mu.at[jnp.arange(off, off + N_leaf) + tot].set(mu_leaf[N_leaf:])
        off += N_leaf
    return mu, V


Structure = Tuple[Tuple[bool, ...], Tuple[int, ...], Tuple[Tuple[int, ...], ...]]


def extract_structure(params: Dict[str, Any]) -> Structure:
    """Read the discrete structure (leaf_active, n_ctrl, per-leaf pnr) concretely
    from decoded params.  Hashable => usable as a static jit argument and as a
    bucket key.  Call EAGERLY (outside jit); pass the result back in as
    ``structure=`` so the heavy path can be jitted/vmapped over continuous params.
    """
    lp = params["leaf_params"]
    active = tuple(bool(np.asarray(params["leaf_active"])[i]) for i in range(8))
    n_ctrl = tuple(int(np.asarray(lp["n_ctrl"])[i]) for i in range(8))
    pnr = tuple(
        tuple(int(x) for x in np.asarray(lp["pnr"][i])[: n_ctrl[i]])
        for i in range(8)
    )
    return active, n_ctrl, pnr


def jax_equivalent_gaussian(params: Dict[str, Any],
                            structure: Structure = None) -> Dict[str, Any]:
    """Differentiable JAX mirror of
    ``frontend.gaussian_decomposition.compute_equivalent_gaussian(light=True)``.

    Returns the full pre-PNR Gaussian state (cov, mu) over the surviving modes
    (1 signal + all active-leaf control modes, post point-homodyne) plus the
    signal/control bookkeeping and per-node homodyne densities.

    ``structure`` (leaf_active, n_ctrl, pnr) is STATIC -- read concretely from
    ``params`` when None (eager use), or supplied by the caller so this traces
    to a static graph over the continuous params (r/phases/disp/theta/phi/
    homodyne_x/final_gauss), which carry exact gradients.
    """
    lp = params["leaf_params"]
    if structure is None:
        structure = extract_structure(params)
    active_t, n_ctrl_t, pnr_t = structure
    leaf_active = [bool(x) for x in active_t]
    n_ctrl = [int(x) for x in n_ctrl_t]

    # --- per-leaf moments + mode bookkeeping (mirror numpy assembly) --------- #
    blocks_V: List[Tuple[int, jnp.ndarray]] = []
    blocks_mu: List[Tuple[int, jnp.ndarray]] = []
    modes: List[Dict[str, Any]] = []
    tot = 0
    for i in range(8):
        if not leaf_active[i]:
            continue
        N_leaf = n_ctrl[i] + 1
        tot += N_leaf
        mu_leaf, V_leaf = _leaf_moments(lp["r"][i], lp["phases"][i], lp["disp"][i], N_leaf)
        blocks_V.append((N_leaf, V_leaf))
        blocks_mu.append((N_leaf, mu_leaf))
        pnr_i = [int(x) for x in pnr_t[i]][: n_ctrl[i]]
        modes.append({"type": "signal", "leaf": i})
        for c in range(n_ctrl[i]):
            modes.append({"type": "control", "leaf": i, "pnr_val": pnr_i[c]})

    mu_g, V_g = _global_assemble(blocks_V, blocks_mu, tot)
    N_cur = tot

    # --- homodyne values broadcast to 7 nodes -------------------------------- #
    hx_raw = params.get("homodyne_x", 0.0)
    hx_arr = jnp.atleast_1d(jnp.asarray(hx_raw, dtype=jnp.float64))
    if hx_arr.shape[0] == 1:
        hx_values = [hx_arr[0]] * 7
    else:
        hx_values = [hx_arr[j] if j < hx_arr.shape[0] else jnp.array(0.0)
                     for j in range(7)]

    def signal_index(leaf_idx: int) -> int:
        for i, m in enumerate(modes):
            if m["type"] == "signal" and m["leaf"] == leaf_idx:
                return i
        return -1

    active_signals = [i if leaf_active[i] else None for i in range(8)]
    homodyne_densities: List[jnp.ndarray] = []

    # --- mixing tree (static schedule; concrete active routing) -------------- #
    mix_node = 0
    for _layer, num_pairs in [(1, 4), (2, 2), (3, 1)]:
        nxt: List[Any] = []
        for j in range(num_pairs):
            theta = params["mix_params"][mix_node][0]
            phi = params["mix_params"][mix_node][1]
            hx = hx_values[mix_node]
            a_leaf = active_signals[2 * j]
            b_leaf = active_signals[2 * j + 1]
            if a_leaf is not None and b_leaf is not None:
                idxA = signal_index(a_leaf)
                idxB = signal_index(b_leaf)
                S = _bs_symplectic(theta, phi, N_cur, idxA, idxB)
                V_g = S @ V_g @ S.T
                mu_g = S @ mu_g
                V_g, mu_g, dens = _measure_homodyne(V_g, mu_g, idxB, N_cur, hx)
                homodyne_densities.append(dens)
                modes.pop(idxB)
                N_cur -= 1
                nxt.append(a_leaf)
            elif a_leaf is not None:
                nxt.append(a_leaf)
            elif b_leaf is not None:
                nxt.append(b_leaf)
            else:
                nxt.append(None)
            mix_node += 1
        active_signals = nxt

    # --- final Gaussian on the root signal ----------------------------------- #
    fg = params.get("final_gauss", None)
    if fg:
        root_leaf = [s for s in active_signals if s is not None][0]
        root_idx = signal_index(root_leaf)
        V_g, mu_g = _apply_final_gaussian(V_g, mu_g, fg, root_idx, N_cur)

    # --- signal / control bookkeeping ---------------------------------------- #
    signal_idx = None
    control_idx: List[int] = []
    pnr_outcomes: List[int] = []
    for i, m in enumerate(modes):
        if m["type"] == "control":
            control_idx.append(i)
            pnr_outcomes.append(int(m.get("pnr_val", 0)))
        elif signal_idx is None:
            signal_idx = i

    return {
        "cov": V_g,
        "mu": mu_g,
        "num_final_modes": N_cur,
        "signal_idx": signal_idx,
        "control_idx": control_idx,
        "pnr_outcomes": pnr_outcomes,
        "homodyne_densities": homodyne_densities,
    }


# =========================================================================== #
# Piece 2b: jax_reduced_herald -- analytic vacuum conditioning + multidim
# Hermite recurrence.  Bit-exact JAX mirror of gbs_optimizer.reduced_herald /
# _gaussian_amplitudes (SAME convention, not just up-to-conjugation).
#
# Budget is on tensor amplitudes prod(n_j+1)*L, never on mode count: the box
# uses each fired mode's ACTUAL n_j+1, so many fired modes with modest photons
# are cheap.  Discrete fired-shape is static (per-bucket trace).
# =========================================================================== #
from src.simulation.jax.herald import Amat as _Amat, Qmat as _Qmat, \
    complex_to_real_displacements as _c2r


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=256)
def _base_slab_schedule(sub: Tuple[int, ...]):
    """Anti-diagonal fill schedule for the fired-mode box (signal index 0),
    cached per static ``sub`` = (n_1+1,...,n_kf+1).

    The box entry ``idx`` (first nonzero axis i, global axis ai=i+1) obeys
        full = idx - e_i;  R0[idx] =
          ( m_vec[ai]*R0[full] + Σ_a A[ai,a+1]*sqrt(full[a])*R0[full-e_a] )
          / sqrt(idx[i])
    Entries at total-photon level t = Σ idx depend only on levels t-1 (full) and
    t-2 (full-e_a), so we fill level-by-level: T = Σ n_j SEQUENTIAL layers, each
    of <= W entries filled in PARALLEL.  This replaces the O(∏) fori_loop with an
    O(Σn) lax.scan -- cheap, and crucially reverse-mode-AD-friendly.

    Returns padded (T, W) tables (or None if the box is a single point).
    """
    kf = len(sub)
    P = int(np.prod(sub)) if kf else 1
    if kf == 0 or P == 1:
        return None
    idxs = list(np.ndindex(*sub))
    flat = {idx: int(np.ravel_multi_index(idx, sub)) for idx in idxs}
    levels: Dict[int, list] = {}
    for idx in idxs:
        levels.setdefault(sum(idx), []).append(idx)
    T = max(levels)
    W = max((len(levels.get(t, [])) for t in range(1, T + 1)), default=0)
    q = np.zeros((T, W), np.int32)
    ai = np.zeros((T, W), np.int32)
    inv = np.zeros((T, W), np.float64)
    mp = np.zeros((T, W), np.int32)
    pA = np.zeros((T, W, kf), np.int32)
    sA = np.zeros((T, W, kf), np.float64)
    mask = np.zeros((T, W), np.float64)
    for ti, t in enumerate(range(1, T + 1)):
        for wi, idx in enumerate(levels.get(t, [])):
            i = next(a for a, v in enumerate(idx) if v > 0)
            full = list(idx); full[i] -= 1
            q[ti, wi] = flat[idx]; ai[ti, wi] = i + 1
            inv[ti, wi] = 1.0 / np.sqrt(idx[i])
            mp[ti, wi] = flat[tuple(full)]
            for a in range(kf):
                if full[a] > 0:
                    nm = list(full); nm[a] -= 1
                    pA[ti, wi, a] = flat[tuple(nm)]
                    sA[ti, wi, a] = np.sqrt(full[a])
            mask[ti, wi] = 1.0
    return dict(T=T, W=W, P=P, kf=kf, q=q, ai=ai, inv=inv, mp=mp, pA=pA,
                sA=sA, mask=mask)


def _gaussian_amplitudes_jax(cov: jnp.ndarray, mu: jnp.ndarray,
                             cuts: Tuple[int, ...]) -> jnp.ndarray:
    """<n|psi> of a pure M-mode Gaussian on a tensor with per-mode cutoffs
    ``cuts`` (static).  axis 0 = signal (length L), axes 1.. = fired modes.
    Differentiable in (cov, mu).  Mirrors numpy _gaussian_amplitudes exactly."""
    M = len(cuts)
    L = int(cuts[0])
    sub = tuple(int(c) for c in cuts[1:])
    cdt = jnp.complex128

    beta = _c2r(mu, HBAR)
    B = _Amat(cov, HBAR)[:M, :M]
    alpha = beta[:M]
    gamma = jnp.conj(alpha) - B @ alpha
    pref = jnp.exp(jnp.conj(-0.5 * (jnp.sum(jnp.abs(alpha) ** 2) - alpha @ B @ alpha)))
    detQ = jnp.real(jnp.linalg.det(_Qmat(cov, HBAR)))
    denom = jnp.sqrt(jnp.sqrt(detQ))
    A = jnp.conj(B).astype(cdt)
    m_vec = jnp.conj(gamma).astype(cdt)

    # ---- base slab (signal index 0): anti-diagonal lax.scan over levels ---- #
    sched = _base_slab_schedule(sub)
    P = int(np.prod(sub)) if sub else 1
    R0 = jnp.zeros(P, dtype=cdt).at[0].set(1.0 + 0j)
    if sched is not None:
        cols = jnp.arange(sched["kf"]) + 1            # global A-columns of fired axes
        q_a = jnp.asarray(sched["q"]); ai_a = jnp.asarray(sched["ai"])
        inv_a = jnp.asarray(sched["inv"]).astype(cdt); mp_a = jnp.asarray(sched["mp"])
        pA_a = jnp.asarray(sched["pA"]); sA_a = jnp.asarray(sched["sA"]).astype(cdt)
        mask_a = jnp.asarray(sched["mask"]).astype(cdt)

        def level(R, x):
            q_t, ai_t, inv_t, mp_t, pA_t, sA_t, mk_t = x   # each leading dim W
            main = m_vec[ai_t] * R[mp_t]                    # (W,)
            Acols = A[ai_t][:, cols]                        # (W, kf)
            contrib = jnp.sum(Acols * sA_t * R[pA_t], axis=1)
            vals = (main + contrib) * inv_t
            R = R.at[q_t].set(jnp.where(mk_t != 0, vals, R[q_t]))
            return R, None

        R0, _ = jax.lax.scan(level, R0, (q_a, ai_a, inv_a, mp_a, pA_a, sA_a, mask_a))

    box0 = R0.reshape(sub) if sub else R0.reshape(())

    # ---- signal axis (m >= 1): lax.scan over L ----------------------------- #
    def shift_up(arr, axis):
        sl_to = [slice(None)] * len(sub); sl_to[axis] = slice(1, None)
        sl_from = [slice(None)] * len(sub); sl_from[axis] = slice(0, -1)
        return jnp.zeros_like(arr).at[tuple(sl_to)].set(arr[tuple(sl_from)])

    sqrtk = []
    for axis in range(len(sub)):
        w = jnp.sqrt(jnp.arange(sub[axis])).astype(cdt)
        shape = [sub[axis] if a == axis else 1 for a in range(len(sub))]
        sqrtk.append(w.reshape(shape))

    def step(carry, m):
        prev1, prev2 = carry
        slab = m_vec[0] * prev1
        slab = slab + jnp.where(m >= 2, A[0, 0] * jnp.sqrt(jnp.maximum(m - 1, 0.0)), 0.0) * prev2
        for axis in range(len(sub)):
            slab = slab + A[0, axis + 1] * sqrtk[axis] * shift_up(prev1, axis)
        slab = slab / jnp.sqrt(m.astype(jnp.float64))
        return (slab, prev1), slab

    if L > 1:
        ms = jnp.arange(1, L)
        _, slabs = jax.lax.scan(step, (box0, jnp.zeros_like(box0)), ms)
        R = jnp.concatenate([box0[None], slabs], axis=0)
    else:
        R = box0[None]
    return pref * R / denom


def _vacuum_condition(cov: jnp.ndarray, mu: jnp.ndarray,
                      keep: List[int], vac: List[int]
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Analytically condition the n=0 (vacuum-detected) modes ``vac`` out of a
    pure Gaussian state (Schur complement, measurement covariance = I, hbar=2),
    keeping modes ``keep``.  Returns (cov_r, mu_r, p_vac).  Mirrors the vacuum
    block of gbs_optimizer.reduced_herald exactly."""
    N = cov.shape[0] // 2
    if not vac:
        ki = jnp.asarray([k for k in keep] + [k + N for k in keep])
        return cov[jnp.ix_(ki, ki)], mu[ki], jnp.array(1.0)
    ki = jnp.asarray([k for k in keep] + [k + N for k in keep])
    vi = jnp.asarray([v for v in vac] + [v + N for v in vac])
    Vk = cov[jnp.ix_(ki, ki)]
    Vv = cov[jnp.ix_(vi, vi)]
    Vkv = cov[jnp.ix_(ki, vi)]
    Mm = Vv + jnp.eye(vi.shape[0])
    rhs = jnp.column_stack([Vkv.T, mu[vi]])
    sol = jnp.linalg.solve(Mm, rhs)
    cov_r = Vk - Vkv @ sol[:, :-1]
    cov_r = 0.5 * (cov_r + cov_r.T)
    mu_r = mu[ki] - Vkv @ sol[:, -1]
    p_vac = (2.0 ** len(vac) / jnp.sqrt(jnp.linalg.det(Mm))
             * jnp.exp(-0.5 * mu[vi] @ jnp.linalg.solve(Mm, mu[vi])))
    return cov_r, mu_r, p_vac


# Max fired-box amplitudes prod(n_j+1) handled in-loop. The base-slab fill is
# O(prod) sequential (lax.fori_loop), so this also bounds TIME, not just memory.
# Fired patterns above this (the high-Sigma-n extreme tail, ~1-8% of genotypes,
# vanishing probability) are routed to the exact CPU reduced_herald fallback by
# the caller (MOMENT_SCORER_PLAN.md Sec. 2b).  prod is data-independent within a
# structure bucket, so this is a static, per-bucket decision.
REDUCED_HERALD_PROD_BUDGET = 1 << 14   # 16384


def fired_product(n: Tuple[int, ...]) -> int:
    """prod(n_j+1) over fired (n>=1) control modes -- the fired-box size."""
    p = 1
    for nv in n:
        if int(nv) >= 1:
            p *= int(nv) + 1
    return p


def jax_reduced_herald(cov: jnp.ndarray, mu: jnp.ndarray, signal_idx: int,
                       control_idx: Tuple[int, ...], n: Tuple[int, ...],
                       cutoff: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Heralded signal state + herald probability, exact at any squeezing.
    JAX mirror of gbs_optimizer.reduced_herald.  ``control_idx`` and ``n`` are
    static (per-bucket); ``cov``/``mu`` traced.  Returns (psi[cutoff], prob)."""
    N = cov.shape[0] // 2
    fired = [(int(c), int(nv)) for c, nv in zip(control_idx, n) if int(nv) >= 1]
    vac = [int(c) for c, nv in zip(control_idx, n) if int(nv) == 0]
    keep = [int(signal_idx)] + [c for c, _ in fired]

    cov_r, mu_r, p_vac = _vacuum_condition(cov, mu, keep, vac)
    cuts = tuple([int(cutoff)] + [nv + 1 for _, nv in fired])
    box = _gaussian_amplitudes_jax(cov_r, mu_r, cuts)
    sl = (slice(None),) + tuple(nv for _, nv in fired)
    v = box[sl].ravel()
    p_fock = jnp.sum(jnp.abs(v) ** 2)
    prob = p_vac * p_fock
    psi = jnp.where(p_fock > 0, v / jnp.sqrt(jnp.maximum(p_fock, 1e-300)), v)
    return psi, prob


# =========================================================================== #
# Piece 3a: differentiable per-genotype scoring kernel.
#   exp = <O> on the exact heralded state; P_leaf = product of exact per-leaf
#   herald probs (the optimizer's truncation-free 'leaf' probability).
# Structure is STATIC => this traces to one graph per fired/active signature,
# vmappable within a bucket and value_and_grad-able for the gradient emitters.
# =========================================================================== #
def _leaf_herald_prob(lp: Dict[str, Any], i: int, n_ctrl_i: int,
                      pnr_i: Tuple[int, ...], cutoff: int) -> jnp.ndarray:
    """Exact moment-space herald probability of one leaf (signal=0, controls
    1..n_ctrl)."""
    if n_ctrl_i == 0:
        return jnp.array(1.0)
    N = n_ctrl_i + 1
    mu, cov = _leaf_moments(lp["r"][i], lp["phases"][i], lp["disp"][i], N)
    _, p = jax_reduced_herald(cov, mu, 0, tuple(range(1, N)),
                              tuple(int(x) for x in pnr_i[:n_ctrl_i]), cutoff)
    return p


def moment_score_one(params: Dict[str, Any], operator: jnp.ndarray,
                     structure: Structure, cutoff: int
                     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Score one genotype's decoded ``params`` in moment space.

    Returns (exp, prob_pnr, P_leaf, info):
      exp      = <O> on the exact heralded signal state (real)
      prob_pnr = exact PNR herald probability on the equivalent generator
      P_leaf   = product of exact per-leaf herald probs (fitness 'leaf' P)
      info     = {signal_idx, control_idx, pnr_outcomes, fired_product, cov, mu}

    ``operator`` is the L x L target operator (L = ``cutoff`` here, the FINAL
    single-mode Fock cutoff -- large & cheap, e.g. 100). Differentiable in the
    continuous params; ``structure`` is static.
    """
    eq = jax_equivalent_gaussian(params, structure)
    sig = eq["signal_idx"]
    ctrl = tuple(int(c) for c in eq["control_idx"])
    n0 = tuple(int(x) for x in eq["pnr_outcomes"])
    psi, prob_pnr = jax_reduced_herald(eq["cov"], eq["mu"], sig, ctrl, n0, cutoff)
    Lp = psi.shape[0]
    Oe = operator[:Lp, :Lp]
    exp = jnp.real(jnp.vdot(psi, Oe @ psi))

    active, n_ctrl_t, pnr_t = structure
    P_leaf = jnp.array(1.0)
    lp = params["leaf_params"]
    for i in range(8):
        if active[i] and n_ctrl_t[i] > 0:
            P_leaf = P_leaf * _leaf_herald_prob(lp, i, n_ctrl_t[i], pnr_t[i], cutoff)

    info = {"signal_idx": sig, "control_idx": ctrl, "pnr_outcomes": n0,
            "fired_product": fired_product(n0), "cov": eq["cov"], "mu": eq["mu"]}
    return exp, prob_pnr, P_leaf, info


# =========================================================================== #
# Piece 3b/3c: batched, structure-bucketed population scorer producing the same
# (fitnesses, descriptors, extras) contract as runner._score_batch_shard, so it
# drops into jax_scoring_fn_batch behind config "scorer"="moment".
# =========================================================================== #
from functools import partial as _partial


@_lru_cache(maxsize=8)
def moment_operator(L: int, alpha_str: str, beta_str: str) -> jnp.ndarray:
    """Cached L x L GKP target operator (jax) for the moment scorer's <O>."""
    from src.utils.gkp_operator import construct_gkp_operator
    a = complex(str(alpha_str).replace("i", "j"))
    b = complex(str(beta_str).replace("i", "j"))
    return construct_gkp_operator(int(L), a, b, backend="jax")


def _struct_fired_product(structure: Structure) -> int:
    """prod(n+1) over fired control modes, from the discrete structure alone."""
    active, n_ctrl, pnr = structure
    p = 1
    for i in range(8):
        if not active[i]:
            continue
        for c in range(n_ctrl[i]):
            if int(pnr[i][c]) >= 1:
                p *= int(pnr[i][c]) + 1
    return p


def _effective_photons(cov: jnp.ndarray, signal_idx: int,
                       control_idx: Tuple[int, ...], n0: Tuple[int, ...],
                       eps: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Effective (coupled) photons from the final equivalent-Gaussian cov: each
    fired mode's detected count weighted by a smooth gate on its covariance
    coupling to (signal + other controls).  Differentiable; closes the dud-photon
    exploit at the source (matches rescore's coupling audit)."""
    N = cov.shape[0] // 2
    n_eff = jnp.array(0.0)
    max_eff = jnp.array(0.0)
    for j, c in enumerate(control_idx):
        if int(n0[j]) < 1:
            continue
        others = [int(signal_idx)] + [c2 for k2, c2 in enumerate(control_idx) if k2 != j]
        oi = jnp.asarray([i for o in others for i in (int(o), int(o) + N)])
        rows = cov[jnp.asarray([int(c), int(c) + N])][:, oi]
        cpl = jnp.sqrt(jnp.sum(rows ** 2))
        w = cpl ** 2 / (cpl ** 2 + eps ** 2)
        ne = int(n0[j]) * w
        n_eff = n_eff + ne
        max_eff = jnp.maximum(max_eff, ne)
    return n_eff, max_eff


@_partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def _bucket_score(gs: jnp.ndarray, operator_L: jnp.ndarray, structure: Structure,
                  genotype_name: str, config_hashable: tuple,
                  base_cutoff: int, L: int, floats: tuple):
    """Score one bucket of same-structure genotypes: vmap(value_and_grad(loss)).
    Returns (fitnesses[nb,4], descriptors[nb,3], gradients[nb,D],
    raw_exp[nb], joint_prob[nb]).  Mirrors _score_batch_shard's loss/fitness
    assembly (Tchebycheff), minus the truncation-only penalties (none here)."""
    gs_eig, gaussian_limit, w_exp, w_prob, coupling_eps = floats
    cfg = dict(config_hashable)
    decoder = get_genotype_decoder(genotype_name, depth=int(cfg.get("depth", 3)),
                                   config=cfg)
    active_modes = jnp.asarray(float(sum(bool(x) for x in structure[0])))

    def loss(g):
        params = decoder.decode(g, base_cutoff)
        exp, _prob_pnr, P_leaf, info = moment_score_one(params, operator_L,
                                                        structure, L)
        raw_exp = jnp.real(exp)
        joint_prob = jnp.real(P_leaf)
        n_eff, max_eff = _effective_photons(info["cov"], info["signal_idx"],
                                            info["control_idx"],
                                            info["pnr_outcomes"], coupling_eps)
        exp_val = jnp.where(joint_prob > 1e-40, raw_exp, 0.0)
        prob_capped = jnp.minimum(jnp.maximum(joint_prob, 1e-45), 1.0)
        log_prob = -jnp.log10(prob_capped)
        log_prob = log_prob + jnp.where(jnp.maximum(joint_prob - 1.0, 0.0) > 1e-4,
                                        jnp.inf, 0.0)
        # physics artifact guard: sub-Gaussian <O> with no EFFECTIVE photons
        is_artifact = jnp.logical_and(exp_val < gaussian_limit, n_eff < 0.5)
        art = jnp.where(is_artifact, jnp.inf, 0.0)
        exp_val = exp_val + art
        log_prob = log_prob + art
        eps0 = 0.01
        d_e = w_exp * jnp.abs(exp_val - (gs_eig - eps0))
        d_p = w_prob * jnp.abs(log_prob - (-eps0))
        loss_val = jnp.maximum(d_e, d_p) + 0.01 * (d_e + d_p)
        final_exp = jnp.where(joint_prob > 1e-40, raw_exp, jnp.inf) + art
        aux = dict(final_exp=final_exp, log_prob=log_prob, max_pnr=max_eff,
                   photons=n_eff, raw_exp=raw_exp, joint_prob=joint_prob)
        return jnp.real(loss_val), aux

    (_lv, aux), grad = jax.vmap(jax.value_and_grad(loss, has_aux=True))(gs)
    ones = jnp.ones_like(aux["log_prob"])
    fit = jnp.stack([-aux["final_exp"], -aux["log_prob"],
                     -active_modes * ones, -aux["photons"]], axis=1)
    desc = jnp.stack([active_modes * ones, aux["max_pnr"], aux["photons"]], axis=1)
    return fit, desc, grad, aux["raw_exp"], aux["joint_prob"]


def moment_score_population(genotypes, operator_L, genotype_name: str,
                            config_hashable: tuple, base_cutoff: int, L: int,
                            gs_eig: float, gaussian_limit: float):
    """Structure-bucketed moment-space scorer. Same return contract as
    runner._score_batch_shard: (fitnesses[N,4], descriptors[N,3], extras).
    Over-budget (extreme-Σn) genotypes are scored exactly via the numpy
    reduced_herald fallback with zero gradient (graceful, kept exact)."""
    cfg = dict(config_hashable)
    w_exp = float(cfg.get("alpha_expectation", 1.0))
    w_prob = float(cfg.get("alpha_probability", 0.0))
    coupling_eps = float(cfg.get("coupling_eps", 0.05))
    floats = (float(gs_eig), float(gaussian_limit), w_exp, w_prob, coupling_eps)

    g_np = np.asarray(genotypes)
    Npop, D = g_np.shape
    decoder = get_genotype_decoder(genotype_name, depth=int(cfg.get("depth", 3)),
                                   config=cfg)
    structs = []
    for i in range(Npop):
        params = decoder.decode(jnp.asarray(g_np[i]), base_cutoff)
        structs.append(extract_structure(params))

    fit = np.full((Npop, 4), -np.inf)
    desc = np.zeros((Npop, 3))
    grads = np.zeros((Npop, D))
    rawe = np.zeros(Npop)
    jprob = np.zeros(Npop)

    buckets: Dict[Structure, list] = {}
    over = []
    for i, st in enumerate(structs):
        if _struct_fired_product(st) > REDUCED_HERALD_PROD_BUDGET:
            over.append(i)
        else:
            buckets.setdefault(st, []).append(i)

    for st, idxs in buckets.items():
        f, d, gr, re, jp = _bucket_score(
            jnp.asarray(g_np[idxs]), operator_L, st, genotype_name,
            config_hashable, int(base_cutoff), int(L), floats)
        f = np.asarray(f); d = np.asarray(d); gr = np.asarray(gr)
        re = np.asarray(re); jp = np.asarray(jp)
        for k, ii in enumerate(idxs):
            fit[ii] = f[k]; desc[ii] = d[k]; grads[ii] = gr[k]
            rawe[ii] = re[k]; jprob[ii] = jp[k]

    if over:
        _score_over_budget_numpy(over, g_np, decoder, cfg, np.asarray(operator_L),
                                 base_cutoff, L, w_prob, fit, desc, rawe, jprob)

    extras = {
        "gradients": jnp.asarray(grads),
        "raw_expectation": jnp.asarray(rawe),
        "joint_probability": jnp.asarray(jprob),
        "leakage": jnp.zeros(Npop),
        "pnr_cost": jnp.asarray(desc[:, 2]),
        "final_state": jnp.zeros((Npop, base_cutoff), dtype=jnp.complex128),
    }
    return jnp.asarray(fit), jnp.asarray(desc), extras


def _numpy_leaf_prob_product(params, L):
    """Exact product of per-leaf herald probs (numpy), the 'leaf' fitness P."""
    from frontend.independent_verifier import _build_gaussian_moments
    from frontend.gbs_optimizer import reduced_herald
    lp = params["leaf_params"]
    P = 1.0
    for i in range(8):
        if not bool(np.asarray(params["leaf_active"])[i]):
            continue
        nc = int(np.asarray(lp["n_ctrl"])[i])
        if nc == 0:
            continue
        N = nc + 1
        r = np.asarray(lp["r"][i], float)[:N]
        ph = np.asarray(lp["phases"][i], float)[:N * N]
        dv = np.asarray(lp["disp"][i])[:N]
        pnr = [int(x) for x in np.asarray(lp["pnr"][i])[:nc]]
        mu, cov = _build_gaussian_moments(r, ph, dv, N)
        _, p = reduced_herald(cov, mu, 0, list(range(1, N)), pnr, cutoff=L)
        P *= float(p)
    return P


def _score_over_budget_numpy(over, g_np, decoder, cfg, O_np, base_cutoff, L,
                             w_prob, fit, desc, rawe, jprob):
    """Exact CPU scoring for the rare extreme-Σn tail (zero gradient): keeps
    these solutions scored correctly so they aren't artificially penalised."""
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend.gbs_optimizer import reduced_herald
    eps = float(cfg.get("coupling_eps", 0.05))
    for i in over:
        try:
            params = decoder.decode(jnp.asarray(g_np[i]), base_cutoff)
            pnp = {k: (np.asarray(v) if hasattr(v, "shape") else v)
                   for k, v in params.items()}
            eq = compute_equivalent_gaussian(pnp, light=True)
            cov = np.asarray(eq["cov"], float); mu = np.asarray(eq["mu"], float)
            Nm = cov.shape[0] // 2
            ctrl = [int(x) for x in eq["control_idx"]]; sig = int(eq["signal_idx"])
            n0 = [int(x) for x in eq["pnr_outcomes"]]
            psi, _ = reduced_herald(cov, mu, sig, ctrl, n0, cutoff=L)
            if not np.isfinite(psi).all() or np.linalg.norm(psi) < 0.5:
                continue
            Le = len(psi)
            exp = float(np.real(np.vdot(psi, O_np[:Le, :Le] @ psi)))
            # effective photons (coupling audit)
            n_eff = 0.0; max_eff = 0.0
            for j, c in enumerate(ctrl):
                if n0[j] < 1:
                    continue
                others = [sig] + [c2 for k2, c2 in enumerate(ctrl) if k2 != j]
                oi = [m for o in others for m in (o, o + Nm)]
                cpl = float(np.linalg.norm(cov[np.ix_([c, c + Nm], oi)]))
                w = cpl ** 2 / (cpl ** 2 + eps ** 2)
                n_eff += n0[j] * w; max_eff = max(max_eff, n0[j] * w)
            P_leaf = _numpy_leaf_prob_product(params, L)
            active = float(sum(bool(x) for x in np.asarray(params["leaf_active"])))
            log_prob = -np.log10(min(max(P_leaf, 1e-45), 1.0))
            fit[i] = [-exp, -log_prob, -active, -n_eff]
            desc[i] = [active, max_eff, n_eff]
            rawe[i] = exp; jprob[i] = P_leaf
        except Exception:
            continue
