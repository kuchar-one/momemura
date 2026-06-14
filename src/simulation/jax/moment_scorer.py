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


def jax_equivalent_gaussian(params: Dict[str, Any]) -> Dict[str, Any]:
    """Differentiable JAX mirror of
    ``frontend.gaussian_decomposition.compute_equivalent_gaussian(light=True)``.

    Returns the full pre-PNR Gaussian state (cov, mu) over the surviving modes
    (1 signal + all active-leaf control modes, post point-homodyne) plus the
    signal/control bookkeeping and per-node homodyne densities.

    The discrete structure (leaf_active, n_ctrl, pnr) is read concretely; the
    continuous params are traced => gradients flow to r/phases/disp/theta/phi/
    homodyne_x/final_gauss.
    """
    lp = params["leaf_params"]
    leaf_active = [bool(np.asarray(params["leaf_active"])[i]) for i in range(8)]
    n_ctrl = [int(np.asarray(lp["n_ctrl"])[i]) for i in range(8)]

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
        pnr_i = [int(x) for x in np.asarray(lp["pnr"][i])[: n_ctrl[i]]]
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


def _base_slab_tables(sub: Tuple[int, ...]):
    """Precompute (numpy, trace-time) the gather/coefficient tables that fill the
    fired-mode box at signal index 0, in C-order. ``sub`` = (n_1+1,...,n_kf+1).

    For box index ``idx`` with first nonzero axis ``i`` (global axis ai=i+1):
        full = idx - e_i  (the 'main' predecessor);  R0[idx] =
          ( m_vec[ai]*R0[full] + Σ_a A[ai,a+1]*sqrt(full[a])*R0[full-e_a] )
          / sqrt(idx[i])
    """
    kf = len(sub)
    P = int(np.prod(sub)) if kf else 1
    ai = np.zeros(P, np.int32)
    inv = np.zeros(P, np.float64)
    main_pred = np.zeros(P, np.int32)
    predA = np.zeros((P, max(kf, 1)), np.int32)
    sqrtA = np.zeros((P, max(kf, 1)), np.float64)
    if kf == 0:
        return ai, inv, main_pred, predA, sqrtA, P
    idxs = list(np.ndindex(*sub))
    for q, idx in enumerate(idxs):
        if all(v == 0 for v in idx):
            continue
        i = next(a for a, v in enumerate(idx) if v > 0)
        ai[q] = i + 1
        inv[q] = 1.0 / np.sqrt(idx[i])
        full = list(idx); full[i] -= 1
        main_pred[q] = int(np.ravel_multi_index(tuple(full), sub))
        for a in range(kf):
            if full[a] > 0:
                nm = list(full); nm[a] -= 1
                predA[q, a] = int(np.ravel_multi_index(tuple(nm), sub))
                sqrtA[q, a] = np.sqrt(full[a])
    return ai, inv, main_pred, predA, sqrtA, P


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

    # ---- base slab (signal index 0): fill the fired box -------------------- #
    ai, inv, main_pred, predA, sqrtA, P = _base_slab_tables(sub)
    R0 = jnp.zeros(P, dtype=cdt).at[0].set(1.0 + 0j)
    if P > 1:
        ai_j = jnp.asarray(ai); inv_j = jnp.asarray(inv).astype(cdt)
        mp_j = jnp.asarray(main_pred)
        pA_j = jnp.asarray(predA); sA_j = jnp.asarray(sqrtA).astype(cdt)
        cols = jnp.arange(len(sub)) + 1

        def body(q, R):
            a_i = ai_j[q]
            acc = m_vec[a_i] * R[mp_j[q]]
            row = A[a_i][cols]                       # A[ai, fired-axis globals]
            acc = acc + jnp.sum(row * sA_j[q] * R[pA_j[q]])
            return R.at[q].set(acc * inv_j[q])

        R0 = jax.lax.fori_loop(1, P, body, R0)

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
