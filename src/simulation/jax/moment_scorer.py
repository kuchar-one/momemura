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

import math
from functools import lru_cache as _lru_cache

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
    nleaves = int(np.asarray(params["leaf_active"]).shape[0])   # 2**depth
    active = tuple(bool(np.asarray(params["leaf_active"])[i]) for i in range(nleaves))
    n_ctrl = tuple(int(np.asarray(lp["n_ctrl"])[i]) for i in range(nleaves))
    pnr = tuple(
        tuple(int(x) for x in np.asarray(lp["pnr"][i])[: n_ctrl[i]])
        for i in range(nleaves)
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
    nleaves = len(leaf_active)                       # 2**depth
    depth = int(round(math.log2(nleaves)))
    nodes = nleaves - 1

    # --- per-leaf moments + mode bookkeeping (mirror numpy assembly) --------- #
    blocks_V: List[Tuple[int, jnp.ndarray]] = []
    blocks_mu: List[Tuple[int, jnp.ndarray]] = []
    modes: List[Dict[str, Any]] = []
    tot = 0
    for i in range(nleaves):
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

    # --- homodyne values broadcast to ``nodes`` nodes ------------------------ #
    hx_raw = params.get("homodyne_x", 0.0)
    hx_arr = jnp.atleast_1d(jnp.asarray(hx_raw, dtype=jnp.float64))
    if hx_arr.shape[0] == 1:
        hx_values = [hx_arr[0]] * nodes
    else:
        hx_values = [hx_arr[j] if j < hx_arr.shape[0] else jnp.array(0.0)
                     for j in range(nodes)]

    def signal_index(leaf_idx: int) -> int:
        for i, m in enumerate(modes):
            if m["type"] == "signal" and m["leaf"] == leaf_idx:
                return i
        return -1

    active_signals = [i if leaf_active[i] else None for i in range(nleaves)]
    homodyne_densities: List[jnp.ndarray] = []

    # --- mixing tree (static schedule; concrete active routing) -------------- #
    # layer k (k=1..depth) reduces pairs -> 2**(depth-k) nodes, in mix_params order
    mix_node = 0
    for _layer, num_pairs in [(k, 2 ** (depth - k)) for k in range(1, depth + 1)]:
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
    roots = [s for s in active_signals if s is not None]
    if fg and roots:
        root_idx = signal_index(roots[0])
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
# OPTION 2 -- single-compile STATIC equivalent-Gaussian.
#
# Everything is fixed-shape (8 leaves x 3 modes = 24; tree = 7 nodes), so the
# Python loops below UNROLL into ONE XLA graph -- no per-structure recompile.
# The discrete genotype info (leaf_active, n_ctrl, pnr) flows as TRACED jnp
# values used only inside jnp.where masks:
#   * inactive leaf / unused control mode -> vacuum & decoupled (mask squeezing,
#     displacement, and the Clements BS gates that touch the padding modes);
#   * tree routing -> masked BS angle per node: identity when one child subtree
#     is dead, a clean swap (theta=pi/2, phi=0) when only the RIGHT child is
#     live (so the surviving signal always lands in the fixed A slot, matching
#     the dynamic "propagate the active child, measure the other" logic).
# Output: fixed 17-mode (1 signal + 16 control-slot) cov/mu + effective control
# pnr (unused slots forced to 0 -> conditioned away trivially in piece B).
# =========================================================================== #
@_lru_cache(maxsize=8)
def _static_tree(depth: int):
    """Depth-parametric static topology for the masked single-compile scorer.

    Returns (leaf_x, tree, keep, nleaves, N):
      * leaf_x  : per-leaf x-mode triple (signal, c0, c1) in a 3-mode-per-leaf
                  layout, so leaf i owns x-modes (3i, 3i+1, 3i+2).
      * tree    : balanced binary reduction as (A_x, B_x, left_leaves,
                  right_leaves), in mix_params node order (layer 1 first: pairs
                  (0,1),(2,3),...; then layer 2; ...; root last).  A leaf-group's
                  surviving signal always sits at x-mode 3*min(group), so A_x =
                  3*min(left), B_x = 3*min(right); B is homodyned and retired.
      * keep    : signal(x0) + the 2*nleaves control slots in leaf order
                  (leaf0 c0, leaf0 c1, leaf1 c0, ...) -- same order the
                  per-structure scanner appends controls.
      * nleaves = 2**depth ; N = 3*nleaves (x-modes before the keep-projection).

    depth=3 reproduces the original 8-leaf / N=24 / 7-node hardcoding exactly.
    """
    nleaves = 2 ** depth
    N = 3 * nleaves
    leaf_x = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(nleaves)]
    tree = []
    groups = [[i] for i in range(nleaves)]          # signal of group at 3*group[0]
    while len(groups) > 1:
        nxt = []
        for j in range(0, len(groups), 2):
            gA, gB = groups[j], groups[j + 1]
            tree.append((3 * gA[0], 3 * gB[0], tuple(gA), tuple(gB)))
            nxt.append(gA + gB)
        groups = nxt
    keep = [0] + [m for i in range(nleaves) for m in (3 * i + 1, 3 * i + 2)]
    return leaf_x, tree, keep, nleaves, N


def _leaf_moments_masked(r: jnp.ndarray, ph: jnp.ndarray, dv: jnp.ndarray,
                         n_real: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """3-mode leaf moments with modes >= n_real masked to vacuum & decoupled.
    n_real (traced scalar) = active ? n_ctrl+1 : 0.  Bit-equivalent to building
    at the true mode count: the Clements gates touching padding modes are zeroed
    (theta->0 => identity BS) so those modes stay decoupled vacuum."""
    idx = jnp.arange(3)
    live = idx < n_real
    rm = jnp.where(live, r[:3], 0.0)
    dvm = jnp.where(live, dv[:3], 0.0 + 0j)
    # The decoder builds the leaf interferometer with the N-mode Clements
    # (phases[:N^2]); the gate layout differs per N, so reproduce each N exactly
    # and embed in 3 modes, then select by n_real (=active ? n_ctrl+1 : 0).
    def _embed(Uk, k):
        return jnp.eye(3, dtype=jnp.complex128).at[:k, :k].set(Uk)
    U1 = _embed(jax_clements_unitary(ph[:1], 1), 1)
    U2 = _embed(jax_clements_unitary(ph[:4], 2), 2)
    U3 = jax_clements_unitary(ph[:9], 3)
    U = jnp.where(n_real < 1.5, U1, jnp.where(n_real < 2.5, U2, U3))
    S_sq = jnp.diag(jnp.concatenate([jnp.exp(-rm), jnp.exp(rm)]))
    S = passive_unitary_to_symplectic(U) @ S_sq
    mu = complex_alpha_to_qp(dvm, HBAR)
    cov = S @ vacuum_covariance(3, HBAR) @ S.T
    return mu, cov


def _measure_homodyne_fixed(V, mu, b, N, x):
    """Point-homodyne on mode ``b`` (x-quadrature) keeping fixed 2N shape: Schur
    update on all modes, then retire b to decoupled vacuum.  Returns (V,mu,dens)."""
    B11 = V[b, b]
    c = V[:, b]
    Vn = V - jnp.outer(c, c) / B11
    mun = mu + (x - mu[b]) / B11 * c
    dens = jnp.exp(-0.5 * (x - mu[b]) ** 2 / B11) / jnp.sqrt(2.0 * jnp.pi * B11)
    for q in (b, b + N):
        Vn = Vn.at[q, :].set(0.0).at[:, q].set(0.0).at[q, q].set(1.0)
        mun = mun.at[q].set(0.0)
    return Vn, mun, dens


def jax_equivalent_gaussian_static(params: Dict[str, Any], depth: int = 3
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-compile, fully-masked JAX equivalent-Gaussian for any ``depth``.
    All genotype structure is traced (only ``depth`` is static) => one XLA graph
    per depth.  At ``depth`` there are nleaves=2**depth leaves, nodes=nleaves-1
    mixing nodes, and 2*nleaves control slots.

    Returns (cov[2M,2M], mu[2M], eff_pnr[2*nleaves], densities[nodes]) with
    M=1+2*nleaves:
      signal = mode 0; control slots 1..2*nleaves in leaf order (leaf0 c0,
      leaf0 c1, leaf1 c0, ...); eff_pnr[s] = pnr of that control slot, or 0 if
      the slot is unused/inactive (so piece B conditions it away as decoupled
      vacuum).  depth=3 reproduces the original cov17[34,34]/eff_pnr[16] output."""
    leaf_x, tree, keep_l, nleaves, N = _static_tree(int(depth))
    nodes = nleaves - 1

    lp = params["leaf_params"]
    active = jnp.asarray(params["leaf_active"]).astype(jnp.float64)   # (nleaves,)
    # Prefer the STE (straight-through) float variants when the decoder provides
    # them: forward values are IDENTICAL to the int fields (exact round), but the
    # backward pass sees d(round)/dx = 1, so any smooth downstream use of the
    # detector counts carries gradient back to the genotype.
    n_ctrl = jnp.asarray(lp.get("n_ctrl_ste", lp["n_ctrl"])).astype(jnp.float64)
    r = jnp.asarray(lp["r"]); ph = jnp.asarray(lp["phases"]); dv = jnp.asarray(lp["disp"])
    pnr = jnp.asarray(lp.get("pnr_ste", lp["pnr"]))                  # (nleaves, >=2)

    Vg = jnp.eye(2 * N)              # vacuum (hbar=2 => cov = I)
    mug = jnp.zeros(2 * N)
    for i in range(nleaves):
        n_real = jnp.where(active[i] > 0.5, n_ctrl[i] + 1.0, 0.0)
        mu_l, cov_l = _leaf_moments_masked(r[i], ph[i], dv[i], n_real)
        xs = leaf_x[i]
        idxs = jnp.asarray([xs[0], xs[1], xs[2],
                            xs[0] + N, xs[1] + N, xs[2] + N])
        Vg = Vg.at[jnp.ix_(idxs, idxs)].set(cov_l)
        mug = mug.at[idxs].set(mu_l)

    # homodyne broadcast to ``nodes`` nodes (Design0 supplies one x per node;
    # the global-homodyne designs supply a single value to broadcast)
    hx_raw = params.get("homodyne_x", 0.0)
    hx_arr = jnp.atleast_1d(jnp.asarray(hx_raw, dtype=jnp.float64))
    hx = hx_arr if hx_arr.shape[0] >= nodes else jnp.broadcast_to(hx_arr[:1], (nodes,))

    densities = []
    for node, (A, B, Lk, Rk) in enumerate(tree):
        La = jnp.max(jnp.stack([active[l] for l in Lk]))
        Ra = jnp.max(jnp.stack([active[l] for l in Rk]))
        both = (La > 0.5) & (Ra > 0.5)
        only_right = (Ra > 0.5) & (La <= 0.5)
        th0 = params["mix_params"][node][0]; ph0 = params["mix_params"][node][1]
        theta = jnp.where(both, th0, jnp.where(only_right, jnp.pi / 2.0, 0.0))
        phi = jnp.where(both, ph0, 0.0)
        S = _bs_symplectic(theta, phi, N, A, B)
        Vg = S @ Vg @ S.T
        mug = S @ mug
        Vg, mug, dens = _measure_homodyne_fixed(Vg, mug, B, N, hx[node])
        densities.append(dens)

    fg = params.get("final_gauss", None)
    if fg:
        Vg, mug = _apply_final_gaussian(Vg, mug, fg, 0, N)

    keep = jnp.asarray(keep_l)
    idxs = jnp.concatenate([keep, keep + N])
    cov_k = Vg[jnp.ix_(idxs, idxs)]
    mu_k = mug[idxs]

    # effective control pnr per slot (leaf order): real iff active & c < n_ctrl
    eff = []
    for i in range(nleaves):
        for c in range(2):
            real = (active[i] > 0.5) & (c < n_ctrl[i])
            eff.append(jnp.where(real, pnr[i, c].astype(jnp.float64), 0.0))
    eff_pnr = jnp.stack(eff)          # (2*nleaves,)
    return cov_k, mu_k, eff_pnr, jnp.stack(densities)


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


# =========================================================================== #
# OPTION 2 piece B: single-compile reduced_herald.
#   * vacuum-condition the non-fired controls (fixed Schur on a fixed mode set
#     after a fired-first reorder);
#   * Hermite box over (signal + MAXF fired slots) in a FIXED flat buffer whose
#     mixed-radix strides / predecessor indices are computed at RUNTIME from the
#     pnr-derived radii -> data-dependent INDEXING, not shapes => one XLA graph.
#   * fired-box base slab filled by a LEVEL-scan (<=T_MAX fixed steps), signal
#     axis by a separate lax.scan over L.  Both keep AD memory ~O(steps*B_F).
# Genotypes with kf>MAXF or prod(n_j+1)>B_F route to the exact CPU fallback.
# =========================================================================== #
MOMENT_MAXF = 8                 # max fired control modes handled in-graph
MOMENT_BF = 1 << 12             # fired-box flat buffer (prod(n_j+1) budget) = 4096
_T_MAX = MOMENT_MAXF * 15       # max total fired photons (pnr_max=15)


def _flat_amplitudes(cov, mu, radii, L, MAXF=MOMENT_MAXF, BF=MOMENT_BF):
    """Signal amplitudes psi[L] at the detection slice (each fired axis j at
    k_j = radii[j]-1), for a pure (1+MAXF)-mode Gaussian.  ``radii`` (MAXF,)
    traced ints (n_j+1, =1 for unused slots).  Fixed flat buffer BF.
    Same convention as _gaussian_amplitudes_jax (bit-exact)."""
    T_max = MAXF * 15
    M = 1 + MAXF
    cdt = jnp.complex128
    beta = _c2r(mu, HBAR)
    B = _Amat(cov, HBAR)[:M, :M]
    alpha = beta[:M]
    gamma = jnp.conj(alpha) - B @ alpha
    pref = jnp.exp(jnp.conj(-0.5 * (jnp.sum(jnp.abs(alpha) ** 2) - alpha @ B @ alpha)))
    denom = jnp.sqrt(jnp.sqrt(jnp.real(jnp.linalg.det(_Qmat(cov, HBAR)))))
    A = jnp.conj(B).astype(cdt)
    m_vec = jnp.conj(gamma).astype(cdt)

    r = radii.astype(jnp.int32)                       # (MAXF,)
    # C-order strides over the MAXF fired axes (stride[j] = prod_{k>j} r[k])
    rev = jnp.concatenate([jnp.ones((1,), jnp.int32), jnp.cumprod(r[::-1])[:-1]])
    stride = rev[::-1]                                # (MAXF,)
    prodf = stride[0] * r[0]
    det_flat = jnp.clip(jnp.sum((r - 1) * stride), 0, BF - 1)   # clamp (over-budget safety)

    q = jnp.arange(BF)
    kjq = (q[None, :] // stride[:, None]) % r[:, None]    # (MAXF, BF) k_j(q)
    valid = q < prodf
    level = jnp.sum(kjq, axis=0)                          # (BF,)
    pos = kjq > 0                                         # (MAXF, BF)
    i_q = jnp.argmax(pos, axis=0)                         # first nonzero axis
    stride_i = stride[i_q]                               # (BF,)
    full = q - stride_i                                   # main predecessor (BF,)
    ki = jnp.take_along_axis(kjq, i_q[None, :], 0)[0]     # k_i(q)
    inv = jnp.where(ki > 0, 1.0 / jnp.sqrt(jnp.maximum(ki, 1)), 0.0).astype(cdt)
    ai = i_q + 1                                          # mode index of axis i
    mvec_ai = m_vec[ai]                                   # (BF,)
    fullc = jnp.clip(full, 0, BF - 1)

    # ---- base slab (signal index 0): level-scan ---------------------------- #
    R0 = jnp.zeros(BF, dtype=cdt).at[0].set(1.0 + 0j)

    def base_step(R, t):
        term = mvec_ai * R[fullc]
        for a in range(MAXF):
            full_a = kjq[a] - (i_q == a).astype(jnp.int32)     # (full)[a]
            preda = jnp.clip(full - stride[a], 0, BF - 1)
            coef = A[ai, a + 1] * jnp.sqrt(jnp.maximum(full_a, 0).astype(cdt))
            term = term + jnp.where(full_a > 0, coef * R[preda], 0.0 + 0j)
        val = term * inv
        commit = (level == t) & valid
        return jnp.where(commit, val, R), None

    R0, _ = jax.lax.scan(base_step, R0, jnp.arange(1, T_max + 1))

    # ---- signal axis: lax.scan over L (vectorised over fired buffer) -------- #
    sqrt_k = [jnp.sqrt(kjq[j].astype(cdt)) for j in range(MAXF)]
    predj = [jnp.clip(q - stride[j], 0, BF - 1) for j in range(MAXF)]

    def sig_step(carry, m):
        R1, R2 = carry
        slab = m_vec[0] * R1
        slab = slab + jnp.where(m >= 2, A[0, 0] * jnp.sqrt(jnp.maximum(m - 1, 0.0)), 0.0) * R2
        for j in range(MAXF):
            slab = slab + A[0, j + 1] * jnp.where(kjq[j] > 0, sqrt_k[j] * R1[predj[j]], 0.0 + 0j)
        slab = slab / jnp.sqrt(m.astype(jnp.float64))
        return (slab, R1), slab[det_flat]

    if L > 1:
        (_, _), outs = jax.lax.scan(sig_step, (R0, jnp.zeros_like(R0)),
                                    jnp.arange(1, L))
        psi = jnp.concatenate([R0[det_flat][None], outs])
    else:
        psi = R0[det_flat][None]
    return psi * pref / denom


def _vac_condition_one(V, mu, b, do, N=17):
    """Vacuum-project (herald |0>) control mode ``b`` if ``do``, keeping the fixed
    2N shape and decoupling b to vacuum.  Returns (V, mu, prob_factor).  A vacuum
    projection is Gaussian (Schur complement, measurement cov = I, hbar=2)."""
    bi = jnp.asarray([b, b + N])
    Vbb = V[jnp.ix_(bi, bi)]
    Mm = Vbb + jnp.eye(2)
    Minv = jnp.linalg.inv(Mm)
    Vob = V[:, bi]                                   # (2N, 2)
    Vn = V - Vob @ Minv @ Vob.T
    mun = mu - Vob @ Minv @ mu[bi]
    pf = 2.0 / jnp.sqrt(jnp.linalg.det(Mm)) * jnp.exp(-0.5 * mu[bi] @ Minv @ mu[bi])
    # decouple b -> vacuum
    Vn = Vn.at[bi, :].set(0.0).at[:, bi].set(0.0).at[b, b].set(1.0).at[b + N, b + N].set(1.0)
    mun = mun.at[bi].set(0.0)
    V2 = jnp.where(do, Vn, V)
    mu2 = jnp.where(do, mun, mu)
    pfac = jnp.where(do, pf, 1.0)
    return V2, mu2, pfac


def _herald_static(cov, mu, eff, ncontrol, Nmodes, MAXF, BF, L):
    """Generic single-compile herald: vacuum-condition eff==0 controls (masked
    Schur), then Hermite box over (signal + <=MAXF fired) in a flat buffer BF.
    cov/mu over Nmodes=1+ncontrol modes; eff (ncontrol,).  Returns (psi[L], prob)."""
    V = cov; m = mu
    p_vac = jnp.array(1.0)
    for b in range(1, ncontrol + 1):
        do = eff[b - 1] < 0.5
        V, m, pf = _vac_condition_one(V, m, b, do, N=Nmodes)
        p_vac = p_vac * pf
    fired = (eff >= 1).astype(jnp.int32)
    order = jnp.argsort(1 - fired)                        # fired controls first
    perm = jnp.concatenate([jnp.zeros((1,), jnp.int32), 1 + order.astype(jnp.int32)])
    keepc = perm[: 1 + MAXF]
    kk = jnp.concatenate([keepc, keepc + Nmodes])
    cov_k = V[jnp.ix_(kk, kk)]
    mu_k = m[kk]
    radii = eff[order][:MAXF].astype(jnp.int32) + 1
    psi_raw = _flat_amplitudes(cov_k, mu_k, radii, L, MAXF, BF)
    p_fock = jnp.sum(jnp.abs(psi_raw) ** 2)
    prob = p_vac * p_fock
    psi = jnp.where(p_fock > 0, psi_raw / jnp.sqrt(jnp.maximum(p_fock, 1e-300)), psi_raw)
    return psi, prob


def jax_reduced_herald_static(cov, mu, eff_pnr, L, BF=MOMENT_BF, depth: int = 3,
                              maxf: int = MOMENT_MAXF):
    """Single-compile heralded signal state + prob from the static
    equivalent-Gaussian outputs over M=1+2*nleaves modes (nleaves=2**depth):
    cov[2M,2M], mu[2M], eff_pnr[2*nleaves].  ``BF`` (fired-box buffer) and
    ``maxf`` (in-graph fired-mode cap) are perf/VRAM knobs -- both depth-
    independent, since the Hermite box is a (1+maxf)-mode object regardless of
    how many control slots the tree carries; ``L`` is the signal cutoff.
    depth=3 reproduces the original 16-control / 17-mode behaviour."""
    nleaves = 2 ** int(depth)
    ncontrol = 2 * nleaves
    Nmodes = 1 + 2 * nleaves
    return _herald_static(cov, mu, eff_pnr, ncontrol, Nmodes, int(maxf), int(BF), L)


_LEAF_MAXF = 2
_LEAF_BF = (15 + 1) ** 2          # 256: leaf has <=2 controls, each <=pnr_max


def _leaf_prob_product_static(params, L, depth: int = 3):
    """Product of exact per-leaf herald probs (fitness 'leaf' P), single-compile.
    A leaf with no real controls (n_ctrl=0 or inactive) contributes 1.  Each leaf
    is a fixed 3-mode object (1 signal + <=2 controls) independent of depth; only
    the number of leaves (=2**depth) scales."""
    nleaves = 2 ** int(depth)
    lp = params["leaf_params"]
    active = jnp.asarray(params["leaf_active"]).astype(jnp.float64)
    n_ctrl = jnp.asarray(lp.get("n_ctrl_ste", lp["n_ctrl"])).astype(jnp.float64)
    r = jnp.asarray(lp["r"]); ph = jnp.asarray(lp["phases"]); dv = jnp.asarray(lp["disp"])
    pnr = jnp.asarray(lp.get("pnr_ste", lp["pnr"]))
    P = jnp.array(1.0)
    for i in range(nleaves):
        n_real = jnp.where(active[i] > 0.5, n_ctrl[i] + 1.0, 0.0)
        mu_l, cov_l = _leaf_moments_masked(r[i], ph[i], dv[i], n_real)
        eff2 = jnp.stack([
            jnp.where((active[i] > 0.5) & (0 < n_ctrl[i]), pnr[i, 0].astype(jnp.float64), 0.0),
            jnp.where((active[i] > 0.5) & (1 < n_ctrl[i]), pnr[i, 1].astype(jnp.float64), 0.0),
        ])
        _, p = _herald_static(cov_l, mu_l, eff2, 2, 3, _LEAF_MAXF, _LEAF_BF, L)
        real_ctrl = jnp.where((active[i] > 0.5), jnp.minimum(n_ctrl[i], 2.0), 0.0)
        P = P * jnp.where(real_ctrl < 0.5, 1.0, p)
    return P


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
    for i in range(len(active)):
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
    for i in range(len(active)):
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
    vac = []
    for i, st in enumerate(structs):
        if sum(bool(x) for x in st[0]) == 0:
            vac.append(i)            # no active leaves (e.g. injected vacuum seed)
        elif _struct_fired_product(st) > REDUCED_HERALD_PROD_BUDGET:
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

    if vac:
        # All-vacuum genotype: single-mode vacuum signal, no photons. <O>=O[0,0],
        # P=1. Subject to the same physics artifact guard (a no-photon state below
        # the Gaussian limit is penalised, exactly as in the Fock path). grad = 0.
        O00 = float(np.real(np.asarray(operator_L)[0, 0]))
        artifact = O00 < gaussian_limit
        f_exp = np.inf if artifact else O00
        f_lp = np.inf if artifact else 0.0
        for i in vac:
            fit[i] = [-f_exp, -f_lp, 0.0, 0.0]
            desc[i] = [0.0, 0.0, 0.0]
            rawe[i] = O00; jprob[i] = 1.0

    extras = {
        "gradients": jnp.asarray(grads),
        "raw_expectation": jnp.asarray(rawe),
        "joint_probability": jnp.asarray(jprob),
        "leakage": jnp.zeros(Npop),
        "pnr_cost": jnp.asarray(desc[:, 2]),
        "final_state": jnp.zeros((Npop, base_cutoff), dtype=jnp.complex128),
    }
    return jnp.asarray(fit), jnp.asarray(desc), extras


def _effective_photons_static(cov_k, eff_pnr, eps, depth: int = 3):
    """Effective (coupled) photons from the final M=1+2*nleaves-mode equivalent
    cov (nleaves=2**depth): each fired control's detected count weighted by a
    smooth gate on its coupling to (signal + other controls).  Differentiable;
    fixed shape per depth.  depth=3 reproduces the 17-mode version."""
    N = 1 + 2 * (2 ** int(depth))
    n_eff = jnp.array(0.0); max_eff = jnp.array(0.0)
    for c in range(1, N):
        others = [0] + [c2 for c2 in range(1, N) if c2 != c]
        oi = jnp.asarray([i for o in others for i in (o, o + N)])
        rows = cov_k[jnp.asarray([c, c + N])][:, oi]
        cpl = jnp.sqrt(jnp.sum(rows ** 2))
        w = cpl ** 2 / (cpl ** 2 + eps ** 2)
        ne = jnp.where(eff_pnr[c - 1] >= 1, eff_pnr[c - 1] * w, 0.0)
        n_eff = n_eff + ne; max_eff = jnp.maximum(max_eff, ne)
    return n_eff, max_eff


def _nongaussianity_delta(psi):
    """Relative-entropy non-Gaussianity delta = g(nu) of a single-mode PURE state
    ``psi`` (Fock amplitudes).  nu = sqrt(det(cov)) is the symplectic eigenvalue of
    the state's 2x2 covariance (hbar=2 => vacuum cov = I, nu = 1).  A pure state is
    Gaussian iff nu = 1 (delta = 0); non-Gaussian => nu > 1, delta = g(nu) > 0
    (= S(rho_Gaussian-reference), since S(pure)=0).  Cheap (2nd moments only) and
    DIFFERENTIABLE, so it can drive the gradient emitter.  This is a resource
    proxy for exploration ONLY -- it never enters the stored fitness/<O>.

      g(x) = ((x+1)/2) log2((x+1)/2) - ((x-1)/2) log2((x-1)/2)
    """
    L = psi.shape[0]
    cdt = psi.dtype
    p = jnp.abs(psi) ** 2
    nbar = jnp.sum(jnp.arange(L).astype(jnp.float64) * p)              # <a†a>
    sq1 = jnp.sqrt(jnp.arange(1, L).astype(jnp.float64)).astype(cdt)
    mean_a = jnp.sum(jnp.conj(psi[:-1]) * psi[1:] * sq1)               # <a>
    sq2 = jnp.sqrt((jnp.arange(1, L - 1) * jnp.arange(2, L)).astype(jnp.float64)).astype(cdt)
    mean_a2 = jnp.sum(jnp.conj(psi[:-2]) * psi[2:] * sq2)              # <a^2>
    re_a, im_a = jnp.real(mean_a), jnp.imag(mean_a)
    re_a2, im_a2 = jnp.real(mean_a2), jnp.imag(mean_a2)
    # x = a + a†, p = -i(a - a†); hbar=2 (vacuum <x^2>=<p^2>=1)
    mean_x, mean_p = 2.0 * re_a, 2.0 * im_a
    xx = 2.0 * re_a2 + 2.0 * nbar + 1.0
    pp = -2.0 * re_a2 + 2.0 * nbar + 1.0
    xp = 2.0 * im_a2                                                   # (1/2)<{x,p}>
    sxx = xx - mean_x ** 2
    spp = pp - mean_p ** 2
    sxp = xp - mean_x * mean_p
    det = sxx * spp - sxp ** 2
    nu = jnp.sqrt(jnp.maximum(det, 1.0))          # uncertainty floor det>=1 (hbar=2)
    a = (nu + 1.0) / 2.0
    b = (nu - 1.0) / 2.0
    return a * jnp.log2(a) - jnp.where(b > 1e-12, b * jnp.log2(jnp.maximum(b, 1e-300)), 0.0)


@_partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _score_pop_static_jit(genotypes, operator_L, genotype_name, config_hashable,
                          base_cutoff, L, floats):
    """ONE-compile vmap'd moment scorer over a population (no per-structure
    recompile).  Returns (fitnesses[N,4], descriptors[N,3], gradients[N,D],
    raw_exp[N], joint_prob[N])."""
    gs_eig, gaussian_limit, w_exp, w_prob, coupling_eps, w_ng = floats
    cfg = dict(config_hashable)
    depth = int(cfg.get("depth", 3))
    ng_desc = bool(cfg.get("moment_ng_descriptor", False))  # non-Gaussianity as MAP-Elites axis
    maxf = int(cfg.get("moment_maxf", MOMENT_MAXF))     # in-graph fired-mode cap
    # gradient checkpointing (rematerialisation): recompute the per-genotype
    # forward during backward instead of storing it.  ~2x compute, big peak-VRAM
    # cut on the deep trees.  Default on for depth>=4 (off at depth 3 to keep the
    # validated fast path identical).
    remat = bool(cfg.get("moment_remat", depth >= 4))
    BF = int(cfg.get("moment_bf", MOMENT_BF))           # fired-box buffer (perf knob)
    # IN-LOOP truncation gate: a state with too much mass near the TOP of the
    # L-box has an unreliable renormalised <O> (the herald normalises within L,
    # so mass beyond L is invisible), and gradient descent / the GA can exploit
    # that truncation -- the moment-scorer analogue of the Fock artifact.  The
    # honest criterion is the periodic dual-L sweep; this cheap gate kills the
    # gross exploits AT INSERTION TIME so they never become elites: reject any
    # state whose top-decile Fock bins carry more than ``moment_tail_tol`` of
    # its (normalised) mass.  NOTE: the default must stay loose enough for
    # legitimately photon-rich states (r~1.9 squeezing has percent-level tails
    # at L=50); the dual-L sweep remains the exact judge.
    tail_tol = float(cfg.get("moment_tail_tol", 0.05))
    # 'fast' search mode: skip the exact per-leaf probability (8 sub-heralds) when
    # probability isn't being optimised (w_prob~0) -> big speedup; periodic exact
    # re-validation recovers the true probability.  Default on when w_prob==0.
    skip_leaf = bool(cfg.get("moment_fast", w_prob < 1e-9))
    decoder = get_genotype_decoder(genotype_name, depth=int(cfg.get("depth", 3)),
                                   config=cfg)

    def loss(g):
        params = decoder.decode(g, base_cutoff)
        cov_k, mu_k, eff_pnr, _dens = jax_equivalent_gaussian_static(params, depth)
        psi, _prob_pnr = jax_reduced_herald_static(cov_k, mu_k, eff_pnr, L, BF,
                                                   depth, maxf)
        Lp = psi.shape[0]
        raw_exp = jnp.real(jnp.vdot(psi, operator_L[:Lp, :Lp] @ psi))
        # the herald is valid only if it produced a normalised state; a zero-norm
        # psi (impossible PNR pattern, or numerical underflow) must be INVALID,
        # not scored as <O>=0 (which would masquerade as the global best).
        herald_norm = jnp.sum(jnp.abs(psi) ** 2)
        P_leaf = jnp.array(1.0) if skip_leaf else _leaf_prob_product_static(params, L, depth)
        joint_prob = jnp.real(P_leaf)
        # over-budget detection IN-GRAPH (no eager pre-pass): kf>maxf (fired modes
        # dropped) or prod(n_j+1)>BF (flat buffer overflow) => not representable
        # here -> mark invalid (the rare extreme-Sigma-n tail).
        fired_mask = eff_pnr >= 1
        kf = jnp.sum(fired_mask.astype(jnp.float64))
        logprod = jnp.sum(jnp.where(fired_mask, jnp.log(eff_pnr + 1.0), 0.0))
        in_budget = (kf <= maxf) & (logprod <= jnp.log(float(BF)) + 1e-6)
        # truncation gate (see tail_tol above): psi is normalised within L, so
        # heavy mass in the top decile of the box flags an L-truncation exploit.
        n_tail = max(1, Lp // 10)
        tail_mass = jnp.sum(jnp.abs(psi[Lp - n_tail:]) ** 2)
        tail_ok = tail_mass < tail_tol
        valid = (joint_prob > 1e-40) & (herald_norm > 0.5) & in_budget & tail_ok
        n_eff, max_eff = _effective_photons_static(cov_k, eff_pnr, coupling_eps, depth)
        active_modes = jnp.sum(jnp.asarray(params["leaf_active"]).astype(jnp.float64))
        exp_val = jnp.where(valid, raw_exp, 0.0)            # grad-safe placeholder
        prob_capped = jnp.minimum(jnp.maximum(joint_prob, 1e-45), 1.0)
        log_prob = -jnp.log10(prob_capped)
        log_prob = log_prob + jnp.where(jnp.maximum(joint_prob - 1.0, 0.0) > 1e-4, jnp.inf, 0.0)
        is_artifact = jnp.logical_and(exp_val < gaussian_limit, n_eff < 0.5)
        art = jnp.where(is_artifact, jnp.inf, 0.0)
        invalid_pen = jnp.where(valid, 0.0, jnp.inf)        # mark invalid herald
        exp_val = exp_val + art; log_prob = log_prob + art + invalid_pen
        eps0 = 0.01
        d_e = w_exp * jnp.abs(exp_val - (gs_eig - eps0))
        d_p = w_prob * jnp.abs(log_prob - (-eps0))
        loss_val = jnp.maximum(d_e, d_p) + 0.01 * (d_e + d_p)
        # ANNEALED non-Gaussianity exploration reward: subtract w_ng*delta from the
        # OPTIMIZATION objective only, so the gradient is pushed toward Wigner-
        # negative (non-Gaussian) states and out of the Gaussian basin. w_ng is
        # annealed to 0 by the pipeline. CRITICAL: this touches ONLY loss_opt (the
        # gradient); final_exp/raw_exp/fit[:,0] remain the TRUE <O>, so the archive
        # and every reported value are unaffected by the reward.
        delta_ng = jnp.where(valid, _nongaussianity_delta(psi), 0.0)
        loss_opt = loss_val - w_ng * delta_ng
        final_exp = jnp.where(valid, raw_exp, jnp.inf) + art
        aux = dict(final_exp=final_exp, log_prob=log_prob, active=active_modes,
                   max_pnr=max_eff, photons=n_eff, raw_exp=raw_exp, joint_prob=joint_prob,
                   delta_ng=delta_ng)
        return jnp.real(loss_opt), aux

    loss_fn = jax.checkpoint(loss) if remat else loss
    (_lv, aux), grad = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True))(genotypes)
    # 2-OBJECTIVE Pareto dominance: (exp, prob) ONLY.  The former objectives 3/4
    # (-active, -photons) made every proto-non-Gaussian candidate Pareto-DOMINATED
    # by the Gaussian corner (fewer photons/leaves always won inside a cell), so
    # stepping stones toward breeding structure were deleted at insertion.
    # Complexity/photons remain DESCRIPTOR axes (diversity), not objectives.
    fit = jnp.stack([-aux["final_exp"], -aux["log_prob"]], axis=1)
    desc_axes = [aux["active"], aux["max_pnr"], aux["photons"]]
    if ng_desc:                                   # optional 4th MAP-Elites axis
        desc_axes.append(aux["delta_ng"])
    desc = jnp.stack(desc_axes, axis=1)
    return fit, desc, grad, aux["raw_exp"], aux["joint_prob"], aux["delta_ng"]


def _default_moment_chunk(depth: int, npop: int) -> int:
    """Population-vmap shard size that keeps peak VRAM bounded as the tree grows.
    The heavy Hermite box is depth-independent, but the equivalent-Gaussian and
    vac-conditioning tapes grow ~(2*nleaves)^2 per genotype, so we shrink the
    simultaneous-vmap width with depth.  Override with cfg['moment_chunk']."""
    if depth <= 3:
        return npop                # validated full-pop path unchanged
    if depth == 4:
        return min(npop, 16)
    if depth == 5:
        return min(npop, 4)
    return min(npop, 2)            # depth>=6: best-effort


def moment_score_population_static(genotypes, operator_L, genotype_name,
                                   config_hashable, base_cutoff, L,
                                   gs_eig, gaussian_limit):
    """Option-2 scorer: ONE XLA compile, vmap over the population in fixed-size
    chunks (depth-derived default, or cfg['moment_chunk']).  Chunking bounds peak
    VRAM as the breeding tree deepens; every chunk shares one compiled graph (the
    final partial chunk is padded to the chunk width, then truncated).
    Over-budget genotypes (kf>maxf or prod(n_j+1)>BF -- the rare extreme-Sigma-n
    tail) are detected in-graph and marked invalid (dropped from the archive)."""
    cfg = dict(config_hashable)
    w_exp = float(cfg.get("alpha_expectation", 1.0))
    w_prob = float(cfg.get("alpha_probability", 0.0))
    coupling_eps = float(cfg.get("coupling_eps", 0.05))
    w_ng = float(cfg.get("alpha_nongauss", 0.0))    # annealed non-Gaussianity reward
    floats = (float(gs_eig), float(gaussian_limit), w_exp, w_prob, coupling_eps, w_ng)
    g_all = jnp.asarray(genotypes)
    Npop = int(g_all.shape[0])

    depth = int(cfg.get("depth", 3))
    chunk = int(cfg.get("moment_chunk", 0)) or _default_moment_chunk(depth, Npop)
    chunk = max(1, min(int(chunk), Npop))

    def _run(gs):
        return _score_pop_static_jit(gs, operator_L, genotype_name,
                                     config_hashable, int(base_cutoff), int(L),
                                     floats)

    if chunk >= Npop:
        f, d, gr, re, jp, dng = _run(g_all)
    else:
        n_chunks = (Npop + chunk - 1) // chunk
        pad = n_chunks * chunk - Npop
        gp = g_all if pad == 0 else jnp.concatenate(
            [g_all, jnp.broadcast_to(g_all[-1:], (pad,) + g_all.shape[1:])], axis=0)
        fs, ds, grs, res, jps, dngs = [], [], [], [], [], []
        for ci in range(n_chunks):
            sl = jax.lax.stop_gradient(gp[ci * chunk:(ci + 1) * chunk])
            cf, cd, cgr, cre, cjp, cdng = _run(sl)
            fs.append(cf); ds.append(cd); grs.append(cgr); res.append(cre)
            jps.append(cjp); dngs.append(cdng)
        f = jnp.concatenate(fs, 0)[:Npop]
        d = jnp.concatenate(ds, 0)[:Npop]
        gr = jnp.concatenate(grs, 0)[:Npop]
        re = jnp.concatenate(res, 0)[:Npop]
        jp = jnp.concatenate(jps, 0)[:Npop]
        dng = jnp.concatenate(dngs, 0)[:Npop]

    extras = {
        "gradients": gr,
        "raw_expectation": re,
        "joint_probability": jp,
        "nongaussianity": dng,
        "leakage": jnp.zeros(Npop),
        "pnr_cost": d[:, 2],
        "final_state": jnp.zeros((Npop, base_cutoff), dtype=jnp.complex128),
    }
    return f, d, extras


@_partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _revalidate_jit(genotypes, op_lo, op_hi, genotype_name, config_hashable,
                    base_cutoff, L_lo, L_hi):
    """For each genotype: (<O>_searchL, <O>_highL, herald_norm_searchL).  One
    compile per (L_lo, L_hi)."""
    cfg = dict(config_hashable)
    depth = int(cfg.get("depth", 3))
    maxf = int(cfg.get("moment_maxf", MOMENT_MAXF))
    BF_lo = int(cfg.get("moment_bf", MOMENT_BF))
    BF_hi = int(cfg.get("moment_bf_high", 8192))
    decoder = get_genotype_decoder(genotype_name, depth=depth, config=cfg)

    def one(g):
        p = decoder.decode(g, base_cutoff)
        cs, ms, ep, _ = jax_equivalent_gaussian_static(p, depth)
        plo, _ = jax_reduced_herald_static(cs, ms, ep, L_lo, BF_lo, depth, maxf)
        phi, _ = jax_reduced_herald_static(cs, ms, ep, L_hi, BF_hi, depth, maxf)
        elo = jnp.real(jnp.vdot(plo, op_lo[:plo.shape[0], :plo.shape[0]] @ plo))
        ehi = jnp.real(jnp.vdot(phi, op_hi[:phi.shape[0], :phi.shape[0]] @ phi))
        # exact 'leaf' herald probability (the search may have used --moment-fast,
        # which stores a prob=1 PLACEHOLDER -- recompute it so the archive's
        # probability objective is real). obj1 convention = log10(P) (<= 0).
        P = jnp.real(_leaf_prob_product_static(p, L_hi, depth))
        log10P = jnp.log10(jnp.clip(P, 1e-45, 1.0))
        # tail mass of the HIGH-L state's top decile: plo/phi are normalised, so
        # sum|plo|^2 == 1 always (the old "norm" check was a no-op); the honest
        # truncation diagnostic at L_hi is heavy mass near the top of the box.
        n_tail = max(1, int(L_hi) // 10)
        tail_hi = jnp.sum(jnp.abs(phi[int(L_hi) - n_tail:]) ** 2)
        return elo, ehi, tail_hi, log10P

    return jax.vmap(one)(genotypes)


def clean_archive_moment(repertoire, genotype_name, config_hashable, base_cutoff,
                         L_lo, L_hi, tol=0.02, tail_tol=0.02, chunk=None,
                         prev_fp=None):
    """Periodic dual-L sweep: re-score the archive at high L, refresh fitness[0]
    to the exact high-L <O>, and DROP any cell whose search-L <O> disagrees by
    >tol (or whose HIGH-L state still has heavy top-decile tail mass) -- i.e. an
    L-truncation artifact.  Returns (cleaned_repertoire, num_removed, fp).

    INCREMENTAL MODE: pass ``prev_fp`` (the fingerprint array returned by the
    previous sweep) and only cells whose fitness[0] changed since then (new or
    replaced solutions) are re-scored; unchanged cells keep their validated
    values.  This makes an every-250-generation cadence affordable at depth 5,
    where a full-archive sweep costs tens of minutes.

    NB: this re-score runs at L_hi AND moment_bf_high (default 8192) -- a MUCH
    bigger Hermite box than the search -- so it is the heaviest VRAM moment of a
    deep run.  The validation chunk therefore defaults SMALLER than the search
    chunk and shrinks with depth; override with cfg['moment_validate_chunk']."""
    cfg = dict(config_hashable)
    if chunk is None:
        _depth = int(cfg.get("depth", 3))
        chunk = int(cfg.get("moment_validate_chunk", 0)) or (
            32 if _depth <= 3 else 8 if _depth == 4 else 2)
    a, b = cfg.get("target_alpha"), cfg.get("target_beta")
    op_lo = moment_operator(int(L_lo), a, b)
    op_hi = moment_operator(int(L_hi), a, b)
    # np.asarray() on a JAX array (QDax MOME repertoire.fitnesses) yields a
    # READ-ONLY view -> the in-place refresh below raises "assignment destination
    # is read-only". np.array(..., copy) forces a writable host copy.
    fit = np.array(repertoire.fitnesses, dtype=float)
    shp = fit.shape
    fitf = fit.reshape(-1, shp[-1])
    gen = np.asarray(repertoire.genotypes).reshape(fitf.shape[0], -1)
    valid_mask = np.isfinite(fitf[:, 0]) & (fitf[:, 0] > -1e9)
    if prev_fp is not None and prev_fp.shape[0] == fitf.shape[0]:
        # only re-validate cells whose stored fitness changed since last sweep
        changed = valid_mask & (fitf[:, 0] != prev_fp)
    else:
        changed = valid_mask
    valid = np.where(changed)[0]
    n_removed = 0
    has_prob = fitf.shape[1] > 1                   # (exp, prob[, legacy extras])
    for s in range(0, len(valid), chunk):
        idx = valid[s:s + chunk]
        elo, ehi, tail_hi, lp = _revalidate_jit(
            jnp.asarray(gen[idx].astype(np.float64)), op_lo, op_hi,
            genotype_name, config_hashable, int(base_cutoff), int(L_lo), int(L_hi))
        elo = np.asarray(elo); ehi = np.asarray(ehi)
        tail_hi = np.asarray(tail_hi)
        art = (np.abs(ehi - elo) > tol) | (tail_hi > tail_tol)
        fitf[idx, 0] = -ehi                       # refresh to exact high-L <O>
        if has_prob:
            fitf[idx, 1] = np.asarray(lp)         # refresh exact probability (obj1=log10 P)
        fitf[idx[art], 0] = -np.inf               # drop the artifacts
        n_removed += int(art.sum())
    fp = fitf[:, 0].copy()                        # post-refresh fingerprint
    new = repertoire.replace(fitnesses=jnp.asarray(fitf.reshape(shp)))
    return new, n_removed, fp


def _numpy_leaf_prob_product(params, L):
    """Exact product of per-leaf herald probs (numpy), the 'leaf' fitness P."""
    from frontend.independent_verifier import _build_gaussian_moments
    from frontend.gbs_optimizer import reduced_herald
    lp = params["leaf_params"]
    P = 1.0
    for i in range(int(np.asarray(params["leaf_active"]).shape[0])):
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
