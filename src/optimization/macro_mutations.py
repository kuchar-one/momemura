"""Physics-motivated macro-mutations for the B30 family.

Polynomial mutation perturbs one gene at a time, but the breeding-protocol
solution manifold is defined by COORDINATED structure: matched leaf states,
balanced beamsplitters, x=0 homodyne, specific PNR patterns.  A single-gene
step off the Gaussian optimum is almost always fitness-negative, so the GA
never assembles the structure by accident.  These operators jump straight to
physically meaningful configurations (photon subtraction, cat breeding steps,
subtree symmetrization, ...), turning known CV state-engineering moves into
mutation moves.

All operators are pure jnp index operations on the RAW genotype (static layout
per (genotype, depth, config)), so the whole registry compiles into the
emitter's XLA graph via lax.switch.
"""

from typing import Dict, Any, List

import numpy as np
import jax
import jax.numpy as jnp

from src.genotypes.pnr_seeds import b30_layout, _BALANCED_THETA_RAW
from src.genotypes.converter import _layer_node_offsets

# raw value giving near-maximal squeezing under tanh scaling
_MAX_SQUEEZE_RAW = float(np.arctanh(0.95))


def make_macro_mutation(genotype_name: str, depth: int, config: Dict[str, Any]):
    """Build ``macro_mutation(x[B, D], key) -> x'[B, D]``: each individual gets
    ONE randomly chosen physics operator applied.  Compatible with the QDax
    mutation_fn call convention used across this repo (single return value)."""
    lay = b30_layout(genotype_name, depth, config)
    forced = genotype_name == "B30F"
    L, nodes = lay["L"], lay["nodes"]
    U, PN = lay["U"], lay["PN"]
    pm, nc, pl = lay["pnr_max"], lay["n_control"], lay["pnr_len"]
    u0 = lay["unique"][0]
    hom0, hom1 = lay["hom"]
    mix0, mix1 = lay["mix"]
    fin0, fin1 = lay["final"]
    r0, r1 = lay["shared_r"]
    p0, p1 = lay["shared_phases"]
    d0, d1 = lay["shared_disp"]

    # static index arrays into the unique block
    idx_active = np.asarray([u0 + i * U + 0 for i in range(L)])
    idx_nctrl = np.asarray([u0 + i * U + 1 for i in range(L)])
    idx_pnr0 = np.asarray([u0 + i * U + 2 for i in range(L)])
    idx_pnr_rest = np.asarray(
        [u0 + i * U + 2 + c for i in range(L) for c in range(1, pl)]
    ) if pl > 1 else np.zeros(0, dtype=int)
    idx_unique_rows = np.asarray([[u0 + i * U + c for c in range(U)]
                                  for i in range(L)])          # (L, U)
    idx_mix_rows = np.asarray([[mix0 + n * PN + c for c in range(PN)]
                               for n in range(nodes)])          # (nodes, PN)
    layer_offs = _layer_node_offsets(depth)                     # python ints

    # raw-space encodings (bin centres) for the discrete genes
    if forced:
        enc_pnr0_one = 0.5 if pm <= 1 else 0.0        # k=1
        pnr0_step = 1.0 if pm <= 1 else 1.0 / (pm - 1)
        enc_nctrl_one = 0.0 if nc <= 1 else -1.0      # n=1
    else:
        enc_pnr0_one = 1.0 / pm
        pnr0_step = 1.0 / pm
        enc_nctrl_one = 2.0 / nc - 1.0

    balanced = _BALANCED_THETA_RAW

    def _rand_leaf(key):
        return jax.random.randint(key, (), 0, L)

    # ----------------------------------------------------------------- ops --
    def op_subtract_photon_leaf(g, key):
        """Turn one leaf into a canonical 1-photon-subtracted squeezed leaf:
        active, one control, single click, second detector off."""
        i = _rand_leaf(key)
        base = u0 + i * U
        g = g.at[base + 0].set(1.0)
        g = g.at[base + 1].set(enc_nctrl_one)
        g = g.at[base + 2].set(enc_pnr0_one)
        for c in range(1, pl):
            g = g.at[base + 2 + c].set(0.0)
        return g

    def op_add_click(g, key):
        """One more photon on a random leaf's first detector."""
        i = _rand_leaf(key)
        j = u0 + i * U + 2
        return g.at[j].set(jnp.clip(g[j] + pnr0_step, 0.0, 1.0))

    def op_remove_click(g, key):
        """One photon fewer on a random leaf's first detector."""
        i = _rand_leaf(key)
        j = u0 + i * U + 2
        return g.at[j].set(jnp.clip(g[j] - pnr0_step, 0.0, 1.0))

    def op_toggle_leaf(g, key):
        """Flip a random leaf's active flag (grow/prune the tree)."""
        i = _rand_leaf(key)
        j = u0 + i * U
        return g.at[j].set(jnp.where(jnp.abs(g[j]) < 1e-6, -1.0, -g[j]))

    def op_deactivate_subtree(g, key):
        """Prune a random aligned subtree (all its leaves inactive)."""
        k1, k2 = jax.random.split(key)
        lvl = jax.random.randint(k1, (), 1, max(depth, 2))       # subtree height
        size = jnp.left_shift(1, lvl)
        blk = jax.random.randint(k2, (), 0, L) // size
        pos = jnp.arange(L)
        mask = (pos >= blk * size) & (pos < (blk + 1) * size)
        vals = jnp.where(mask, -1.0, g[idx_active])
        return g.at[idx_active].set(vals)

    def op_activate_all(g, key):
        """All leaves active (full breeding tree)."""
        return g.at[idx_active].set(jnp.abs(g[idx_active]) + 1e-3)

    def op_symmetrize_sibling(g, key):
        """Copy a random leaf's discrete block onto its sibling (breeding wants
        pairwise-identical inputs at each merge node)."""
        i = _rand_leaf(key)
        sib = i ^ 1
        rows = jnp.asarray(idx_unique_rows)
        return g.at[rows[sib]].set(g[rows[i]])

    def op_symmetrize_all(g, key):
        """Broadcast one random leaf's discrete block to ALL leaves (fully
        symmetric breeding: every leaf prepares the same resource state)."""
        i = _rand_leaf(key)
        rows = jnp.asarray(idx_unique_rows)
        return g.at[rows].set(jnp.broadcast_to(g[rows[i]], (L, U)))

    def op_mirror_half(g, key):
        """Copy the LEFT half-tree onto the RIGHT: leaves, per-layer homodyne
        and mixing parameters.  One breeding stage = interfering two copies of
        the same sub-protocol."""
        rows = jnp.asarray(idx_unique_rows)
        g = g.at[rows[L // 2:]].set(g[rows[:L // 2]])
        mrows = jnp.asarray(idx_mix_rows)
        for k in range(1, depth):                 # layers below the root
            off = layer_offs[k - 1]
            cnt = 2 ** (depth - k)
            half = cnt // 2
            g = g.at[hom0 + off + half:hom0 + off + cnt].set(
                g[hom0 + off:hom0 + off + half])
            g = g.at[mrows[off + half:off + cnt]].set(
                g[mrows[off:off + half]])
        return g

    def op_breed_level(g, key):
        """Set one random tree layer to the canonical breeding merge:
        balanced BS (theta=pi/4, phi=varphi=0) + x=0 homodyne."""
        lvl = jax.random.randint(key, (), 1, depth + 1)
        node_pos = jnp.arange(nodes)
        offs = jnp.asarray(layer_offs + [nodes])
        lo = offs[lvl - 1]
        hi = offs[lvl]
        mask = (node_pos >= lo) & (node_pos < hi)
        g = g.at[jnp.asarray(idx_mix_rows)[:, 0]].set(
            jnp.where(mask, balanced, g[jnp.asarray(idx_mix_rows)[:, 0]]))
        for c in range(1, PN):
            col = jnp.asarray(idx_mix_rows)[:, c]
            g = g.at[col].set(jnp.where(mask, 0.0, g[col]))
        hom_idx = jnp.arange(hom0, hom1)
        g = g.at[hom_idx].set(jnp.where(mask, 0.0, g[hom_idx]))
        return g

    def op_breed_all(g, key):
        """Canonical breeding everywhere: every node balanced + x=0 homodyne."""
        mrows = jnp.asarray(idx_mix_rows)
        g = g.at[mrows[:, 0]].set(balanced)
        for c in range(1, PN):
            g = g.at[mrows[:, c]].set(0.0)
        return g.at[jnp.arange(hom0, hom1)].set(0.0)

    def op_zero_disp(g, key):
        """Remove displacement (breeding resources are centred states)."""
        return g.at[jnp.arange(d0, d1)].set(0.0)

    def op_boost_squeeze(g, key):
        """Crank the signal-mode squeezing toward the clamp (cat/GKP quality
        is squeezing-limited)."""
        sgn = jnp.where(g[r0] >= 0, 1.0, -1.0)
        return g.at[r0].set(sgn * _MAX_SQUEEZE_RAW)

    def op_hom_zero(g, key):
        """All homodyne conditions to x=0 (the breeding post-selection)."""
        return g.at[jnp.arange(hom0, hom1)].set(0.0)

    def op_final_identity(g, key):
        """Strip the final Gaussian (look at the bred state undressed)."""
        return g.at[jnp.arange(fin0, fin1)].set(0.0)

    def op_phase_shuffle(g, key):
        """Jitter the shared leaf-interferometer phases (re-aim the
        signal-control coupling without touching structure)."""
        noise = jax.random.normal(key, (p1 - p0,)) * 0.4
        return g.at[jnp.arange(p0, p1)].add(noise)

    def op_toggle_second_detector(g, key):
        """Flip a random leaf between 1 and 2 active control detectors."""
        i = _rand_leaf(key)
        j = u0 + i * U + 1
        return g.at[j].set(jnp.where(jnp.abs(g[j]) < 1e-6, 1.0, -g[j]))

    def op_pnr_uniformize(g, key):
        """Tie every leaf's first-detector click count to a random leaf's."""
        i = _rand_leaf(key)
        return g.at[jnp.asarray(idx_pnr0)].set(g[u0 + i * U + 2])

    ops: List = [
        op_subtract_photon_leaf,
        op_add_click,
        op_remove_click,
        op_toggle_leaf,
        op_deactivate_subtree,
        op_activate_all,
        op_symmetrize_sibling,
        op_symmetrize_all,
        op_mirror_half,
        op_breed_level,
        op_breed_all,
        op_zero_disp,
        op_boost_squeeze,
        op_hom_zero,
        op_final_identity,
        op_phase_shuffle,
        op_toggle_second_detector,
        op_pnr_uniformize,
    ]

    def _mutate_one(g, key):
        k_op, k_use = jax.random.split(key)
        op_i = jax.random.randint(k_op, (), 0, len(ops))
        return jax.lax.switch(op_i, ops, g, k_use)

    def macro_mutation(x, key):
        keys = jax.random.split(key, x.shape[0])
        return jax.vmap(_mutate_one)(x, keys)

    macro_mutation.n_ops = len(ops)
    return macro_mutation


def make_mixed_mutation(poly_fn, macro_fn, macro_prob: float):
    """Compose polynomial mutation with macro-mutations: each offspring takes a
    macro operator with probability ``macro_prob``, else the polynomial move.
    Handles both single-return and (value, key) QDax mutation conventions."""
    def mixed(x, key):
        k_sel, k_poly, k_macro = jax.random.split(key, 3)
        res = poly_fn(x, k_poly)
        x_poly = res[0] if isinstance(res, tuple) else res
        x_macro = macro_fn(x, k_macro)
        take = jax.random.bernoulli(k_sel, macro_prob, (x.shape[0], 1))
        return jnp.where(take, x_macro, x_poly)
    return mixed
