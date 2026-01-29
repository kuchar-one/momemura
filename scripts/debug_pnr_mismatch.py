#!/usr/bin/env python3
"""
Diagnostic script for investigating PNR decoding mismatch.

This script traces the exact values through:
1. Raw genotype values
2. Decoded PNR values from genotype decoder
3. Parameters used in herald simulation
4. Final expectation value

Run with: python scripts/debug_pnr_mismatch.py <experiment_path> <solution_idx>
Example: python scripts/debug_pnr_mismatch.py output/experiments/00B_c30_a1p00_b1p41 3711
"""

import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import jax.numpy as jnp

from src.utils.result_manager import AggregatedOptimizationResult
from src.genotypes.genotypes import get_genotype_decoder


def debug_solution(exp_path: str, idx: int):
    """Debug a specific solution to identify PNR decoding issues."""

    print("\n" + "=" * 80)
    print(f"DEBUGGING SOLUTION #{idx}")
    print(f"Experiment: {exp_path}")
    print("=" * 80)

    # 1. Load the experiment
    print("\n[1] Loading experiment...")
    agg = AggregatedOptimizationResult.load_group(exp_path)
    config = agg.config

    genotype_name = config.get("genotype", "A")
    pnr_max = int(config.get("pnr_max", 3))
    depth = int(config.get("depth", 3))
    cutoff = int(config.get("cutoff", 30))

    print(f"    Genotype: {genotype_name}")
    print(f"    Config pnr_max: {pnr_max}")
    print(f"    Depth: {depth}")
    print(f"    Cutoff: {cutoff}")

    # 2. Extract raw genotype
    print("\n[2] Extracting genotype...")
    # Must call get_pareto_front() first to populate the cache
    df = agg.get_pareto_front()
    print(f"    Loaded {len(df)} solutions from {len(agg.runs)} runs")

    if idx >= len(df):
        print(f"    ERROR: Index {idx} out of range (max {len(df) - 1})")
        return None, None

    genotype = agg._cached_valid_genotypes[idx]
    print(f"    Genotype shape: {genotype.shape}")
    print(f"    Genotype dtype: {genotype.dtype}")

    # 3. Get the decoder
    print("\n[3] Initializing decoder...")
    decoder = get_genotype_decoder(genotype_name, depth=depth, config=config)
    print(f"    Decoder class: {decoder.__class__.__name__}")
    print(f"    P_leaf_full: {getattr(decoder, 'P_leaf_full', 'N/A')}")
    print(f"    n_modes: {decoder.n_modes}")
    print(f"    n_control: {decoder.n_control}")
    print(f"    pnr_len: {decoder.pnr_len}")
    print(f"    gg_len: {decoder.gg_len}")

    # 4. Manual extraction of raw PNR values
    print("\n[4] Manual extraction of RAW genotype values for PNR slots...")

    # For Design 00B (Design 0 + Balanced), layout is:
    # [0 : nodes] = Per-node homodyne_x
    # Then follows DesignA structure with P_leaf_full blocks

    if hasattr(decoder, "P_leaf_full"):
        P_leaf = decoder.P_leaf_full
        nodes = decoder.nodes
        leaves = decoder.leaves
        pnr_len = decoder.pnr_len

        print(
            f"\n    Layout: hom_x({nodes}) + leaves({leaves} x {P_leaf}) + mix + final"
        )
        print(
            f"    Per-leaf structure: [Active(1), NCtrl(1), PNR({pnr_len}), GG_params]"
        )

        # Index where leaf blocks start
        leaf_block_start = nodes  # After homodyne_x vector

        print("\n    RAW genotype values for each leaf:")
        print("    " + "-" * 70)

        for leaf_i in range(leaves):
            leaf_start = leaf_block_start + leaf_i * P_leaf

            # Extract raw values
            active_raw = float(genotype[leaf_start])
            n_ctrl_raw = float(genotype[leaf_start + 1])
            pnr_raw_0 = float(genotype[leaf_start + 2])
            pnr_raw_1 = float(genotype[leaf_start + 3]) if pnr_len >= 2 else 0.0

            # Manual decode
            is_active = active_raw > 0.0
            # n_ctrl decode: round((raw+1)/2 * n_control), clip to [0, n_control]
            n_ctrl_decoded = int(round((n_ctrl_raw + 1.0) / 2.0 * decoder.n_control))
            n_ctrl_decoded = max(0, min(n_ctrl_decoded, decoder.n_control))

            # pnr decode: round(clip(raw, 0, 1) * pnr_max)
            pnr_0_clipped = max(0.0, min(1.0, pnr_raw_0))
            pnr_1_clipped = max(0.0, min(1.0, pnr_raw_1))
            pnr_0_decoded = int(round(pnr_0_clipped * pnr_max))
            pnr_1_decoded = int(round(pnr_1_clipped * pnr_max))

            print(f"\n    Leaf {leaf_i}:")
            print(f"        Index range: [{leaf_start}:{leaf_start + P_leaf}]")
            print(f"        active_raw = {active_raw:.6f} -> Active: {is_active}")
            print(f"        n_ctrl_raw = {n_ctrl_raw:.6f} -> n_ctrl: {n_ctrl_decoded}")
            print(
                f"        pnr_raw[0] = {pnr_raw_0:.6f} -> clip={pnr_0_clipped:.4f} -> PNR[0]: {pnr_0_decoded}"
            )
            print(
                f"        pnr_raw[1] = {pnr_raw_1:.6f} -> clip={pnr_1_clipped:.4f} -> PNR[1]: {pnr_1_decoded}"
            )

            # Highlight if there's a discrepancy
            if n_ctrl_decoded == 0 and (pnr_0_decoded > 0 or pnr_1_decoded > 0):
                print(f"        ⚠️  n_ctrl=0 but PNR values decoded as non-zero!")
            if n_ctrl_decoded == 1 and pnr_1_decoded > 0:
                print(
                    f"        ⚠️  n_ctrl=1 but PNR[1] decoded as non-zero (should be ignored)"
                )

    # 5. Use decoder to get official decoded params
    print("\n[5] Official decoder.decode() output...")
    g_jax = jnp.array(genotype, dtype=jnp.float32)
    params = decoder.decode(g_jax, cutoff)

    leaf_params = params["leaf_params"]
    leaf_active = params["leaf_active"]

    print("\n    Decoded leaf_params:")
    for leaf_i in range(decoder.leaves):
        n_ctrl = int(leaf_params["n_ctrl"][leaf_i])
        pnr = leaf_params["pnr"][leaf_i]
        is_active = bool(leaf_active[leaf_i])

        pnr_str = [int(pnr[j]) for j in range(len(pnr))]
        print(f"    Leaf {leaf_i}: active={is_active}, n_ctrl={n_ctrl}, PNR={pnr_str}")

    # 6. Compute state and expectation value
    print("\n[6] Computing state with JAX runner...")
    from src.simulation.jax.runner import jax_scoring_fn_batch

    # Build operator for GKP ground state
    alpha = float(config.get("alpha", 1.0))
    beta = complex(config.get("beta", 1 + 1j))

    from src.utils.gkp_operator import build_gkp_ground_state_operator

    operator = build_gkp_ground_state_operator(alpha, beta, cutoff)
    gs_eig = np.min(np.real(np.linalg.eigvalsh(np.array(operator))))

    g_batch = jnp.array(genotype[None, :], dtype=jnp.float32)

    fitnesses, descriptors, extras = jax_scoring_fn_batch(
        g_batch,
        cutoff=cutoff,
        operator=operator,
        genotype_name=genotype_name,
        genotype_config=config,
        pnr_max=pnr_max,
        gs_eig=gs_eig,
    )

    exp_val = float(-fitnesses[0, 0])
    log_prob = float(-fitnesses[0, 1])
    complexity = float(-fitnesses[0, 2])
    total_photons = float(-fitnesses[0, 3])

    print(f"\n    Computed Metrics:")
    print(f"        Expectation: {exp_val:.6f}")
    print(f"        LogProb:     {log_prob:.6f}")
    print(f"        Complexity:  {complexity:.2f}")
    print(f"        TotalPhotons: {total_photons:.2f}")

    # 7. Compare with stored metrics
    print("\n[7] Stored metrics from aggregation...")
    df = agg.get_pareto_front()
    if idx < len(df):
        stored_exp = df.iloc[idx]["Expectation"]
        stored_logp = df.iloc[idx]["LogProb"]
        stored_complex = df.iloc[idx].get("Complexity", "N/A")
        stored_photons = df.iloc[idx].get("TotalPhotons", "N/A")

        print(f"    Stored values:")
        print(f"        Expectation: {stored_exp}")
        print(f"        LogProb:     {stored_logp}")
        print(f"        Complexity:  {stored_complex}")
        print(f"        TotalPhotons: {stored_photons}")

        # Check for mismatch
        if abs(exp_val - stored_exp) > 0.01:
            print(
                f"\n    ⚠️  EXPECTATION MISMATCH: computed={exp_val:.6f}, stored={stored_exp}"
            )
        else:
            print(f"\n    ✅ Expectation values match!")
    else:
        print(f"    WARNING: Index {idx} not found in dataframe of length {len(df)}")

    # 8. Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)

    # Count active leaves with non-zero effective PNR
    active_with_pnr = 0
    for leaf_i in range(decoder.leaves):
        if leaf_active[leaf_i]:
            n_ctrl = int(leaf_params["n_ctrl"][leaf_i])
            pnr = leaf_params["pnr"][leaf_i]
            # Check effective PNR (only first n_ctrl values matter)
            has_nonzero = False
            for j in range(min(n_ctrl, len(pnr))):
                if int(pnr[j]) > 0:
                    has_nonzero = True
                    break
            if has_nonzero:
                active_with_pnr += 1

    print(f"\nActive leaves with non-zero effective PNR: {active_with_pnr}")

    # Check if expectation suggests non-Gaussianity
    if exp_val < 1.0:  # Good GKP states have expectation < 1
        print(
            f"\nExpectation = {exp_val:.4f} suggests HIGH NON-GAUSSIANITY (good GKP state)"
        )
        if active_with_pnr == 0:
            print("⚠️  BUT all active leaves claim PNR=0! This is INCONSISTENT!")
            print("    Possible causes:")
            print("    1. Raw genotype values may actually decode to non-zero PNR")
            print("    2. Inactive leaves are somehow affecting the simulation")
            print("    3. n_ctrl is being ignored in the simulation")
    else:
        print(f"\nExpectation = {exp_val:.4f} suggests GAUSSIAN-LIKE state")

    print("\n" + "=" * 80)
    return params, extras


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/debug_pnr_mismatch.py <experiment_path> <solution_idx>"
        )
        print(
            "Example: python scripts/debug_pnr_mismatch.py output/experiments/00B_c30_a1p00_b1p41 3711"
        )
        sys.exit(1)

    exp_path = sys.argv[1]
    idx = int(sys.argv[2])

    debug_solution(exp_path, idx)
