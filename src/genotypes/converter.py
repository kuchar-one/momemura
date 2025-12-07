import numpy as np
from src.genotypes.genotypes import get_genotype_decoder


def create_vacuum_genotype(
    genotype_name: str, depth: int = 3, config: dict = None
) -> np.ndarray:
    """
    Creates a 'vacuum' genotype (all zeros) for the specified design key.
    Physical meaning:
      - R=0 (no squeezing)
      - D=0 (no displacement)
      - Angles=0 (identity unitaries)
      - Active=False (if flag < 0)
    """
    decoder = get_genotype_decoder(genotype_name, depth=depth, config=config)
    length = decoder.get_length(depth)
    # Zeros map to:
    # tanh(0) = 0 -> R=0, D=0
    # tanh(0) = 0 -> Angles=0
    # 0.0 -> Active False (if threshold is > 0)
    # Clip(0) -> PNR=0
    # So strictly zero vector is perfect Identity.
    return np.zeros(length, dtype=np.float32)


def upgrade_genotype(
    source_g: np.ndarray,
    source_name: str,
    target_name: str,
    depth: int = 3,
    config: dict = None,
) -> np.ndarray:
    """
    Converts a genotype from a simpler design to a more complex one.
    Supported upgrades:
      - C1/C2 -> B1/B2 -> A
      - C1 -> C2, B1 -> B2 (Adding active flags)
    """
    if source_name == target_name:
        return source_g.copy()

    if "legacy" in source_name.lower() or "legacy" in target_name.lower():
        raise NotImplementedError("Conversion involving 'Legacy' is not supported.")

    # Step 1: Extract "Expanded" from Source
    # We pass config so we get correct lengths for the source genotype
    expanded = _extract_to_expanded(source_g, source_name, depth, config)

    # Step 2: Flatten "Expanded" to Target
    target_g = _flatten_from_expanded(expanded, target_name, depth, config)

    return target_g


def _extract_to_expanded(g, name, depth, config):
    # Returns: hom_x(1), blocks(L, params_per_leaf), mix(Nodes, PN), final(F), active(L)
    decoder = get_genotype_decoder(name, depth, config)
    L = 2**depth
    nodes = L - 1

    # Common: Hom(1) at 0
    hom_x = g[0:1]

    if name == "A":
        # A: Hom(1) + L * P_leaf_full + Mix + Final
        # P_leaf_full includes Active (1) + Params
        block_size = decoder.P_leaf_full
        blocks_start = 1
        blocks_len = L * block_size

        blocks_raw = g[blocks_start : blocks_start + blocks_len].reshape(L, block_size)

        # Split Active (idx 0) and Params (1..end)
        active_raw = blocks_raw[:, 0]
        params_raw = blocks_raw[:, 1:]

        mix_start = blocks_start + blocks_len
        mix_len = nodes * decoder.PN
        mix_raw = g[mix_start : mix_start + mix_len].reshape(nodes, decoder.PN)

        final_start = mix_start + mix_len
        final_len = decoder.F
        final_raw = g[final_start : final_start + final_len]

        return hom_x, params_raw, mix_raw, final_raw, active_raw

    elif name in ["B1", "B2"]:
        # B1: Hom(1) + Shared(BP) + Mix + Final
        shared_start = 1
        shared_len = decoder.BP
        shared_block = g[shared_start : shared_start + shared_len]

        # Broadcast block to L
        # Note: B/C shared block does NOT have active flag (it's params only)
        # So we can use it directly as params_raw for A
        blocks_raw = np.tile(shared_block, (L, 1))

        mix_start = shared_start + shared_len
        mix_len = nodes * decoder.PN
        mix_raw = g[mix_start : mix_start + mix_len].reshape(nodes, decoder.PN)

        final_start = mix_start + mix_len
        final_len = decoder.F
        final_raw = g[final_start : final_start + final_len]

        if name == "B2":
            active_start = final_start + final_len
            active_raw = g[active_start : active_start + L]
        else:
            active_raw = np.ones(L, dtype=np.float32) * 1.0

        return hom_x, blocks_raw, mix_raw, final_raw, active_raw

    elif name == "B3":
        # B3: Hom(1) + Shared(Sharedv) + L*Unique(U) + Mix + Final
        # Need to reconstruct "Logical Blocks" (BP) for compatibility.
        # BP Layout: NCtrl, TMSS, US, UC, Disp, PNR.

        shared_start = 1
        shared_len = decoder.Sharedv
        shared_raw = g[shared_start : shared_start + shared_len]

        unique_start = shared_start + shared_len
        unique_len = L * decoder.Unique
        unique_raw = g[unique_start : unique_start + unique_len].reshape(
            L, decoder.Unique
        )

        # B3 Unique: Active(0), NCtrl(1), PNR(2...)
        active_raw = unique_raw[:, 0]
        n_ctrl_raw = unique_raw[:, 1:2]
        pnr_raw = unique_raw[:, 2:]

        # B3 Shared: TMSS(0), US(1), UC..., Disp...
        tmss_us_uc_disp = shared_raw  # All shared params

        # Broadcast Shared
        broadcast_shared = np.tile(tmss_us_uc_disp, (L, 1))

        # Reassemble Logical Blocks (BP)
        # Order: NCtrl, Shared(TMSS..Disp), PNR
        # Note: shared_raw order in B3 is TMSS, US, UC, Disp.
        # BP order in A/B1: NCtrl, TMSS, US, UC, Disp, PNR.
        # So we sandwich Shared between NCtrl and PNR.
        blocks_list = [n_ctrl_raw, broadcast_shared, pnr_raw]
        blocks_raw = np.hstack(blocks_list)

        mix_start = unique_start + unique_len
        mix_len = nodes * decoder.PN
        mix_raw = g[mix_start : mix_start + mix_len].reshape(nodes, decoder.PN)

        final_start = mix_start + mix_len
        final_len = decoder.F
        final_raw = g[final_start : final_start + final_len]

        return hom_x, blocks_raw, mix_raw, final_raw, active_raw

    elif name in ["C1", "C2"]:
        # C1: Hom + Shared(BP) + SharedMix(PN) + Final
        shared_start = 1
        shared_len = decoder.BP
        shared_block = g[shared_start : shared_start + shared_len]
        blocks_raw = np.tile(shared_block, (L, 1))

        mix_start = shared_start + shared_len
        mix_len = decoder.PN
        shared_mix = g[mix_start : mix_start + mix_len]
        mix_raw = np.tile(shared_mix, (nodes, 1))

        final_start = mix_start + mix_len
        final_len = decoder.F
        final_raw = g[final_start : final_start + final_len]

        if name == "C2":
            active_start = final_start + final_len
            active_raw = g[active_start : active_start + L]
        else:
            active_raw = np.ones(L, dtype=np.float32) * 1.0

        return hom_x, blocks_raw, mix_raw, final_raw, active_raw

    else:
        raise ValueError(f"Unknown genotype: {name}")


def _flatten_from_expanded(expanded, name, depth, config):
    hom_x, blocks_raw, mix_raw, final_raw, active_raw = expanded

    parts = [hom_x]

    if name == "A":
        # A: Blocks include Active at pos 0
        # blocks_raw is (L, 15). active_raw is (L,)
        # Combine -> (L, 16)

        # We need to interleave or just concat: [Act, Param, Param...]
        # active_raw shape (L,) -> (L, 1)
        active_col = active_raw[:, None]

        # Concat
        full_blocks = np.hstack([active_col, blocks_raw])
        parts.append(full_blocks.flatten())

        parts.append(mix_raw.flatten())
        parts.append(final_raw)

    elif name in ["B1", "B2"]:
        # B: Shared Block?
        # WAIT. We have L blocks in `expanded`.
        # If target is B, we can only support it if ALL L blocks are identical.
        # Otherwise, lossy compression?
        # User requested: "simple -> complex".
        # So we should generally NOT go A -> B.
        # But we might go C -> B.
        # In that case, C's blocks are already identical (tiled).
        # We can just take the first one.

        # Verify identical?
        # For a seeding tool, we assume valid "upcast".
        # If we "downcast" (A -> B), we might just take the mean or the first?
        # Let's take the mean of blocks to find a "centroid" shared block.

        avg_block = np.mean(blocks_raw, axis=0)  # (15,)
        parts.append(avg_block)

        parts.append(mix_raw.flatten())
        parts.append(final_raw)

        if name == "B2":
            parts.append(active_raw)

    elif name == "B3":
        # B3: Shared(TMSS..Disp) + Unique(Act, NCtrl, PNR)

        # 1. Extract Shared (Mean of blocks)
        # Logical Block: NCtrl(0), Shared(1..-PNR), PNR(-PNR..)
        # We need to identify indices.
        # decoder = get_genotype_decoder("B3", depth=depth, config=config)
        # But we don't have decoder here easily? We passed depth/config.
        # We can instantiate dummy decoder to get lengths.
        from src.genotypes.genotypes import get_genotype_decoder

        decoder = get_genotype_decoder("B3", depth=depth, config=config)

        # Calculate indices based on Sharedv
        # Sharedv in B3 = TMSS(1)+US(1)+UC+Disp = Total Shared Params.
        # In Logical Block (BP), these are indices 1 to 1+Sharedv.
        # Col 0 is NCtrl.
        # Col 1..1+Sharedv is Shared.
        # Col 1+Sharedv..End is PNR.

        shared_start_idx = 1
        shared_end_idx = 1 + decoder.Sharedv

        # Take mean of shared columns
        shared_cols = blocks_raw[:, shared_start_idx:shared_end_idx]
        avg_shared = np.mean(shared_cols, axis=0)
        parts.append(avg_shared)

        # 2. Extract Unique (Per Leaf)
        # Active (from active_raw), NCtrl (Col 0), PNR (Col -PNR..)

        n_ctrl_col = blocks_raw[:, 0:1]  # (L, 1)
        pnr_cols = blocks_raw[:, shared_end_idx:]  # (L, N_C)
        active_col = active_raw[:, None]  # (L, 1)

        # B3 Unique Order: Active, NCtrl, PNR
        unique_block = np.hstack([active_col, n_ctrl_col, pnr_cols])  # (L, Unique)
        parts.append(unique_block.flatten())

        parts.append(mix_raw.flatten())
        parts.append(final_raw)

    elif name in ["C1", "C2"]:
        # C: Shared Block + Shared Mix
        avg_block = np.mean(blocks_raw, axis=0)
        parts.append(avg_block)

        avg_mix = np.mean(mix_raw, axis=0)
        parts.append(avg_mix)

        parts.append(final_raw)

        if name == "C2":
            parts.append(active_raw)

    else:
        raise ValueError(f"Unknown target: {name}")

    return np.concatenate(parts)
