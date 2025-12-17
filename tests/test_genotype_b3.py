import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.genotypes.genotypes import DesignB3Genotype


def test_design_b3_structure():
    # Setup
    depth = 2  # 4 leaves
    # config = {"n_control": 2, "pnr_max": 3, "r_scale": 2.0}
    # Using defaults from class if None

    genotype = DesignB3Genotype(depth=depth)
    length = genotype.get_length(depth)

    print(f"B3 Length (Depth {depth}): {length}")

    # Formula check
    # Sharedv = 1(TMSS) + 1(US) + 3(UC, N=2) + 2(DispS) + 4(DispC) = 11
    # Unique = 1(Act) + 1(NCtrl) + 2(PNR) = 4
    # Length = 1(Hom) + 11(Shared) + 4*4(Unique) + 4*3(Mix) + 5(Final)
    # = 1 + 11 + 16 + 12 + 5 = 45

    # Using default N_C=1 (BaseGenotype default?)
    # BaseGenotype default n_control=1 unless config overrides.
    # Let's verify defaults.
    assert genotype.n_control == 1

    # Recalculate for N_C=1
    # Sharedv = 1 + 1 + 1(UC) + 2 + 2 = 7
    # Unique = 1 + 1 + 1 = 3
    # Length = 1 + 7 + 4*3 + 12 + 5 = 1 + 7 + 12 + 12 + 5 = 37

    assert length == 34, f"Expected 34, got {length}"


def test_design_b3_independence():
    # Verify that different leaves can have different PNRs/NCtrls
    # while sharing TMSS.

    # BaseGenotype uses 'modes' to determine n_control (modes - 1)
    config = {"modes": 3, "pnr_max": 5}
    depth = 1  # 2 Leaves
    genotype = DesignB3Genotype(depth=depth, config=config)

    print(f"DEBUG: Genotype N_Control: {genotype.n_control}")
    print(f"DEBUG: Config: {genotype.config}")

    L = 2

    # Calculate indices
    # Sharedv (N=2) = 1(TMSS) + 1(US) + 4(UC: 2*1+2) + 2(DispS) + 4(DispC) = 12
    # Unique = 1(Act) + 1(NCtrl) + 2(PNR) = 4
    # Mix = 4 * (2-1) = 4
    # Final = 5
    # Total = 1 + 12 + 2*4 + 4 + 5 = 30

    g = np.zeros(30, dtype=np.float32)

    # Set Homodyne X
    g[0] = 0.5  # tanh(0.5) * 4.0 ~ 1.8

    # Set Shared TMSS (Index 1 + 0)
    # Target r = 1.0. tanh(x)*2.0 = 1.0 -> tanh(x)=0.5 -> x ~ 0.55
    g[1] = 0.55

    # Set Shared US Phase (Index 1 + 1) -> 0.0

    # Set Unique Block
    # Unique Start = 1 + 12 = 13
    # Leaf 0: Active=1, NCtrl=0, PNR=[1, 1]
    # Leaf 1: Active=1, NCtrl=2, PNR=[0, 5]

    u_idx = 13
    # Leaf 0
    g[u_idx] = 1.0  # Active > 0
    g[u_idx + 1] = -0.9  # NCtrl -> 0
    # PNR [1, 1]. Max=5. Clip(0,1). 1/5 = 0.2
    g[u_idx + 2] = 0.21
    g[u_idx + 3] = 0.21

    # Leaf 1
    u_idx += 4
    g[u_idx] = 1.0  # Active
    g[u_idx + 1] = 0.9  # NCtrl -> 2 (Max)
    # PNR [0, 5].
    g[u_idx + 2] = 0.0
    g[u_idx + 3] = 1.0  # Max

    # Decode
    g_jax = jnp.array(g)
    decoded = genotype.decode(g_jax, cutoff=10)

    leaf_params = decoded["leaf_params"]

    # Verify Shared TMSS is identical
    r_vals = leaf_params["tmss_r"]
    print(f"TMSS R: {r_vals}")
    assert jnp.allclose(r_vals[0], r_vals[1]), "TMSS should be shared"
    assert r_vals[0] > 0.9, "TMSS should be ~1.0"

    # Verify Unique NCtrl
    n_ctrls = leaf_params["n_ctrl"]
    print(f"NCtrl: {n_ctrls}")
    assert n_ctrls[0] == 0
    assert n_ctrls[1] == 2

    # Verify Unique PNR
    pnrs = leaf_params["pnr"]
    print(f"PNR: {pnrs}")
    assert jnp.array_equal(pnrs[0], jnp.array([1, 1]))
    assert jnp.array_equal(pnrs[1], jnp.array([0, 5]))

    # Total Photons check: Sum(PNR)
    # Leaf 0: 2
    # Leaf 1: 5
    # Complexity: 2 active leaves
    # If tied (B2), they would be equal. Here they differ.
    # This PROVES B3 breaks the redundancy.
    print(
        "Design B3 successfully enables unique PNRs per leaf with shared continuous params!"
    )


if __name__ == "__main__":
    test_design_b3_structure()
    test_design_b3_independence()
