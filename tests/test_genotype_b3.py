import jax.numpy as jnp
import numpy as np
from src.genotypes.genotypes import DesignB3Genotype


def test_design_b3_structure():
    # Setup
    depth = 2  # 4 leaves
    # Use explicit config to be safe
    config = {"modes": 3}  # N=3 -> N_Ctrl=2

    genotype = DesignB3Genotype(depth=depth, config=config)
    length = genotype.get_length(depth)

    print(f"B3 Length (Depth {depth}): {length}")

    # Formula check for N=3:
    # Sharedv = GG_Len
    #   r(3) + ph(9) + disp(6) = 18
    # Unique = 1(Act) + 1(NCtrl) + 2(PNR) = 4
    # Mix = 3 * (Nodes=3) = 9
    # Final = 5
    # Total = 1 + 18 + 4*4 + 9 + 5 = 1 + 18 + 16 + 9 + 5 = 49

    assert genotype.n_control == 2
    assert length == 49, f"Expected 49, got {length}"


def test_design_b3_independence():
    # Verify that different leaves can have different PNRs/NCtrls
    # while sharing TMSS.

    config = {"modes": 3, "pnr_max": 5}
    depth = 1  # 2 Leaves
    genotype = DesignB3Genotype(depth=depth, config=config)

    # Use dynamic length from decoder
    length = genotype.get_length(depth)
    g = np.zeros(length, dtype=np.float32)

    # Set Homodyne X (idx 0)
    g[0] = 0.5

    # Shared Block starts at 1
    # GG Params: r(3) -> phases(9) -> disp(6). Total 18.
    # Set r[0] (Signal squeezing) to ~1.0
    # tanh(0.55)*2.0 approx 1.0
    g[1] = 0.55

    # Unique Blocks start after Shared (1+18 = 19)
    # Unique Len = 4 (Act, NC, PNR, PNR)
    u_idx = 19

    # Leaf 0
    g[u_idx] = 1.0  # Active
    g[u_idx + 1] = -0.9  # NCtrl -> 0
    # PNR [1, 1] -> 0.21 * 5 > 1
    g[u_idx + 2] = 0.21
    g[u_idx + 3] = 0.21

    # Leaf 1
    u_idx += 4
    g[u_idx] = 1.0
    g[u_idx + 1] = 0.9  # NCtrl -> 2
    # PNR [0, 5]
    g[u_idx + 2] = 0.0
    g[u_idx + 3] = 1.0

    # Decode
    g_jax = jnp.array(g)
    decoded = genotype.decode(g_jax, cutoff=10)

    leaf_params = decoded["leaf_params"]

    # Verify Shared TMSS is identical (first mode r)
    r_vals = leaf_params["r"]
    print(f"R Vals: {r_vals}")
    # r_vals shape (L, 3)
    assert jnp.allclose(r_vals[0], r_vals[1]), "TMSS params should be shared"
    assert r_vals[0, 0] > 0.9, "Signal squeezing should be ~1.0"

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

    print("Design B3 independence verified.")


if __name__ == "__main__":
    test_design_b3_structure()
    test_design_b3_independence()
