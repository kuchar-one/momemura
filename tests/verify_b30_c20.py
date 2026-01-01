import sys
import os
import numpy as np
import jax

# Ensure we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.genotypes.genotypes import get_genotype_decoder  # noqa: E402
from src.simulation.cpu.composer import SuperblockTopology, Composer  # noqa: E402


def verify_genotype(name: str, depth: int):
    print(f"\n--- Verifying {name} (Depth {depth}) ---")

    decoder = get_genotype_decoder(name, depth=depth)
    expected_len = decoder.get_length(depth)
    print(f"Expected Length: {expected_len}")

    # Create random genotype
    key = jax.random.PRNGKey(42)
    g = jax.random.normal(key, (expected_len,))

    decoded = decoder.decode(g, cutoff=10)

    # Check keys
    required_keys = ["homodyne_x", "mix_params", "leaf_active"]
    for k in required_keys:
        if k not in decoded:
            print(f"FAILED: Missing key {k}")
            return False

    # Check Homodyne X shape
    hom_x = decoded["homodyne_x"]
    L = 2**depth
    expected_nodes = L - 1

    if np.ndim(hom_x) > 0 and len(hom_x) == expected_nodes:
        print(f"SUCCESS: homodyne_x shape is ({len(hom_x)},) matching L-1 nodes.")
    else:
        print(
            f"FAILED: homodyne_x shape mismatch. Got {np.shape(hom_x)}, expected ({expected_nodes},)"
        )
        # For Design A/B/C original, it might be scalar?
        # But B30/C20 MUST be array.
        if name in ["B30", "C20"]:
            return False

    print("Decoding verified.")
    return True


def verify_cpu_topology_array_support():
    print("\n--- Verifying SuperblockTopology Array Support ---")

    depth = 3
    L = 2**depth
    n_nodes = L - 1

    # Create dummy topology
    topo = SuperblockTopology.from_full_binary(depth)
    composer = Composer(cutoff=5)

    # Dummy inputs
    fock_vecs = [np.zeros(5) for _ in range(L)]
    for v in fock_vecs:
        v[0] = 1.0  # |0>
    p_heralds = [1.0] * L

    # Array parameters
    # hom_x: array of length n_nodes
    # theta, phi: array of length n_nodes
    hom_x = np.linspace(-1, 1, n_nodes)
    theta = np.full(n_nodes, np.pi / 4)
    phi = np.zeros(n_nodes)

    print(f"Testing with hom_x shape {hom_x.shape}")

    try:
        # Evaluate
        state, prob = topo.evaluate_topology(
            composer,
            fock_vecs,
            p_heralds,
            homodyne_x=hom_x,
            theta=theta,
            phi=phi,
            exact_mixed=False,
        )
        print(f"Evaluation Successful. Prob: {prob:.4f}")
        return True
    except Exception as e:
        print(f"FAILED: Evaluation raised exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True

    success &= verify_genotype("B30", 3)
    success &= verify_genotype("C20", 3)

    success &= verify_cpu_topology_array_support()

    if success:
        print("\nALL CHECKS PASSED.")
        sys.exit(0)
    else:
        print("\nSOME CHECKS FAILED.")
        sys.exit(1)
