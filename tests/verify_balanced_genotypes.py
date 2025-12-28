import sys
import os
import jax.numpy as jnp

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.genotypes.genotypes import (  # noqa: E402
    DesignC2BGenotype,
    DesignC20BGenotype,
    DesignB3BGenotype,
    DesignB30BGenotype,
    DesignC2Genotype,
    DesignC20Genotype,
    DesignB3Genotype,
    DesignB30Genotype,
)


def verify_genotype(name, cls_balanced, cls_base, tied_mix=False):
    print(f"\n--- Verifying {name} vs Base ---")

    # Defaults
    depth = 3

    base_obj = cls_base(depth=depth)
    bal_obj = cls_balanced(depth=depth)

    # 1. Length Check
    len_base = base_obj.get_length(depth)
    len_bal = bal_obj.get_length(depth)

    nodes = 2**depth - 1

    if tied_mix:
        # C2/C20: Tied mixing (3 params total usually, wait C1 implies 4 params?)
        # My implementation assumed 3 params (theta, phi, varphi).
        # C1 length calc: "4(Mix)". Wait.
        # Let's check diff.
        expected_diff = 3
    else:
        # B3/B30: Independent mixing (3 * nodes)
        expected_diff = 3 * nodes

    diff = len_base - len_bal
    print(f"Length Base: {len_base}, Balanced: {len_bal}, Diff: {diff}")

    if diff != expected_diff:
        print(f"FAIL: Expected difference {expected_diff}, got {diff}")
        return False

    # 2. Decode Check
    # Create random genome of balanced length
    key = jnp.arange(len_bal, dtype=float) / len_bal  # deterministic

    decoded = bal_obj.decode(key, cutoff=10)

    # Check Mix Params
    mix = decoded["mix_params"]
    print(f"Mix Params shape: {mix.shape}")

    # Expected: All rows are [pi/4, 0, 0]
    expected_row = jnp.array([jnp.pi / 4, 0.0, 0.0])

    # Check first row
    row0 = mix[0]
    err = jnp.linalg.norm(row0 - expected_row)
    print(f"Mix Error (Row 0): {err}")

    if err > 1e-6:
        print(f"FAIL: Mix params not balanced. Got {row0}")
        return False

    # Check if all rows identical (broadcast/fill check)
    if not jnp.allclose(mix, expected_row, atol=1e-6):
        print("FAIL: Not all mix params are balanced.")
        return False

    print(f"PASS: {name} validated.")
    return True


def run_tests():
    # C2B vs C2 (Tied)
    if not verify_genotype("C2B", DesignC2BGenotype, DesignC2Genotype, tied_mix=True):
        return False

    # C20B vs C20 (Tied)
    if not verify_genotype(
        "C20B", DesignC20BGenotype, DesignC20Genotype, tied_mix=True
    ):
        return False

    # B3B vs B3 (Independent)
    if not verify_genotype("B3B", DesignB3BGenotype, DesignB3Genotype, tied_mix=False):
        return False

    # B30B vs B30 (Independent)
    if not verify_genotype(
        "B30B", DesignB30BGenotype, DesignB30Genotype, tied_mix=False
    ):
        return False

    return True


if __name__ == "__main__":
    if run_tests():
        print("\nAll Balanced Genotypes Verified Successfully.")
        sys.exit(0)
    else:
        sys.exit(1)
