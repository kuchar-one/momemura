import numpy as np
from src.genotypes.converter import upgrade_genotype
from src.genotypes.genotypes import get_genotype_decoder


def test_heuristic_conversion():
    """Verify that 'legacy' source name allows pass-through if lengths match."""

    # Target: Genotype 0, Depth 3
    target_name = "0"
    depth = 3
    config = {"modes": 3}

    dec = get_genotype_decoder(target_name, depth, config)
    target_len = dec.get_length(depth)

    print(f"Target Length: {target_len}")

    # Create fake genotype of correct length
    dummy_g = np.random.rand(target_len)

    # Try converting from "legacy"
    # Should succeed because lengths match
    result = upgrade_genotype(dummy_g, "legacy", target_name, depth, config)

    assert np.array_equal(result, dummy_g)
    print("Same length legacy -> 0: SUCCESS")

    # Try converting mismatch length
    # Should fail
    wrong_g = np.random.rand(target_len - 1)

    try:
        upgrade_genotype(wrong_g, "legacy", target_name, depth, config)
        print("Mismatch length: FAILED (Should have raised Error)")
        assert False
    except NotImplementedError:
        print("Mismatch length: SUCCESS (Raised Error)")
    except Exception as e:
        print(f"Mismatch length: Unexpected Error {e}")
        assert False


if __name__ == "__main__":
    test_heuristic_conversion()
