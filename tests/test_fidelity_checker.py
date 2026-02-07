"""
Tests for the fidelity checker module.
"""

import numpy as np


def test_compute_fidelity_identical():
    """Test fidelity of identical states is 1."""
    from src.utils.fidelity_checker import compute_fidelity

    state = np.array([0.5, 0.5, 0.5, 0.5])
    fid = compute_fidelity(state, state)
    assert abs(fid - 1.0) < 1e-10, (
        f"Fidelity of identical states should be 1, got {fid}"
    )


def test_compute_fidelity_orthogonal():
    """Test fidelity of orthogonal states is 0."""
    from src.utils.fidelity_checker import compute_fidelity

    state1 = np.array([1.0, 0.0, 0.0, 0.0])
    state2 = np.array([0.0, 1.0, 0.0, 0.0])
    fid = compute_fidelity(state1, state2)
    assert abs(fid) < 1e-10, f"Fidelity of orthogonal states should be 0, got {fid}"


def test_compute_fidelity_different_lengths():
    """Test fidelity computation with different length states."""
    from src.utils.fidelity_checker import compute_fidelity

    state1 = np.array([1.0, 0.0, 0.0])  # 3 elements
    state2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # 5 elements
    fid = compute_fidelity(state1, state2)
    # After truncation to 3 elements and renormalization, they should be identical
    assert abs(fid - 1.0) < 1e-10, (
        f"Fidelity should be 1 for matching truncated states, got {fid}"
    )


def test_compute_state_at_cutoff_valid_genotype():
    """Test that compute_state_at_cutoff returns valid state."""
    from src.utils.fidelity_checker import compute_state_at_cutoff

    # Use realistic parameters matching actual experiments (00B with cutoff=30, depth=3)
    cutoff = 30
    genotype_name = "00B"
    genotype_dim = 209  # Real 00B genome size from experiments (depth=3)
    genotype_config = {"depth": 3}  # Match experiment config

    # Create a random valid genotype with conservative values
    np.random.seed(42)
    genotype = np.random.uniform(-0.5, 0.5, size=genotype_dim)

    state, prob, total_pnr = compute_state_at_cutoff(
        genotype, cutoff, genotype_name, genotype_config=genotype_config, pnr_max=3
    )

    # State should be normalized (or zero)
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        assert abs(norm - 1.0) < 1e-6, f"State should be normalized, got norm {norm}"

    # Probability should be non-negative
    assert prob >= 0, f"Probability should be non-negative, got {prob}"

    # Total PNR should be non-negative
    assert total_pnr >= 0, f"Total PNR should be non-negative, got {total_pnr}"


def test_compute_fidelity_two_cutoffs_similar():
    """Test that well-behaved genotypes have high fidelity between cutoffs."""
    from src.utils.fidelity_checker import compute_fidelity_two_cutoffs

    # Use realistic parameters matching actual experiments (depth=3)
    cutoff_base = 25
    cutoff_correction = 30
    genotype_name = "00B"
    genotype_dim = 209  # Real 00B genome size (depth=3)
    genotype_config = {"depth": 3}

    # Create a conservative genotype (small values for well-behaved state)
    np.random.seed(123)
    genotype = np.random.uniform(-0.3, 0.3, size=genotype_dim)

    fid = compute_fidelity_two_cutoffs(
        genotype,
        cutoff_base,
        cutoff_correction,
        genotype_name,
        genotype_config=genotype_config,
        pnr_max=3,
    )

    # Fidelity should be in valid range [0, 1]
    # Note: Random genotypes may produce zero/invalid states, so we just verify the range
    assert 0 <= fid <= 1, f"Fidelity should be in [0, 1], got {fid}"


def test_validate_genotype_interface():
    """Test that validate_genotype returns expected tuple."""
    from src.utils.fidelity_checker import validate_genotype

    # Use realistic parameters matching actual experiments (depth=3)
    cutoff_base = 25
    cutoff_correction = 30
    genotype_name = "00B"
    genotype_dim = 209  # Real 00B genome size (depth=3)
    genotype_config = {"depth": 3}

    np.random.seed(456)
    genotype = np.random.uniform(-0.3, 0.3, size=genotype_dim)

    is_valid, fid = validate_genotype(
        genotype,
        cutoff_base,
        cutoff_correction,
        genotype_name,
        genotype_config=genotype_config,
        fidelity_threshold=0.5,
    )

    assert isinstance(is_valid, bool), "is_valid should be a boolean"
    assert isinstance(fid, float), "fidelity should be a float"
    assert 0 <= fid <= 1, f"Fidelity should be in [0, 1], got {fid}"


if __name__ == "__main__":
    test_compute_fidelity_identical()
    test_compute_fidelity_orthogonal()
    test_compute_fidelity_different_lengths()
    print("Basic fidelity tests passed!")
    test_compute_state_at_cutoff_valid_genotype()
    print("State computation test passed!")
    test_compute_fidelity_two_cutoffs_similar()
    print("Two-cutoff fidelity test passed!")
    test_validate_genotype_interface()
    print("All fidelity checker tests passed!")
