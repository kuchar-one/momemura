"""
Tests for the fidelity checker module.
"""

import numpy as np
import pytest


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


@pytest.mark.skip(reason="00B genotype requires huge genome size based on cutoff")
def test_compute_state_at_cutoff_valid_genotype():
    """Test that compute_state_at_cutoff returns valid state."""
    from src.utils.fidelity_checker import compute_state_at_cutoff

    # Use a simple genotype from the 00B type
    cutoff = 20
    genotype_name = "00B"

    # Create a random valid genotype
    np.random.seed(42)
    # Need to know the genome size for 00B
    from src.genotypes.genotypes import get_genotype_decoder

    decoder = get_genotype_decoder(genotype_name, cutoff, None)
    genome_size = decoder.get_length()
    genotype = np.random.uniform(-1, 1, size=genome_size)

    state, prob, total_pnr = compute_state_at_cutoff(
        genotype, cutoff, genotype_name, genotype_config=None, pnr_max=3
    )

    # State should be normalized (or zero)
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        assert abs(norm - 1.0) < 1e-6, f"State should be normalized, got norm {norm}"

    # Probability should be non-negative
    assert prob >= 0, f"Probability should be non-negative, got {prob}"

    # Total PNR should be non-negative
    assert total_pnr >= 0, f"Total PNR should be non-negative, got {total_pnr}"


@pytest.mark.skip(reason="00B genotype requires huge genome size based on cutoff")
def test_compute_fidelity_two_cutoffs_similar():
    """Test that well-behaved genotypes have high fidelity between cutoffs."""
    from src.utils.fidelity_checker import compute_fidelity_two_cutoffs

    # Use a conservative genotype (small parameters)
    cutoff_base = 15
    cutoff_correction = 20
    genotype_name = "00B"

    # Create a conservative genotype
    np.random.seed(123)
    from src.genotypes.genotypes import get_genotype_decoder

    decoder = get_genotype_decoder(genotype_name, cutoff_correction, None)
    genome_size = decoder.get_length()
    # Small values to ensure well-behaved state
    genotype = np.random.uniform(-0.3, 0.3, size=genome_size)

    fid = compute_fidelity_two_cutoffs(
        genotype, cutoff_base, cutoff_correction, genotype_name, pnr_max=3
    )

    # Fidelity should be reasonably high for conservative parameters
    # (exact threshold depends on the genotype, but should be > 0.5)
    assert fid > 0.5, f"Expected high fidelity for conservative genotype, got {fid}"


@pytest.mark.skip(reason="00B genotype requires huge genome size based on cutoff")
def test_validate_genotype_interface():
    """Test that validate_genotype returns expected tuple."""
    from src.utils.fidelity_checker import validate_genotype

    cutoff_base = 15
    cutoff_correction = 20
    genotype_name = "00B"

    np.random.seed(456)
    from src.genotypes.genotypes import get_genotype_decoder

    decoder = get_genotype_decoder(genotype_name, cutoff_correction, None)
    genotype = np.random.uniform(-0.3, 0.3, size=decoder.get_length())

    is_valid, fid = validate_genotype(
        genotype, cutoff_base, cutoff_correction, genotype_name, fidelity_threshold=0.5
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
