"""
Tests for the archive validator module.

Note: These tests use mock repertoires since full integration with QDax
requires large genotypes and significant compute time.
"""

import numpy as np
import pytest


def test_get_archive_valid_genotypes():
    """Test extraction of valid genotypes from archive."""
    from src.utils.archive_validator import get_archive_valid_genotypes
    from unittest.mock import MagicMock

    # Create mock repertoire
    mock_repertoire = MagicMock()

    # 3x3 archive with 2 valid and 7 empty cells
    genotypes = np.random.randn(3, 3, 10)  # 10-dim genotypes
    fitnesses = np.full((3, 3, 2), -np.inf)  # 2 objectives, all empty initially
    fitnesses[0, 1] = [0.5, 0.3]  # Valid
    fitnesses[2, 2] = [0.7, 0.1]  # Valid

    mock_repertoire.genotypes = genotypes
    mock_repertoire.fitnesses = fitnesses

    valid_genotypes, valid_indices = get_archive_valid_genotypes(mock_repertoire)

    assert len(valid_genotypes) == 2, (
        f"Expected 2 valid genotypes, got {len(valid_genotypes)}"
    )
    assert len(valid_indices) == 2, (
        f"Expected 2 valid indices, got {len(valid_indices)}"
    )


def test_validate_and_clean_archive_interface():
    """Test that validate_and_clean_archive has correct interface."""
    from src.utils.archive_validator import validate_and_clean_archive

    # Check function signature
    import inspect

    sig = inspect.signature(validate_and_clean_archive)
    params = list(sig.parameters.keys())

    assert "repertoire" in params
    assert "base_cutoff" in params
    assert "correction_cutoff" in params
    assert "fidelity_threshold" in params


def test_final_archive_validation_interface():
    """Test that final_archive_validation has correct interface."""
    from src.utils.archive_validator import final_archive_validation

    # Check function signature
    import inspect

    sig = inspect.signature(final_archive_validation)
    params = list(sig.parameters.keys())

    assert "repertoire" in params
    assert "max_iterations" in params
    assert "fidelity_threshold" in params


if __name__ == "__main__":
    test_get_archive_valid_genotypes()
    test_validate_and_clean_archive_interface()
    test_final_archive_validation_interface()
    print("All archive validator tests passed!")
