"""
Tests for the archive validator module with batched GPU support.
"""

import numpy as np
import jax.numpy as jnp
from unittest.mock import MagicMock, patch


def test_batch_compute_fidelities():
    """Test batch fidelity computation with mocked JAX runner."""
    from src.utils.archive_validator import batch_compute_fidelities

    # Mock data
    genotypes = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    base_cutoff = 10
    correction_cutoff = 15

    # Create distinct states for base and correction passes
    # Genotype 0: states are identical (fidelity 1.0)
    # Genotype 1: states are orthogonal (fidelity 0.0)

    state_0_base = np.zeros(base_cutoff)
    state_0_base[0] = 1.0

    state_1_base = np.zeros(base_cutoff)
    state_1_base[1] = 1.0

    # For correction pass, we need states valid at base cutoff dimension
    state_0_corr = state_0_base.copy()

    # State 1 correction is orthogonal to base
    state_1_corr = np.zeros(base_cutoff)
    state_1_corr[2] = 1.0

    # Mock return values for jax_scoring_fn_batch
    # Call 1 (base): returns states_base
    extras_base = {"final_state": jnp.array([state_0_base, state_1_base])}

    # Call 2 (correction): returns states_corr
    extras_corr = {"final_state": jnp.array([state_0_corr, state_1_corr])}

    # Patch the runner
    with patch("src.simulation.jax.runner.jax_scoring_fn_batch") as mock_runner:
        # Configure mock side effects for two calls
        mock_runner.side_effect = [
            (None, None, extras_base),  # First call (base)
            (None, None, extras_corr),  # Second call (correction)
        ]

        fidelities = batch_compute_fidelities(
            genotypes,
            base_cutoff=base_cutoff,
            correction_cutoff=correction_cutoff,
            genotype_name="test",
            genotype_config={},
            pnr_max=3,
        )

        assert len(fidelities) == 2
        assert np.isclose(fidelities[0], 1.0), (
            f"Expected fidelity 1.0, got {fidelities[0]}"
        )
        assert np.isclose(fidelities[1], 0.0), (
            f"Expected fidelity 0.0, got {fidelities[1]}"
        )

        # Verify calls
        assert mock_runner.call_count == 2

        # Verify call args
        args1, kwargs1 = mock_runner.call_args_list[0]
        assert args1[1] == base_cutoff
        assert kwargs1.get("correction_cutoff") is None  # No correction

        args2, kwargs2 = mock_runner.call_args_list[1]
        assert args2[1] == base_cutoff
        assert kwargs2.get("correction_cutoff") == correction_cutoff


def test_validate_and_clean_archive():
    """Test archive cleaning with batched validation."""
    from src.utils.archive_validator import validate_and_clean_archive

    # Mock repertoire
    mock_repertoire = MagicMock()
    # 2 solutions: index 0 is valid, index 1 is invalid

    genotypes = np.random.randn(2, 5)  # 2 genotypes
    fitnesses = np.array([[1.0, 0.5], [1.0, 0.5]])  # Both exist

    mock_repertoire.genotypes = genotypes
    mock_repertoire.fitnesses = fitnesses

    # Mock batch_compute_fidelities to return [1.0, 0.0]
    # We mock inside the module where it is used
    with patch("src.utils.archive_validator.batch_compute_fidelities") as mock_batch:
        mock_batch.return_value = np.array([1.0, 0.0])

        cleaned_rep, num_removed = validate_and_clean_archive(
            mock_repertoire,
            base_cutoff=10,
            correction_cutoff=15,
            genotype_name="test",
            genotype_config={},
            fidelity_threshold=0.9,
        )

        assert num_removed == 1

        # Check replacement called
        mock_repertoire.replace.assert_called_once()

        # Check fitnesses updated (second solution should be -inf)
        call_args = mock_repertoire.replace.call_args
        new_fitnesses = call_args.kwargs["fitnesses"]

        assert np.all(np.isfinite(new_fitnesses[0]))
        assert np.all(new_fitnesses[1] == -np.inf)


if __name__ == "__main__":
    test_batch_compute_fidelities()
    test_validate_and_clean_archive()
    print("All archive validator tests passed!")
