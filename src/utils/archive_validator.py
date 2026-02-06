"""
Archive validator for dual-cutoff artifact validation.

Provides utilities to validate and clean archives by removing solutions
that fail dual-cutoff fidelity checks (Options D and F).
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, Any, List


def validate_and_clean_archive(
    repertoire,
    base_cutoff: int,
    correction_cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
    fidelity_threshold: float = 0.9,
) -> Tuple[Any, int]:
    """
    Validate all archive solutions and remove invalid ones (Option D).

    Args:
        repertoire: QDax MOME repertoire.
        base_cutoff: Lower cutoff for fidelity check.
        correction_cutoff: Higher cutoff for fidelity check.
        genotype_name: Name of genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.
        fidelity_threshold: Minimum fidelity to be considered valid.

    Returns:
        Tuple of (cleaned_repertoire, num_removed).
    """
    from src.utils.fidelity_checker import validate_genotype

    # Extract all genotypes from repertoire
    genotypes = np.array(repertoire.genotypes)
    fitnesses = np.array(repertoire.fitnesses)

    # Find non-empty cells (valid solutions)
    valid_mask = np.all(np.isfinite(fitnesses), axis=-1)
    valid_indices = np.where(valid_mask)

    num_removed = 0
    removal_mask = np.zeros_like(valid_mask, dtype=bool)

    # Check each valid solution
    for idx in zip(*valid_indices):
        genotype = genotypes[idx]

        # Validate using dual-cutoff fidelity
        is_valid, fidelity = validate_genotype(
            genotype,
            base_cutoff,
            correction_cutoff,
            genotype_name,
            genotype_config,
            pnr_max,
            fidelity_threshold,
        )

        if not is_valid:
            removal_mask[idx] = True
            num_removed += 1
            print(f"  Removing artifact at {idx}: fidelity={fidelity:.3f}")

    if num_removed > 0:
        # Create new fitness array with removed solutions set to -inf
        new_fitnesses = fitnesses.copy()
        new_fitnesses[removal_mask] = -np.inf

        # Create cleaned repertoire
        # Note: QDax repertoire immutability means we return a modified copy
        cleaned_repertoire = repertoire.replace(fitnesses=jnp.array(new_fitnesses))
        return cleaned_repertoire, num_removed

    return repertoire, 0


def final_archive_validation(
    repertoire,
    base_cutoff: int,
    correction_cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
    fidelity_threshold: float = 0.9,
    max_iterations: int = 10,
) -> Tuple[Any, int]:
    """
    Iteratively validate and clean archive until stable (Option F).

    Repeatedly removes invalid solutions and rechecks until no more
    removals occur or max_iterations reached.

    Args:
        repertoire: QDax MOME repertoire.
        base_cutoff: Lower cutoff for fidelity check.
        correction_cutoff: Higher cutoff for fidelity check.
        genotype_name: Name of genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.
        fidelity_threshold: Minimum fidelity to be considered valid.
        max_iterations: Maximum cleaning iterations.

    Returns:
        Tuple of (cleaned_repertoire, total_removed).
    """
    total_removed = 0

    for iteration in range(max_iterations):
        print(f"Final validation iteration {iteration + 1}/{max_iterations}...")

        repertoire, num_removed = validate_and_clean_archive(
            repertoire,
            base_cutoff,
            correction_cutoff,
            genotype_name,
            genotype_config,
            pnr_max,
            fidelity_threshold,
        )

        total_removed += num_removed

        if num_removed == 0:
            print(f"  No more artifacts found. Archive is clean.")
            break

        print(f"  Removed {num_removed} artifacts this iteration.")

    print(f"Final validation complete. Total removed: {total_removed}")
    return repertoire, total_removed


def get_archive_valid_genotypes(repertoire) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all valid genotypes from the archive.

    Returns:
        Tuple of (genotypes, indices) for valid solutions.
    """
    genotypes = np.array(repertoire.genotypes)
    fitnesses = np.array(repertoire.fitnesses)

    # Find non-empty cells
    valid_mask = np.all(np.isfinite(fitnesses), axis=-1)
    valid_indices = np.array(np.where(valid_mask)).T  # Shape: (N, ndim)

    valid_genotypes = genotypes[valid_mask]
    return valid_genotypes, valid_indices
