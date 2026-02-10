"""
Archive validator for dual-cutoff artifact validation.

Uses batched GPU scoring (jax_scoring_fn_batch) for high throughput.
Validates solutions by comparing states at base vs correction cutoff.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, Any


def batch_compute_fidelities(
    genotypes: jnp.ndarray,
    base_cutoff: int,
    correction_cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
) -> np.ndarray:
    """
    Compute fidelity for a batch of genotypes using GPU-batched scoring.

    Runs jax_scoring_fn_batch twice (base + correction cutoff) and computes
    fidelity between the resulting states vectorized.

    Args:
        genotypes: Batch of genotypes, shape (N, genome_dim).
        base_cutoff: Lower cutoff (e.g., 30).
        correction_cutoff: Higher cutoff (e.g., 45).
        genotype_name: Name of genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.

    Returns:
        Array of fidelities, shape (N,).
    """
    from src.simulation.jax.runner import jax_scoring_fn_batch

    # Dummy operator (not used for fidelity, but required by scoring fn)
    operator = jnp.eye(base_cutoff, dtype=jnp.complex128)

    # Run 1: Score at base_cutoff only (no correction) → state at base cutoff
    _, _, extras_base = jax_scoring_fn_batch(
        genotypes,
        base_cutoff,
        operator,
        genotype_name=genotype_name,
        genotype_config=genotype_config,
        correction_cutoff=None,  # No correction → state at base_cutoff
        pnr_max=pnr_max,
    )
    states_base = extras_base["final_state"]  # (N, base_cutoff)

    # Run 2: Score at base_cutoff WITH correction → state truncated from correction_cutoff
    _, _, extras_corr = jax_scoring_fn_batch(
        genotypes,
        base_cutoff,
        operator,
        genotype_name=genotype_name,
        genotype_config=genotype_config,
        correction_cutoff=correction_cutoff,  # Correction → state at correction then truncated
        pnr_max=pnr_max,
    )
    states_corr = extras_corr["final_state"]  # (N, base_cutoff)

    # Vectorized fidelity: |<s1|s2>|^2 for each genotype
    # Both states are already normalized in runner.py (state_eval)
    # Re-normalize for safety
    norms_base = jnp.linalg.norm(states_base, axis=1, keepdims=True)
    norms_corr = jnp.linalg.norm(states_corr, axis=1, keepdims=True)

    # Handle zero-norm states (prob ~ 0)
    safe_norms_base = jnp.maximum(norms_base, 1e-12)
    safe_norms_corr = jnp.maximum(norms_corr, 1e-12)

    states_base_n = states_base / safe_norms_base
    states_corr_n = states_corr / safe_norms_corr

    # Fidelity = |<psi1|psi2>|^2
    overlaps = jnp.sum(jnp.conj(states_base_n) * states_corr_n, axis=1)
    fidelities = jnp.abs(overlaps) ** 2

    # Zero out fidelity for zero-norm states
    valid = (norms_base.squeeze() > 1e-10) & (norms_corr.squeeze() > 1e-10)
    fidelities = jnp.where(valid, fidelities, 0.0)

    return np.array(fidelities)


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

    Uses batched GPU scoring for high throughput.

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
    genotypes = np.array(repertoire.genotypes)
    fitnesses = np.array(repertoire.fitnesses)

    # Find non-empty cells (valid solutions)
    valid_mask = np.all(np.isfinite(fitnesses), axis=-1)
    valid_indices = np.where(valid_mask)

    if len(valid_indices[0]) == 0:
        return repertoire, 0

    # Collect valid genotypes into a batch
    valid_genotypes = genotypes[valid_mask]  # (N, genome_dim)

    # Batch fidelity computation on GPU
    fidelities = batch_compute_fidelities(
        jnp.array(valid_genotypes),
        base_cutoff,
        correction_cutoff,
        genotype_name,
        genotype_config,
        pnr_max,
    )

    # Identify artifacts (fidelity below threshold)
    artifact_mask = fidelities < fidelity_threshold
    num_removed = int(np.sum(artifact_mask))

    if num_removed > 0:
        # Map back to full archive indices
        removal_mask = np.zeros_like(valid_mask, dtype=bool)
        valid_flat_indices = list(zip(*valid_indices))
        for i, idx in enumerate(valid_flat_indices):
            if artifact_mask[i]:
                removal_mask[idx] = True
                print(f"  Removing artifact at {idx}: fidelity={fidelities[i]:.3f}")

        # Invalidate removed solutions
        new_fitnesses = fitnesses.copy()
        new_fitnesses[removal_mask] = -np.inf

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
            print("  No more artifacts found. Archive is clean.")
            break

        print(f"  Removed {num_removed} artifacts this iteration.")

    print(f"Final validation complete. Total removed: {total_removed}")
    return repertoire, total_removed


def batch_validate_genotypes(
    genotypes: jnp.ndarray,
    base_cutoff: int,
    correction_cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
    fidelity_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate a batch of genotypes and return validity mask + fidelities.

    Designed for mid-run checks in single-objective mode.

    Args:
        genotypes: Batch of genotypes, shape (N, genome_dim).
        base_cutoff: Lower cutoff.
        correction_cutoff: Higher cutoff.
        genotype_name: Name of genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.
        fidelity_threshold: Minimum fidelity threshold.

    Returns:
        Tuple of (valid_mask, fidelities) both shape (N,).
    """
    fidelities = batch_compute_fidelities(
        genotypes,
        base_cutoff,
        correction_cutoff,
        genotype_name,
        genotype_config,
        pnr_max,
    )

    valid_mask = fidelities >= fidelity_threshold
    return valid_mask, fidelities
