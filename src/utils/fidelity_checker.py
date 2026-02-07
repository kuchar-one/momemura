"""
Fidelity checker for dual-cutoff artifact validation.

Computes the fidelity between states simulated at different cutoffs to detect
numerical artifacts that arise from truncation errors.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, Any
import jax


def compute_fidelity(state1: jnp.ndarray, state2: jnp.ndarray) -> float:
    """
    Compute the fidelity between two pure states.

    Fidelity = |<psi1|psi2>|^2

    The states are truncated to the minimum dimension and renormalized
    before computing fidelity.

    Args:
        state1: First state vector.
        state2: Second state vector.

    Returns:
        Fidelity between 0 and 1.
    """
    # Truncate to minimum dimension
    min_dim = min(len(state1), len(state2))
    s1 = state1[:min_dim]
    s2 = state2[:min_dim]

    # Normalize
    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0  # One or both states are essentially zero

    s1 = s1 / norm1
    s2 = s2 / norm2

    # Compute fidelity
    overlap = np.abs(np.vdot(s1, s2)) ** 2
    return float(overlap)


def compute_state_at_cutoff(
    genotype: np.ndarray,
    cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute the final state for a genotype at a specific cutoff.

    Args:
        genotype: The genotype array.
        cutoff: Fock space dimension.
        genotype_name: Name of the genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.

    Returns:
        Tuple of (state_vector, joint_probability, total_pnr)
    """
    from src.genotypes.genotypes import get_genotype_decoder

    # Extract depth from config (default to 3 if not specified)
    depth = 3
    if genotype_config is not None and "depth" in genotype_config:
        depth = genotype_config["depth"]

    decoder = get_genotype_decoder(genotype_name, depth, genotype_config)
    params = decoder.decode(genotype, cutoff)

    # Get heralded leaf states
    from src.simulation.jax.runner import jax_get_heralded_state

    # Use JAX's dynamic complex dtype
    float_dtype = jnp.float64
    complex_dtype = jnp.complex128

    def conditional_herald(leaf_p, is_active):
        def compute(_):
            return jax_get_heralded_state(leaf_p, cutoff=cutoff, pnr_max=pnr_max)

        def skip(_):
            dummy_vec = jnp.zeros(cutoff, dtype=complex_dtype)
            return (
                dummy_vec,
                jnp.array(0.0, dtype=float_dtype),
                jnp.array(1.0, dtype=float_dtype),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=float_dtype),
            )

        return jax.lax.cond(is_active, compute, skip, None)

    leaf_results = jax.vmap(conditional_herald)(
        params["leaf_params"], params["leaf_active"]
    )

    leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes = leaf_results

    # Run superblock
    from src.simulation.jax.composer import jax_superblock

    # Handle homodyne parameters
    hom_x = params.get("homodyne_x", jnp.array(0.0))
    hom_win = params.get("homodyne_window", None)

    # Build phi_vec
    n_mix = params["mix_params"].shape[0]
    phi_vec = jnp.zeros((n_mix, cutoff))

    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    (
        final_state,
        _,
        joint_prob,
        _,
        _,
        total_sum_pnr,
        _,
    ) = jax_superblock(
        leaf_vecs,
        leaf_probs,
        params["leaf_active"],
        leaf_max_pnrs,
        leaf_total_pnrs,
        leaf_modes,
        params["mix_params"],
        hom_x,
        hom_win,
        hom_win,
        phi_vec,
        V_matrix,
        dx_weights,
        cutoff,
        True,
        False,
        False,
    )

    # Apply final Gaussian
    from src.simulation.jax.runner import jax_apply_final_gaussian

    final_state_transformed = jax_apply_final_gaussian(
        final_state, params["final_gauss"], cutoff
    )

    # Normalize
    norm = jnp.linalg.norm(final_state_transformed)
    if norm > 1e-12:
        final_state_normalized = final_state_transformed / norm
    else:
        final_state_normalized = final_state_transformed

    return (
        np.array(final_state_normalized),
        float(joint_prob),
        float(total_sum_pnr),
    )


def compute_fidelity_two_cutoffs(
    genotype: np.ndarray,
    base_cutoff: int,
    correction_cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
) -> float:
    """
    Compute the fidelity between states simulated at two different cutoffs.

    This is the key function for artifact detection. If a solution has high
    fidelity (>0.9), it's likely valid. If low fidelity, it's an artifact.

    Args:
        genotype: The genotype array.
        base_cutoff: Lower cutoff (e.g., 30).
        correction_cutoff: Higher cutoff (e.g., 45).
        genotype_name: Name of the genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.

    Returns:
        Fidelity between 0 and 1.
    """
    state_base, _, _ = compute_state_at_cutoff(
        genotype, base_cutoff, genotype_name, genotype_config, pnr_max
    )

    state_correction, _, _ = compute_state_at_cutoff(
        genotype, correction_cutoff, genotype_name, genotype_config, pnr_max
    )

    return compute_fidelity(state_base, state_correction)


def validate_genotype(
    genotype: np.ndarray,
    base_cutoff: int,
    correction_cutoff: int,
    genotype_name: str,
    genotype_config: Dict = None,
    pnr_max: int = 3,
    fidelity_threshold: float = 0.9,
) -> Tuple[bool, float]:
    """
    Validate a genotype by checking dual-cutoff fidelity.

    Args:
        genotype: The genotype array.
        base_cutoff: Lower cutoff.
        correction_cutoff: Higher cutoff.
        genotype_name: Name of genotype type.
        genotype_config: Optional genotype configuration.
        pnr_max: Maximum PNR value.
        fidelity_threshold: Minimum fidelity to be considered valid.

    Returns:
        Tuple of (is_valid, fidelity).
    """
    fidelity = compute_fidelity_two_cutoffs(
        genotype,
        base_cutoff,
        correction_cutoff,
        genotype_name,
        genotype_config,
        pnr_max,
    )
    return (fidelity >= fidelity_threshold, fidelity)
