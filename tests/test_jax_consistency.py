import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.circuits.composer import Composer
from src.circuits.jax_composer import jax_superblock, jax_hermite_phi_matrix
from src.circuits.jax_runner import jax_decode_genotype, jax_get_heralded_state


def test_jax_superblock_consistency():
    """
    Verifies that jax_superblock produces the same result as Composer
    for a specific tree configuration.
    """
    jax.config.update("jax_enable_x64", True)
    cutoff = 10
    hbar = 2.0

    # 1. Setup Composer
    composer = Composer(cutoff=cutoff)

    # 2. Define a simple tree (Depth 3)
    # Leaves: 8 blocks
    # We use random params for blocks

    # Create random genotype
    # We need to ensure it's valid for jax_decode_genotype
    # We can just create a random array and decode it
    key = jax.random.PRNGKey(42)
    g = jax.random.normal(key, (300,))  # Enough for 256 params

    params = jax_decode_genotype(g, cutoff)

    # Extract params
    leaf_params = params["leaf_params"]
    mix_params = params["mix_params"]  # (7, 3) angles
    mix_source = params["mix_source"]  # (7,) source
    hom_x = float(params["homodyne_x"])
    hom_win = 0.0  # Force Point Homodyne for optimization test

    # 3. Build CPU Circuit
    # We need to map JAX params to CPU Composer calls
    # This is tricky because Composer expects specific params.
    # But we can use the leaf states directly!

    # Get leaf states from JAX
    leaf_vecs, leaf_probs, leaf_modes, leaf_pnrs, _ = jax.vmap(
        lambda p: jax_get_heralded_state(p, cutoff)
    )(leaf_params)  # Wait, jax_get_heralded_state takes params dict now?
    # Yes, I updated it to take params dict.
    # But jax_decode_genotype returns "leaf_params" as a dict of arrays.
    # So I need to slice it?
    # No, jax.vmap handles dict of arrays automatically.

    # However, jax_get_heralded_state signature is `(params, cutoff)`.
    # So `vmap(partial(jax_get_heralded_state, cutoff=cutoff))(leaf_params)` works.

    leaf_vecs_np = np.array(leaf_vecs)
    leaf_probs_np = np.array(leaf_probs)

    print(f"Leaf Probs: {leaf_probs_np}")
    print(f"Mix Source: {mix_source}")
    print(f"Mix Params: {mix_params}")

    # Build tree manually with Composer
    # Layer 0: Leaves
    current_states = [leaf_vecs_np[i] for i in range(8)]
    current_probs = [leaf_probs_np[i] for i in range(8)]

    mix_idx = 0

    # Helper to mix on CPU
    def mix_cpu(stateA, stateB, probA, probB, params, source):
        theta, phi, varphi = params
        # Source: 0=Mix, 1=Left, 2=Right

        if source == 1:  # Left
            return stateA, probA
        elif source == 2:  # Right
            return stateB, probB
        else:  # Mix
            # Composer.compose_pair
            # We need U_bs
            # Composer uses `_run_two_mode_program_and_get_full_dm` (now `compose_pair`)
            # It takes U (unitary).
            # We need to construct U from theta, phi, varphi.
            # Same logic as JAX: U = Phase(varphi) @ BS(theta, phi)

            # Homodyne
            # We use Point homodyne (window=0 effectively, or explicit x)
            # Composer.compose_pair(stateA, stateB, U, pA, pB, homodyne_x=..., homodyne_window=...)

            rho, p_meas, joint = composer.compose_pair(
                stateA,
                stateB,
                probA,
                probB,
                homodyne_x=hom_x,
                homodyne_window=hom_win,
                homodyne_resolution=0.01,
                theta=float(theta),
                phi=float(phi),
            )
            return rho, joint

    # Layer 1
    next_states = []
    next_probs = []
    for i in range(4):
        sA, sB = current_states[2 * i], current_states[2 * i + 1]
        pA, pB = current_probs[2 * i], current_probs[2 * i + 1]
        p = mix_params[mix_idx]
        src = mix_source[mix_idx]
        mix_idx += 1

        s, pr = mix_cpu(sA, sB, pA, pB, p, src)
        next_states.append(s)
        next_probs.append(pr)

    current_states = next_states
    current_probs = next_probs

    # Layer 2
    next_states = []
    next_probs = []
    for i in range(2):
        sA, sB = current_states[2 * i], current_states[2 * i + 1]
        pA, pB = current_probs[2 * i], current_probs[2 * i + 1]
        p = mix_params[mix_idx]
        src = mix_source[mix_idx]
        mix_idx += 1

        s, pr = mix_cpu(sA, sB, pA, pB, p, src)
        next_states.append(s)
        next_probs.append(pr)

    current_states = next_states
    current_probs = next_probs

    # Layer 3 (Root)
    sA, sB = current_states[0], current_states[1]
    pA, pB = current_probs[0], current_probs[1]
    p = mix_params[mix_idx]
    src = mix_source[mix_idx]

    root_state_cpu, root_prob_cpu = mix_cpu(sA, sB, pA, pB, p, src)

    # 4. Run JAX Superblock
    # Prepare inputs
    phi_mat = jax_hermite_phi_matrix(jnp.array([hom_x]), cutoff)
    phi_vec = phi_mat[:, 0]
    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    final_state_jax, _, joint_prob_jax, _, _, _ = jax_superblock(
        leaf_vecs,
        leaf_probs,
        params["leaf_active"],
        leaf_pnrs,
        leaf_modes,
        mix_params,
        mix_source,
        hom_x,
        hom_win,
        0.0,
        phi_vec,
        V_matrix,
        dx_weights,
        cutoff,
        True,  # window is none (Point Homodyne)
        False,
        True,
    )

    # 5. Compare
    # Check probabilities
    print(f"CPU Prob: {root_prob_cpu}")
    print(f"JAX Prob: {joint_prob_jax}")
    assert np.isclose(root_prob_cpu, joint_prob_jax, atol=1e-5)

    # Check states
    # JAX state might be complex128, CPU complex128
    # Compare density matrices
    # CPU state might be vector if pure?
    # compose_pair returns rho (matrix).

    # Compare density matrices
    if final_state_jax.ndim == 1:
        final_state_jax = np.outer(final_state_jax, final_state_jax.conj())
    if root_state_cpu.ndim == 1:
        root_state_cpu = np.outer(root_state_cpu, root_state_cpu.conj())

    diff = np.abs(root_state_cpu - final_state_jax)
    print(f"Max State Diff: {np.max(diff)}")
    assert np.allclose(root_state_cpu, final_state_jax, atol=1e-5)


if __name__ == "__main__":
    test_jax_superblock_consistency()
