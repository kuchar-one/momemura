import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
from src.simulation.cpu.composer import Composer
from src.simulation.jax.composer import (
    jax_superblock,
    jax_hermite_phi_matrix,
)
from src.simulation.jax.runner import jax_get_heralded_state
from src.genotypes.genotypes import get_genotype_decoder


def test_jax_superblock_consistency():
    """
    Verifies that jax_superblock produces the same result as Composer
    for a specific tree configuration.
    """
    jax.config.update("jax_enable_x64", True)
    cutoff = 10

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

    # params = jax_decode_genotype(g, cutoff)
    params = get_genotype_decoder("A").decode(g, cutoff)

    # Extract params
    leaf_params = params["leaf_params"]
    # extract mix params
    mix_params = params["mix_params"]  # (7, 3) angles
    hom_x = float(params["homodyne_x"])
    hom_win = 0.0  # Force Point Homodyne for optimization test

    # FORCE LEAF ACTIVE to ensure we test mixing physics
    params["leaf_active"] = jnp.ones(8, dtype=bool)
    leaf_active = params["leaf_active"]

    # 3. Build CPU Circuit
    # We need to map JAX params to CPU Composer calls
    # This is tricky because Composer expects specific params.
    # But we can use the leaf states directly!

    # Get leaf states from JAX
    leaf_vecs, leaf_probs, _, leaf_pnrs, leaf_total_pnrs, leaf_modes = jax.vmap(
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
    # Mix Source trace
    print(f"Mix Params: {mix_params}")

    # Build tree manually with Composer
    # Layer 0: Leaves
    current_states = [leaf_vecs_np[i] for i in range(8)]
    current_probs = [leaf_probs_np[i] for i in range(8)]

    mix_idx = 0

    # Helper to mix on CPU
    def mix_cpu(stateA, stateB, probA, probB, params, activeA, activeB):
        # Implicit Logic:
        # A & B -> Mix
        # A & ~B -> Pass A
        # ~A & B -> Pass B
        # ~A & ~B -> Inactive (Pass A default or handle specially?)

        # For this test, we assume active flags are passed correctly.
        # But wait, the test doesn't explicitly toggle active flags per node?
        # The test uses `leaf_params["leaf_active"]`.
        # We need to trace activity to implement `mix_cpu` correctly if we want to match JAX.

        # However, the JAX logic INSIDE `mix_node` does the switching.
        # If we want parity, we must implement the switch here.

        # But `mix_cpu` is called layers deep. We need accurate `active` status for inputs.
        # This test might be simplified to assume ALL active for consistency if we want to test mixing numerics?
        # Or we implement the full logic.

        # Let's check `jax_consistency.py` context. It tests `jax_superblock`.
        # `jax_superblock` uses `mix_node`.
        # So `mix_cpu` MUST match `mix_node`.

        passedA = activeA and (not activeB)
        passedB = activeB and (not activeA)
        mixed = activeA and activeB

        if passedA:
            return stateA, probA
        elif passedB:
            return stateB, probB
        elif mixed:
            theta, phi, varphi = params
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
        else:
            # Neither active. Return vacuum/dummy?
            # In JAX it returns do_left (stateA) but flag is inactive.
            # Here we just return stateA for structure.
            return stateA, probA

    # Track activity
    leaf_active = params["leaf_active"]  # Bool array (8,)
    # We need to propagate this.
    current_active = [bool(leaf_active[i]) for i in range(8)]

    # Layer 1
    next_states = []
    next_probs = []
    next_active = []
    for i in range(4):
        sA, sB = current_states[2 * i], current_states[2 * i + 1]
        pA, pB = current_probs[2 * i], current_probs[2 * i + 1]
        actA, actB = current_active[2 * i], current_active[2 * i + 1]
        p = mix_params[mix_idx]
        # src removed
        mix_idx += 1

        s, pr = mix_cpu(sA, sB, pA, pB, p, actA, actB)
        next_states.append(s)
        next_probs.append(pr)
        next_active.append(actA or actB)

    current_states = next_states
    current_probs = next_probs
    current_active = next_active

    # Layer 2
    next_states = []
    next_probs = []
    next_active = []
    for i in range(2):
        sA, sB = current_states[2 * i], current_states[2 * i + 1]
        pA, pB = current_probs[2 * i], current_probs[2 * i + 1]
        actA, actB = current_active[2 * i], current_active[2 * i + 1]
        p = mix_params[mix_idx]
        mix_idx += 1

        s, pr = mix_cpu(sA, sB, pA, pB, p, actA, actB)
        next_states.append(s)
        next_probs.append(pr)
        next_active.append(actA or actB)

    current_states = next_states
    current_probs = next_probs
    current_active = next_active

    # Layer 3 (Root)
    sA, sB = current_states[0], current_states[1]
    pA, pB = current_probs[0], current_probs[1]
    actA, actB = current_active[0], current_active[1]
    p = mix_params[mix_idx]

    root_state_cpu, root_prob_cpu = mix_cpu(sA, sB, pA, pB, p, actA, actB)

    # 4. Run JAX Superblock
    # Prepare inputs
    phi_mat = jax_hermite_phi_matrix(jnp.array([hom_x]), cutoff)
    phi_vec = phi_mat[:, 0]
    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    final_state_jax, _, joint_prob_jax, _, _, _, _ = jax_superblock(
        leaf_vecs,
        leaf_probs,
        params["leaf_active"],
        leaf_pnrs,
        leaf_pnrs,  # Using leaf_pnrs as total_pnrs (since 1 mode per leaf in this test)
        leaf_modes,
        mix_params,
        # mix_source removed
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

    # Compare density matrices if prob is significant
    if root_prob_cpu > 1e-9:
        diff = np.abs(root_state_cpu - final_state_jax)
        print(f"Max State Diff: {np.max(diff)}")
        assert np.allclose(root_state_cpu, final_state_jax, atol=1e-5)
    else:
        print("Skipping state comparison due to negligible probability.")


if __name__ == "__main__":
    test_jax_superblock_consistency()
