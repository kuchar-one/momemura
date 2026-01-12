import jax
import jax.numpy as jnp
from src.simulation.jax.composer import (
    jax_compose_pair,
    jax_u_bs,
    jax_hermite_phi_matrix,
)


def test_homodyne_prob_exclusion():
    """
    Verify that the joint probability returned by jax_compose_pair
    EXCLUDES the homodyne measurement probability.
    It should be purely pA * pB.
    """
    jax.config.update("jax_enable_x64", True)
    cutoff = 10
    # Two fock states |1>
    # pA = 0.5, pB = 0.5 (arbitrary passed in probs)
    pA = 0.5
    pB = 0.5

    # State |1>
    vec = jnp.zeros(cutoff)
    vec = vec.at[1].set(1.0)

    # BS: 50:50
    theta = jnp.pi / 4
    phi = 0.0
    U = jax_u_bs(theta, phi, cutoff)

    # Homodyne Measurement that is NOT perfect ( prob < 1)
    # If we measure X=0 on |1>, prob should be very low (node at 0).
    hom_x = 0.0

    # Check shape of phi_vec
    phi_mat = jax_hermite_phi_matrix(jnp.array([hom_x]), cutoff)
    phi_vec = phi_mat[:, 0]

    # Run compose
    res_state, p_measure, joint = jax_compose_pair(
        vec,
        vec,
        U,
        pA,
        pB,
        homodyne_x=hom_x,
        homodyne_window=0.0,
        homodyne_resolution=0.0,
        phi_vec=phi_vec,
        V_matrix=None,
        dx_weights=None,
        cutoff=cutoff,
        homodyne_window_is_none=True,
        homodyne_x_is_none=False,
        homodyne_resolution_is_none=True,
        theta=theta,
        phi=phi,
    )

    # Expected Behavior:
    # joint == pA * pB == 0.25 (New Logic)
    # p_measure depends on physics (should be > 0 but < 1)

    print(f"p_measure: {p_measure}")
    print(f"joint: {joint}")

    # Note: p_measure for |1> at x=0 is 0.0 actually?
    # phi_1(0) = 0?
    # H_1(x) = 2x. At x=0, H_1(0)=0. So yes p_measure=0.
    # Let's use x=0.1 to get non-zero.

    if p_measure == 0.0:
        print("Switching to x=0.5 to ensure non-zero measurement prob")
        hom_x = 0.5
        phi_mat = jax_hermite_phi_matrix(jnp.array([hom_x]), cutoff)
        phi_vec = phi_mat[:, 0]
        res_state, p_measure, joint = jax_compose_pair(
            vec,
            vec,
            U,
            pA,
            pB,
            homodyne_x=hom_x,
            homodyne_window=0.0,
            homodyne_resolution=0.0,
            phi_vec=phi_vec,
            V_matrix=None,
            dx_weights=None,
            cutoff=cutoff,
            homodyne_window_is_none=True,
            homodyne_x_is_none=False,
            homodyne_resolution_is_none=True,
            theta=theta,
            phi=phi,
        )
        print(f"New p_measure: {p_measure}")
        print(f"New joint: {joint}")

    assert jnp.isclose(joint, pA * pB), f"Joint prob {joint} should be {pA * pB}"

    # Ensure p_measure is still physically meaningful (not 1.0, assuming not perfect)
    # For point measurement, typically p_measure is a density, can be anything.
    # But usually not exactly 1.0 (unless by coincidence).
    # Actually for point measurement, p_measure is probability density.
    # We just check it's not influencing joint.

    # Check normalization of state
    # If p_measure > 0, state should be normalized
    if p_measure > 0:
        norm = jnp.linalg.norm(res_state)
        assert jnp.isclose(norm, 1.0), f"State should be normalized, got {norm}"


if __name__ == "__main__":
    test_homodyne_prob_exclusion()
