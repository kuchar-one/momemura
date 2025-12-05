import jax
import jax.numpy as jnp
import numpy as np
from src.circuits.jax_composer import jax_u_bs
from src.utils.gkp_operator import construct_gkp_operator


def check_bs():
    print("Checking BS Unitary (C=25)...")
    theta = 0.1
    phi = 0.2
    cutoff = 25

    try:
        u = jax_u_bs(theta, phi, cutoff)
        print(f"U shape: {u.shape}")
        print(f"Contains NaNs: {jnp.isnan(u).any()}")
        print(f"Contains Infs: {jnp.isinf(u).any()}")

        # Check unitarity
        u_dag = u.conj().T
        identity = jnp.eye(cutoff**2)
        prod = u @ u_dag
        diff = jnp.linalg.norm(prod - identity)
        print(f"Unitarity error (norm(U U^dag - I)): {diff:.2e}")

        if diff > 1e-5:
            print("UNITARITY CHECK FAILED")
        else:
            print("Unitarity check passed")

        # Check gradients
        print("Checking gradients...")

        def loss(theta, phi):
            U = jax_u_bs(theta, phi, cutoff)
            return jnp.real(jnp.trace(U))

        grad_fn = jax.value_and_grad(loss, argnums=(0, 1))
        val, (g_theta, g_phi) = grad_fn(theta, phi)

        print(f"Loss: {val}")
        print(f"Grad Theta: {g_theta}")
        print(f"Grad Phi: {g_phi}")

        if jnp.isnan(g_theta) or jnp.isnan(g_phi):
            print("GRADIENT CHECK FAILED: NaNs in gradients")
        else:
            print("Gradient check passed")

    except Exception as e:
        print(f"BS Check failed with error: {e}")


def check_gkp():
    print("\nChecking GKP Operator (alpha=1.0)...")
    cutoff = 25
    target_alpha = 1.0
    target_beta = 0.0

    try:
        op = construct_gkp_operator(cutoff, target_alpha, target_beta, backend="jax")
        print(f"Op shape: {op.shape}")
        print(f"Contains NaNs: {jnp.isnan(op).any()}")
        print(f"Contains Infs: {jnp.isinf(op).any()}")

        # Check trace (should be close to 1 for normalized state, but this is an operator?)
        # GKP operator is a projector? Or density matrix?
        # It returns a ket or density matrix?
        # construct_gkp_operator returns the truncated operator.
        # If it's a state, trace(rho) should be 1.

        # Wait, construct_gkp_operator returns "The truncated operator as a matrix."
        # Is it a density matrix?
        # It calls high_dim_magic_generator_paper which returns cx*X + cy*Y + cz*Z + U.
        # This is a Hamiltonian or observable?
        # Ah, "GKP magic operator".

        print(f"Trace: {jnp.trace(op)}")

    except Exception as e:
        print(f"GKP Check failed with error: {e}")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    check_bs()
    check_gkp()
