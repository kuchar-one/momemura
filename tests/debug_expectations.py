import jax.numpy as jnp
from src.utils.gkp_operator import construct_gkp_operator
from src.simulation.jax.herald import vacuum_covariance, complex_alpha_to_qp
import numpy as np


def debug_expectations():
    cutoff = 25
    target_alpha = 1.0 + 0j
    target_beta = 1.0 + 0j  # As per user args? No, typical GKP is alpha=2?
    # User args: target-alpha 1.0, target-beta 1.0 (Wait, alpha=sqrt(pi) usually? Or 2?)
    # User command: --target-alpha 1.0 --target-beta 1.0

    # Note: run_mome.py defaults target_alpha=2.0. But user passed 1.0.

    op = construct_gkp_operator(cutoff, target_alpha, target_beta, backend="jax")

    # Check Vacuum |0>
    # In Fock basis, Vacuum is [1, 0, 0, ...]
    vac = np.zeros(cutoff, dtype=complex)
    vac[0] = 1.0
    vac_j = jnp.array(vac)

    exp_vac = jnp.real(jnp.vdot(vac_j, op @ vac_j))
    print(f"Expectation(Vacuum): {exp_vac:.6f}")

    # Check Coherent State |alpha> ?
    # Or Squeezed Vacuum?

    # Check what 1.0656 might be.

    # Check Ground State of Operator (Diagonalization)
    evals, evecs = jnp.linalg.eigh(op)
    print(f"Ground State Eig: {evals[0]:.6f}")  # Should match 0.393251 log

    # Check User's Best State (Coherent)
    # r=0, disp=-2.24j
    from src.simulation.jax.runner import jax_apply_final_gaussian

    # Create simple state
    # Vacuum
    psi_vac = jnp.zeros(cutoff, dtype=complex).at[0].set(1.0)

    # Coherent State (Best Found)
    disp_val = -2.2448j
    params_coh = {"r": 0.0, "phi": 0.0, "varphi": 0.0, "disp": disp_val}
    psi_coh = jax_apply_final_gaussian(psi_vac, params_coh, cutoff)

    exp_coh = jnp.real(jnp.vdot(psi_coh, op @ psi_coh))
    print(f"Expectation(Coherent r=0, d={disp_val}): {exp_coh:.6f}")

    # Squeezed State (r=0.5)
    params_sq = {"r": 0.5, "phi": 0.0, "varphi": 0.0, "disp": disp_val}
    psi_sq = jax_apply_final_gaussian(psi_vac, params_sq, cutoff)
    exp_sq = jnp.real(jnp.vdot(psi_sq, op @ psi_sq))
    print(f"Expectation(Squeezed r=0.5, d={disp_val}): {exp_sq:.6f}")

    # Squeezed State (r=0.1)
    params_sq_small = {"r": 0.1, "phi": 0.0, "varphi": 0.0, "disp": disp_val}
    psi_sq_small = jax_apply_final_gaussian(psi_vac, params_sq_small, cutoff)
    exp_sq_small = jnp.real(jnp.vdot(psi_sq_small, op @ psi_sq_small))
    print(f"Expectation(Squeezed r=0.1, d={disp_val}): {exp_sq_small:.6f}")

    # Squeezed State (r=1.0)
    params_sq_large = {"r": 1.0, "phi": 0.0, "varphi": 0.0, "disp": disp_val}
    psi_sq_large = jax_apply_final_gaussian(psi_vac, params_sq_large, cutoff)
    exp_sq_large = jnp.real(jnp.vdot(psi_sq_large, op @ psi_sq_large))
    print(f"Expectation(Squeezed r=1.0, d={disp_val}): {exp_sq_large:.6f}")


if __name__ == "__main__":
    debug_expectations()
