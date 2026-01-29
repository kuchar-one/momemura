import numpy as np
import jax
import jax.numpy as jnp
import os
import sys
import qutip as qt

sys.path.append(os.getcwd())
from frontend.utils import compute_state_with_jax


def check_gaussianity(rho_or_psi, name="State"):
    if rho_or_psi.ndim == 1:
        rho = qt.ket2dm(qt.Qobj(np.array(rho_or_psi)))
    else:
        rho = qt.Qobj(np.array(rho_or_psi))

    xvec = np.linspace(-5, 5, 100)
    W = qt.wigner(rho, xvec, xvec)
    min_val = np.min(W)

    print(f"[{name}] Min Wigner: {min_val:.5e}")
    return min_val


def run_repro():
    print("--- Starting Param Isolation ---")
    cutoff = 40

    # Base Setup
    leaf_active = np.zeros(8, dtype=bool)
    leaf_active[4] = True

    def get_params(r_val, disp_val):
        leaf_params = {
            "r": np.zeros((8, 3), dtype=np.float32),
            "phases": np.zeros((8, 9), dtype=np.float32),
            "disp": np.zeros((8, 3), dtype=np.complex64),
            "n_ctrl": np.zeros(8, dtype=np.int32),
            "pnr": np.zeros((8, 2), dtype=np.int32),
            "pnr_max": np.full((8,), 3, dtype=np.int32),
        }
        leaf_params["r"][4] = r_val
        leaf_params["disp"][4] = disp_val
        leaf_params["n_ctrl"][4] = 2

        return {
            "leaf_params": leaf_params,
            "leaf_active": leaf_active,
            "mix_params": np.zeros((7, 3)),
            "homodyne_x": 0.0,
            "homodyne_window": 0.1,
        }

    # 1. Vacuum (Reference)
    print("\n1. Vacuum (r=0, disp=0)")
    p1 = get_params([0, 0, 0], [0, 0, 0])
    psi1, _ = compute_state_with_jax(p1, cutoff=cutoff)
    check_gaussianity(psi1, "Vacuum")

    # 2. Squeezing Only (r=-0.18)
    print("\n2. Squeezing Only (r=[0, -0.18, 0])")
    p2 = get_params([0.0, -0.18, 0.0], [0, 0, 0])
    psi2, _ = compute_state_with_jax(p2, cutoff=cutoff)
    check_gaussianity(psi2, "Squeezed")

    # 3. Displacement Only (disp=-2.3-4j)
    print("\n3. Displacement Only (disp=[-2.3-4j])")
    p3 = get_params([0, 0, 0], [-2.375 - 4.157j, 0, 0])
    psi3, _ = compute_state_with_jax(p3, cutoff=cutoff)
    check_gaussianity(psi3, "Displaced")

    # 4. Full (User Case)
    print("\n4. Full (Sqz + Disp)")
    p4 = get_params([0.0, -0.18, 0.0], [-2.375 - 4.157j, -0.002 - 0.001j, 0])
    psi4, _ = compute_state_with_jax(p4, cutoff=cutoff)
    check_gaussianity(psi4, "Full")


if __name__ == "__main__":
    run_repro()
