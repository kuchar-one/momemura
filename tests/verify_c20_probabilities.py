import sys
import os
import numpy as np
import jax
import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.genotypes.genotypes import get_genotype_decoder  # noqa: E402
from src.simulation.jax.runner import jax_get_heralded_state  # noqa: E402
from frontend.utils import compute_state_with_jax  # noqa: E402
from src.simulation.jax.composer import jax_superblock  # noqa: E402


def verify_c20_probabilities():
    print("\n--- Verifying C20 Probabilities & Frontend Match ---")

    depth = 3
    cutoff = 20
    name = "C20"

    decoder = get_genotype_decoder(name, depth=depth)
    expected_len = decoder.get_length(depth)
    print(f"Genotype: {name}, Length: {expected_len}")

    # Create random genotype
    key = jax.random.PRNGKey(42)
    # Use bounds to create realistic values
    g = jax.random.uniform(key, (expected_len,), minval=-1.0, maxval=1.0)

    params = decoder.decode(g, cutoff)

    # Force window measurement
    # C20 has shared mix params and unique homodyne_x.
    # Where does homodyne_window come from?
    # It's usually in decoder config or fixed?
    # In C1/C2 decode:
    #   homodyne_window = self.window (default 0.1)

    print(f"Decoded Window: {params.get('homodyne_window')}")
    params["homodyne_window"] = 0.1  # Explicitly set for test
    params["homodyne_resolution"] = 0.1  # This is what frontend expects?
    # Frontend logic:
    # hom_win_val = params.get("homodyne_window") (usually resolution value)
    # Backend logic:
    # homodyne_window param IS the resolution/width.

    # 1. Backend Evaluation (Simulating run_mome scoring)
    # We call jax_get_heralded_state manually or via runner helpers?
    # Let's use logic similar to run_mome.py evaluate_one (which we updated to use jax_scoring_fn_batch logic basically)

    print("Computing Backend Result...")
    leaf_params = params["leaf_params"]
    # vmap get_heralded
    # C20 has shared PNR -> leaves have it properly?
    # Check leaf_params['pnr'] shape
    pnr_shape = leaf_params["pnr"].shape
    print(f"Leaf PNR Shape (Backend check): {pnr_shape}")

    pnr_max_val = 3
    get_heralded = jax.vmap(lambda p: jax_get_heralded_state(p, cutoff, pnr_max_val))
    (leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes) = (
        get_heralded(leaf_params)
    )

    # Call superblock manually
    # op_matrix = jnp.eye(cutoff)  # Dummy op (Unused)

    # Point Homodyne Logic
    hom_x_val = params["homodyne_x"]
    hom_win_val = params["homodyne_window"]  # Resolution

    # Point Mode: V_matrix unused
    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    mix_params = params["mix_params"]

    # Prepare phi_vec per node for Point Mode
    hom_xs = jnp.atleast_1d(hom_x_val)
    from frontend.utils import jax_hermite_phi_matrix

    phi_mat = jax_hermite_phi_matrix(hom_xs, cutoff)
    if jnp.ndim(hom_x_val) == 0:
        phi_vec = phi_mat[:, 0]
    else:
        phi_vec = phi_mat.T

    # Backend Result
    res_state, _, joint_prob, _, _, _, _ = jax_superblock(
        leaf_vecs,
        leaf_probs,
        params["leaf_active"],
        leaf_max_pnrs,
        leaf_total_pnrs,
        leaf_modes,
        mix_params,
        hom_x_val,
        0.0,  # homodyne_window ignored in Point Mode
        hom_win_val,  # homodyne_resolution
        phi_vec,
        V_matrix,  # Ignored
        dx_weights,  # Ignored
        cutoff,
        homodyne_window_is_none=True,  # Force Point Mode
        homodyne_x_is_none=False,
        homodyne_resolution_is_none=False,  # Apply Scaling
    )

    prob_backend = float(joint_prob)
    print(f"Backend Probability: {prob_backend} (Log: {np.log10(prob_backend)})")

    if prob_backend > 1.0:
        print("FAILED: Backend Probability > 1.0")
        return False

    # 2. Frontend Evaluation
    print("Computing Frontend Result...")
    # Inject cache to ensure matches if frontend uses it (it shouldn't matter for Point)
    # params['nodes_cache'] = ... (Not used in Point Mode)

    frontend_state, frontend_prob = compute_state_with_jax(params, cutoff)
    print(f"Frontend Probability: {frontend_prob}")

    if abs(prob_backend - frontend_prob) > 1e-6:
        print(f"FAILED: Mismatch! Backend={prob_backend}, Frontend={frontend_prob}")
        return False

    print("SUCCESS: Frontend matches Backend and Probability <= 1.0")
    return True


if __name__ == "__main__":
    success = verify_c20_probabilities()
    if not success:
        sys.exit(1)
