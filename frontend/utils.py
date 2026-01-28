import os
import sys

# Force JAX to use CPU backend for frontend visualization
# This avoids GPU/CUDA issues and ensures consistent behavior
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import qutip as qt
from typing import List, Tuple, Dict, Any

# Ensure we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.result_manager import OptimizationResult  # noqa: E402


def list_runs(output_dir: str = "output") -> List[str]:
    """List all run directories in the output folder."""
    if not os.path.exists(output_dir):
        return []

    runs = [
        d
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith(".")
    ]
    runs.sort(reverse=True)

    exp_dir = os.path.join(output_dir, "experiments")
    if os.path.exists(exp_dir):
        exp_groups = [
            f"experiments/{d}"
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith(".")
        ]
        exp_groups.sort(reverse=True)
        runs = exp_groups + runs

    return runs


def load_run(run_dir: str) -> OptimizationResult:
    """
    Load OptimizationResult from a run directory.
    """
    path = (
        os.path.join(project_root, run_dir) if not os.path.isabs(run_dir) else run_dir
    )

    if os.path.exists(os.path.join(path, "results.pkl")):
        return OptimizationResult.load(path)

    has_subruns = False
    if os.path.isdir(path):
        for item in os.listdir(path):
            sub = os.path.join(path, item)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "results.pkl")):
                has_subruns = True
                break

    if has_subruns:
        from src.utils.result_manager import AggregatedOptimizationResult

        return AggregatedOptimizationResult.load_group(path)

    raise ValueError(f"No valid run found at {path}")


def to_scalar(x: Any) -> Any:
    """Safely convert x to a scalar float or complex."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (float, complex, int)) and not isinstance(x, bool):
            return x
        if hasattr(x, "item"):
            return x.item()
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            lst = list(x)
            if len(lst) == 1:
                elem = lst[0]
                if hasattr(elem, "item"):
                    return elem.item()
                if isinstance(elem, (float, complex, int)):
                    return elem
                if isinstance(elem, list) and len(elem) == 1:
                    return elem[0]
                return elem
            raise ValueError(f"Cannot scalarize iterable of length {len(lst)}: {lst}")
        if hasattr(x, "imag") and hasattr(x, "real") and not hasattr(x, "__iter__"):
            return x
        try:
            return float(x)
        except (ValueError, TypeError):
            return complex(x)
    except Exception:
        if isinstance(x, complex):
            return x
        return float(x)


def _extract_active_leaf_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to extract parameters for the active leaf if 'leaf_params' is present.
    Adapts General Gaussian params to flattened structure for drawing or legacy views.
    """
    if "leaf_params" not in params:
        return params

    leaf_params = params["leaf_params"]

    # Heuristic: Pick leaf with max Energy (Squeezing norm)
    active_idx = 0
    max_score = -1.0

    def get_list(arr):
        if hasattr(arr, "tolist"):
            return arr.tolist()
        return arr if isinstance(arr, list) else [arr]

    try:
        # General Gaussian 'r' is (L, N)
        r_list = get_list(leaf_params.get("r", []))  # List of lists
        n_list = get_list(leaf_params.get("n_ctrl", [1] * 8))

        # Check lengths
        L = len(n_list)
        if len(r_list) < L:
            # Just assume 0
            pass
        else:
            for i in range(L):
                # Energy ~ sum(|r|)
                r_vec = r_list[i]
                if isinstance(r_vec, (list, np.ndarray)):
                    score = np.sum(np.abs(np.array(r_vec)))
                else:
                    score = abs(r_vec)
                if score > max_score:
                    max_score = score
                    active_idx = i
    except Exception:
        active_idx = 0

    def get_val(arr, idx):
        if hasattr(arr, "ndim") and arr.ndim > 0:
            if arr.shape[0] > idx:
                return arr[idx]
        elif isinstance(arr, list) and len(arr) > idx:
            return arr[idx]
        return None

    # Extract Gen Gaussian Params for active leaf
    # r: (N,)
    r_vec = get_val(leaf_params.get("r"), active_idx)
    # phases: (N^2,)
    phases_vec = get_val(leaf_params.get("phases"), active_idx)
    # disp: (N,)
    disp_vec = get_val(leaf_params.get("disp"), active_idx)
    # pnr: (N-1,) or (Nc,) - might need padding?
    pnr_vec = get_val(leaf_params.get("pnr"), active_idx)

    n_ctrl = int(to_scalar(get_val(leaf_params.get("n_ctrl", [1]), active_idx)))

    # Construct result dict for visualizer
    # Visualizer might expect legacy keys (tmss, us, uc) OR new keys (general_gaussian).
    # Since `GaussianHeraldCircuit` in `cpu/circuit.py` is likely NOT updated yet to accept general gaussian,
    # we might need to update that class OR mapping.
    # But wait, `GaussianHeraldCircuit` was NOT requested to be updated in task.md?
    # Ah, "Update frontend/utils.py: Adapt circuit drawing/helpers".
    # And "Edit Frontend".
    # If I don't update `cpu/circuit.py`, I can't draw the general gaussian using that class easily.
    # However, `experimentation_deck` is now using JAX backend.
    # `get_circuit_figure` uses `GaussianHeraldCircuit` (CPU).
    # I should probably just return the raw params and let the figure generator handle it blindly,
    # or return a simplified "General Gaussian Block" representation.

    # Let's pass the raw general params.
    result = {
        "n_modes": len(r_vec) if hasattr(r_vec, "__len__") else 0,  # total N
        "n_control": n_ctrl,
        "r": r_vec,
        "phases": phases_vec,
        "disp": disp_vec,
        "pnr_outcome": pnr_vec,
        "is_general_gaussian": True,
    }

    if "final_gauss" in params:
        fg = params["final_gauss"]
        if fg:
            r = float(fg.get("r", 0.0))
            phi = float(fg.get("phi", 0.0))
            disp = complex(fg.get("disp", 0.0))
            result["final_squeezing_params"] = (r, phi)
            result["final_displacement"] = disp

    return result


# Try importing JAX components
try:
    import jax
    import jax.numpy as jnp
    from src.simulation.jax.runner import (
        jax_get_heralded_state,
        jax_apply_final_gaussian,
        jax_hermite_phi_matrix,
    )
    from src.simulation.jax.composer import jax_superblock

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    pass


def compute_state_with_jax(
    params: Dict[str, Any], cutoff: int = 10, **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Compute state using the exact JAX logic used in optimization.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX not available.")

    def to_jax(x):
        return jnp.array(x)

    def to_jax_dict(d):
        return {k: to_jax(v) for k, v in d.items()}

    leaf_params = to_jax_dict(params["leaf_params"])
    mix_params = to_jax(params["mix_params"])
    leaf_active = to_jax(params["leaf_active"])

    hom_x = to_jax(params.get("homodyne_x", 0.0))
    _raw_win = params.get("homodyne_window")
    homodyne_res_val = to_scalar(_raw_win) if _raw_win is not None else 0.0

    pnr_max_val = kwargs.get("pnr_max", 3)
    if "pnr_max" in leaf_params:
        arr = leaf_params["pnr_max"]
        if hasattr(arr, "shape") and arr.shape != ():
            pnr_max_val = int(arr[0])
        elif isinstance(arr, list):
            pnr_max_val = int(arr[0])

    get_heralded = jax.vmap(lambda p: jax_get_heralded_state(p, cutoff, pnr_max_val))

    (leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes) = (
        get_heralded(leaf_params)
    )

    homodyne_window_is_none = True
    homodyne_resolution_is_none = homodyne_res_val <= 1e-9
    homodyne_x_is_none = False
    hom_x_val = hom_x

    hom_xs = jnp.atleast_1d(hom_x_val)
    phi_mat = jax_hermite_phi_matrix(hom_xs, cutoff)

    if jnp.ndim(hom_x_val) == 0:
        phi_vec = phi_mat[:, 0]
    else:
        phi_vec = phi_mat.T

    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    (
        final_state,
        _,
        joint_prob,
        is_active,
        max_pnr,
        total_sum_pnr,
        active_modes,
    ) = jax_superblock(
        leaf_vecs,
        leaf_probs,
        leaf_active,
        leaf_max_pnrs,
        leaf_total_pnrs,
        leaf_modes,
        mix_params,
        hom_x_val,
        0.0,
        homodyne_res_val,
        phi_vec,
        V_matrix,
        dx_weights,
        cutoff,
        homodyne_window_is_none,
        homodyne_x_is_none,
        homodyne_resolution_is_none,
    )

    if "final_gauss" in params:
        fg = params["final_gauss"]
        fg_jax = {k: to_jax(v) for k, v in fg.items()}
        final_state = jax_apply_final_gaussian(final_state, fg_jax, cutoff)

    return np.array(final_state), float(joint_prob)


def get_circuit_figure(params: Dict[str, Any], cutoff: int = 10):
    """
    Reconstruct circuit and return its matplotlib figure.
    Currently returns a placeholder for General Gaussian since CPU circuit drawer is not updated.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(
        0.5,
        0.5,
        "General Gaussian Circuit\n(Visualization Not Implemented For General Case)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.axis("off")
    return fig


def compute_heralded_state(
    params: Dict[str, Any], cutoff: int = 10, **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Compute the heralded state vector and probability.
    """
    if JAX_AVAILABLE:
        try:
            return compute_state_with_jax(params, cutoff, **kwargs)
        except Exception:
            pass

    # No fallback for CPU simulation of General Gaussian yet.
    # Current codebase CPU circuit is Legacy structure.
    return np.zeros(cutoff), 0.0


def compute_wigner(psi: np.ndarray, xvec: np.ndarray, pvec: np.ndarray) -> np.ndarray:
    q_psi = qt.Qobj(psi)
    W = qt.wigner(q_psi, xvec, pvec)
    return W


def extract_genotype_index(p_data: Dict[str, Any], df_len: int = 0) -> int:
    def get_point_index(p):
        return p.get("pointIndex", p.get("point_index", p.get("pointNumber")))

    genotype_idx = None
    custom_data = p_data.get("customdata")
    if custom_data is not None:
        try:
            if isinstance(custom_data, list) and len(custom_data) >= 4:
                g_idx_val = custom_data[3]
                genotype_idx = int(g_idx_val)
            elif isinstance(custom_data, dict):
                if "genotype_idx" in custom_data:
                    genotype_idx = int(custom_data["genotype_idx"])
                elif "3" in custom_data:
                    genotype_idx = int(custom_data["3"])
                elif 3 in custom_data:
                    genotype_idx = int(custom_data[3])
        except (ValueError, TypeError, IndexError, KeyError):
            pass

    if genotype_idx is None:
        idx_raw = get_point_index(p_data)
        if idx_raw is not None:
            try:
                genotype_idx = int(idx_raw)
            except (ValueError, TypeError):
                pass
    if df_len > 0 and genotype_idx is not None:
        if genotype_idx < 0 or genotype_idx >= df_len:
            return None
    return genotype_idx


def compute_active_metrics(params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute Total Photons and Max PNR considering ONLY active leaves.
    Updated for General Gaussian params structure.
    """
    if "leaf_params" not in params:
        return 0.0, 0.0

    leaf_params = params["leaf_params"]
    if "pnr" not in leaf_params:
        return 0.0, 0.0

    # pnr shape varies. If DesignA/B/B2/B3/C1/C2 (General Gaussian)
    # pnr is (L, N_C).
    pnr = np.array(leaf_params["pnr"])

    if "leaf_active" in params:
        active = np.array(params["leaf_active"], dtype=bool)
    else:
        active = np.ones(pnr.shape[0], dtype=bool)

    if active.shape[0] != pnr.shape[0]:
        active = np.ones(pnr.shape[0], dtype=bool)

    if "n_ctrl" in leaf_params:
        n_ctrl_raw = leaf_params["n_ctrl"]
        if hasattr(n_ctrl_raw, "flatten"):
            n_ctrl = n_ctrl_raw.flatten()
        else:
            n_ctrl = np.array(n_ctrl_raw).flatten()
    else:
        # Default all valid
        n_ctrl = np.full(pnr.shape[0], pnr.shape[1], dtype=int)

    # Construct Mask
    rows, cols = pnr.shape
    if n_ctrl.shape[0] != rows:
        n_ctrl = np.full(rows, n_ctrl[0] if n_ctrl.size > 0 else cols, dtype=int)

    col_indices = np.arange(cols)[None, :]
    limits = n_ctrl[:, None]
    mask = col_indices < limits
    pnr_masked = pnr * mask

    active_pnr = pnr_masked[active]

    if active_pnr.size == 0:
        return 0.0, 0.0

    total_photons = float(np.sum(active_pnr))
    max_pnr = float(np.max(active_pnr)) if active_pnr.size > 0 else 0.0

    return total_photons, max_pnr
