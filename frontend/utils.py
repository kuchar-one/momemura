import os
import sys
import numpy as np
import qutip as qt
from typing import List, Tuple, Dict, Any

# Ensure we can import from src
# Assuming this is running from project root or frontend/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.result_manager import OptimizationResult  # noqa: E402
from src.simulation.cpu.circuit import GaussianHeraldCircuit  # noqa: E402


def list_runs(output_dir: str = "output") -> List[str]:
    """List all run directories in the output folder."""
    if not os.path.exists(output_dir):
        return []

    runs = [
        d
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and not d.startswith(".")
    ]
    # Sort by timestamp (descending)
    runs.sort(reverse=True)

    # Also scan for "experiments" folder and include its subfolders?
    # Or just `output/experiments/GROUP_ID`.
    # Let's check `output/experiments` if it exists.
    exp_dir = os.path.join(output_dir, "experiments")
    if os.path.exists(exp_dir):
        exp_groups = [
            f"experiments/{d}"
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith(".")
        ]
        exp_groups.sort(reverse=True)
        # Prepend experiment groups to the list
        runs = exp_groups + runs

    return runs


def load_run(run_dir: str) -> OptimizationResult:
    """
    Load OptimizationResult from a run directory.
    Detects if it's a single run or an experiment group.
    """
    # Check if run_dir contains 'experiments' part or if it has subfolders with results
    # Better logic: Check if run_dir contains 'results.pkl' DIRECTLY.
    # If not, check if children have 'results.pkl'.

    path = (
        os.path.join(project_root, run_dir) if not os.path.isabs(run_dir) else run_dir
    )

    if os.path.exists(os.path.join(path, "results.pkl")):
        return OptimizationResult.load(path)

    # Check for subdirs
    has_subruns = False
    if os.path.isdir(path):
        for item in os.listdir(path):
            sub = os.path.join(path, item)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "results.pkl")):
                has_subruns = True
                break

    if has_subruns:
        # Import dynamically to avoid circular issues if any (though result_manager is safe)
        from src.utils.result_manager import AggregatedOptimizationResult

        return AggregatedOptimizationResult.load_group(path)

    raise ValueError(f"No valid run found at {path}")


def to_scalar(x: Any) -> Any:  # Union[float, complex]
    """Safely convert x to a scalar float or complex."""
    try:
        # 0. Handle None
        if x is None:
            return 0.0

        # 1. Exact Scalar types
        if isinstance(x, (float, complex, int)) and not isinstance(x, bool):
            return x

        # 2. Numpy/Jax Array/Scalar wrapper
        if hasattr(x, "item"):
            val = x.item()
            return val

        # 3. Iterables (Lists, Tuples) - but NOT strings
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            lst = list(x)
            if len(lst) == 1:
                elem = lst[0]
                # Recursively scalarize
                # Or flat handling
                if hasattr(elem, "item"):
                    return elem.item()
                if isinstance(elem, (float, complex, int)):
                    return elem
                if isinstance(elem, list) and len(elem) == 1:
                    return elem[0]

                # Check duck typing on ELEMENT (safe because element is unwrapped)
                if hasattr(elem, "imag"):
                    return elem

                return elem  # Return unpacked element regardless, hoping for the best

            raise ValueError(f"Cannot scalarize iterable of length {len(lst)}: {lst}")

        # 4. Fallback Duck Typing (for scalar-like objects that failed isinstance)
        # Be careful not to catch arrays here if they skipped item check?
        # Arrays hav 'imag' but usually have 'item'.
        # If item failed/missing, maybe it IS a scalar?
        if hasattr(x, "imag") and hasattr(x, "real"):
            # Verify it's not iterable (array)
            if not hasattr(x, "__iter__"):
                return x

        # 5. Fallback Casting
        try:
            return float(x)
        except (ValueError, TypeError):
            # If float conversion fails (e.g. complex), try complex conversion
            return complex(x)

    except (ValueError, TypeError) as e:
        # Final catch-all with useful debug info
        # Check standard types one last time
        if isinstance(x, complex):
            return x

        raise ValueError(
            f"Failed to scalarize value '{x}' (Type: {type(x)}): {e}"
        ) from e
    except Exception:
        # Absolute last resort
        return float(x)


def _extract_active_leaf_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to extract parameters for the active leaf if 'leaf_params' is present.
    Returns a flattened dictionary compatible with GaussianHeraldCircuit.
    """
    if "leaf_params" not in params:
        return params

    leaf_params = params["leaf_params"]
    # We ignore leaf_active array for finding the "Best/Main" leaf
    # Instead, we should trace the tree from root to find the most significant active path.
    # However, simple tracing gets stuck if root mixes 50/50.
    # ALTERNATIVE: Scan all 8 leaves and pick the one with MAX energy (Squeezing * n_ctrl).
    # This guarantees we show the "biggest" configuration.

    active_idx = 0
    max_score = -1.0

    # Helper extracting flattened lists
    def get_list(arr):
        if hasattr(arr, "tolist"):
            return arr.tolist()
        return arr if isinstance(arr, list) else [arr]  # dummy fallback

    # Scan leaves
    try:
        r_list = get_list(leaf_params.get("tmss_r", [0.0] * 8))
        n_list = get_list(leaf_params.get("n_ctrl", [1] * 8))

        # Ensure lists are long enough
        L = 8
        if len(r_list) < 8:
            r_list = list(r_list) + [0.0] * (8 - len(r_list))
        if len(n_list) < 8:
            n_list = list(n_list) + [1] * (8 - len(n_list))

        for i in range(L):
            # Score = |r| + n_ctrl
            # If r is 0 and n_ctrl is small, score is low.
            # If r is -1.17 and n_ctrl 2, score ~ 3.17.
            try:
                # Handle nested lists just in case
                r_val = abs(float(to_scalar(r_list[i])))
                n_val = int(to_scalar(n_list[i]))
                score = r_val + n_val
                if score > max_score:
                    max_score = score
                    active_idx = i
            except Exception:
                pass
    except Exception:
        # Fallback to index 0 if scan fails
        active_idx = 0

    # Helper to scalarize from array

    # Helper to scalarize from array
    def get_val(arr, idx):
        # First, access the row for the leaf
        val = None
        if hasattr(arr, "ndim") and arr.ndim > 0:
            if arr.shape[0] > idx:
                val = arr[idx]
        elif isinstance(arr, list) and len(arr) > idx:
            val = arr[idx]
        else:
            # Fallback: if it's a scalar, return it.
            # If it's a container (list/array) and we missed bounds, return None (safe default)
            is_container = False
            if isinstance(arr, list):
                is_container = True
            elif hasattr(arr, "ndim") and arr.ndim > 0:
                is_container = True

            if is_container:
                val = None  # Out of bounds
            else:
                val = arr  # Scalar fallback

        # Check if we need to unwrap inner dimension (e.g. for us_phase which is (L, 1))
        # If val is a list of len 1, or array of shape (1,)
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        if hasattr(val, "shape") and val.shape == (1,):
            return val[0]
        if hasattr(val, "shape") and val.ndim == 0:
            return val.item()  # Scalar array

        return val

    # Extract info for this leaf
    # Structure of leaf_params matches Genotype decode:
    # tmss_r: (L,) -> Single Scalar per leaf
    # us_phase: (L, 1) -> Single Scalar per leaf
    # n_ctrl: (L,) -> Single Int

    # 1. TMSS
    tmss_r = to_scalar(get_val(leaf_params.get("tmss_r"), active_idx))
    # TMSS requires at least 1 Control + 1 Signal
    # If n_control < 1, we can't have TMSS
    n_raw = get_val(leaf_params.get("n_ctrl", [1]), active_idx)
    n_control = int(to_scalar(n_raw))

    if n_control >= 1:
        tmss_squeezing = [tmss_r]
    else:
        tmss_squeezing = []

    # 2. US (Signal)
    us_phase = to_scalar(get_val(leaf_params.get("us_phase"), active_idx))
    us_params = {"theta": [], "phi": [], "varphi": [us_phase]}

    # 3. UC (Control)
    # Depends on n_control
    n_raw = get_val(leaf_params.get("n_ctrl", [1]), active_idx)
    n_control = int(to_scalar(n_raw))

    # Extract UC arrays
    uc_theta = leaf_params.get("uc_theta")
    uc_phi = leaf_params.get("uc_phi")
    uc_varphi = leaf_params.get("uc_varphi")

    # Slice for active leaf
    # These are (L, ...) arrays OR lists (since ResultManager converts them)
    def get_slice(arr, idx):
        if hasattr(arr, "ndim") and arr.ndim > 1:
            if arr.shape[0] > idx:
                return arr[idx].tolist()
        elif isinstance(arr, list) and len(arr) > idx:
            item = arr[idx]
            # Unwrap if it's a nested list like [[0.1, 0.2]]
            if isinstance(item, list):
                return item
            elif hasattr(item, "tolist"):
                return item.tolist()
            else:
                # Should return list. If item is scalar (e.g. from flat list), wrap it?
                # If uc_theta is (L, N_pairs), item is (N_pairs,). Perfect.
                return [item]
        return []

    # Truncate lists to match n_control
    n_pairs = (n_control * (n_control - 1)) // 2

    # Debug / Safety: slice might be larger than n_pairs if n_control changed?
    # Or if genotype has max limit.
    theta_slice = get_slice(uc_theta, active_idx)
    phi_slice = get_slice(uc_phi, active_idx)
    varphi_slice = get_slice(uc_varphi, active_idx)

    uc_params = {
        "theta": theta_slice[:n_pairs] if len(theta_slice) >= n_pairs else theta_slice,
        "phi": phi_slice[:n_pairs] if len(phi_slice) >= n_pairs else phi_slice,
        "varphi": varphi_slice[:n_control]
        if len(varphi_slice) >= n_control
        else varphi_slice,
    }

    # 4. Displacements
    disp_s = get_slice(leaf_params.get("disp_s"), active_idx)
    disp_c = get_slice(leaf_params.get("disp_c"), active_idx)

    # 5. PNR
    pnr_outcome = get_slice(leaf_params.get("pnr"), active_idx)

    result = {
        "n_signal": 1,  # Canonical form usually 1 signal
        "n_control": n_control,
        "tmss_squeezing": tmss_squeezing,
        "us_params": us_params,
        "uc_params": uc_params,
        "disp_s": disp_s,
        "disp_c": disp_c[:n_control],
        "pnr_outcome": pnr_outcome[:n_control],
    }

    # Check for final_gauss at top level params
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
    Requires JAX and JAX runner components.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX not available.")

    # helper to cast
    def to_jax(x):
        return jnp.array(x)

    def to_jax_dict(d):
        return {k: to_jax(v) for k, v in d.items()}

    leaf_params = to_jax_dict(params["leaf_params"])
    mix_params = to_jax(params["mix_params"])
    # mix_source removed
    leaf_active = to_jax(params["leaf_active"])

    # Global Homodyne Params (used at mix nodes)
    hom_x = to_jax(params.get("homodyne_x", 0.0))

    # Capture actual window value for probability scaling (Resolution)
    _raw_win = params.get("homodyne_window")
    homodyne_res_val = to_scalar(_raw_win) if _raw_win is not None else 0.0
    if homodyne_res_val == 0.0:
        # If 0.0, we probably shouldn't scale by 0? Or is it technically 0 prob?
        # Backend treats 0.0 as 0.0.
        pass

    # FORCE Point Homodyne Mode (matching Backend behavior)
    hom_win = None

    # PNR Max inference
    # Use argument pnr_max if provided, else try to find it, else default 3
    # Actually pnr_max should be passed in.
    pnr_max_val = kwargs.get("pnr_max", 3)

    if "pnr_max" in leaf_params and pnr_max_val == 3:
        # Fallback to attempting to read from leaf_params if not explicit
        pnr_val_raw = leaf_params["pnr_max"]
        # It's an array (L,)
        if hasattr(pnr_val_raw, "ndim") and pnr_val_raw.ndim > 0:
            pnr_max_val = int(pnr_val_raw[0])
        elif isinstance(pnr_val_raw, list) and len(pnr_val_raw) > 0:
            pnr_max_val = int(pnr_val_raw[0])

    get_heralded = jax.vmap(lambda p: jax_get_heralded_state(p, cutoff, pnr_max_val))

    (leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes) = (
        get_heralded(leaf_params)
    )

    homodyne_x_is_none = False  # usually 0.0 means 0.0
    homodyne_window_is_none = (
        hom_win is None
    )  # Forces Point Mode (hom_win is None due to fix above)

    # Logic: hom_win arg is None (forced), but we need the VALUE for resolution.
    # Where to get value? 'hom_win' var was set to None.
    # I need to retrieve the original value before I overwrote it.

    # Define legacy variables used later
    hom_x_val = hom_x
    hom_win_val = homodyne_res_val  # Use captured value, or 0.0?
    if hom_win_val is None:
        hom_win_val = 0.0  # Safety

    # Point homodyne setup
    hom_xs = jnp.atleast_1d(hom_x_val)
    phi_mat = jax_hermite_phi_matrix(hom_xs, cutoff)

    if jnp.ndim(hom_x_val) == 0:
        phi_vec = phi_mat[:, 0]
    else:
        phi_vec = phi_mat.T

    # Window setup
    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    # Note: homodyne_window_is_none is True (forced), so this block skipped.
    if not homodyne_window_is_none:
        from numpy.polynomial.legendre import leggauss

        n_nodes = 61
        _, weights = leggauss(n_nodes)
        half_w = hom_win_val / 2.0
        center = hom_x_val
        xs = center + half_w * params.get("nodes_cache", np.zeros(n_nodes))
        nodes, weights = leggauss(n_nodes)
        xs = center + half_w * nodes
        dx_weights = jnp.array(half_w * weights)
        V_matrix = jax_hermite_phi_matrix(jnp.array(xs), cutoff)

    # Call Superblock
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
        # mix_source removed
        hom_x_val,
        hom_win_val,
        homodyne_res_val,  # resolution = window size
        phi_vec,
        V_matrix,
        dx_weights,
        cutoff,
        homodyne_window_is_none,
        homodyne_x_is_none,
        False,  # homodyne_resolution_is_none=False (Apply Scaling)
    )

    # 3. Final Gaussian
    if "final_gauss" in params:
        fg = params["final_gauss"]
        fg_jax = {k: to_jax(v) for k, v in fg.items()}
        final_state = jax_apply_final_gaussian(final_state, fg_jax, cutoff)

    return np.array(final_state), float(joint_prob)


def get_circuit_figure(params: Dict[str, Any], cutoff: int = 10):
    """
    Reconstruct circuit and return its matplotlib figure.
    params: Dictionary of circuit parameters
    """
    # If tree (mix_params present), render tree placeholder?
    if "mix_params" in params and len(params.get("mix_params", [])) > 0:
        pass

    # Adapt params if coming from tree genotype
    flat_params = _extract_active_leaf_params(params)

    n_signal = int(flat_params.get("n_signal", 1))
    n_control = int(flat_params.get("n_control", 1))

    # Extract final ops
    final_sq = flat_params.get("final_squeezing_params")
    final_disp = flat_params.get("final_displacement")

    circ = GaussianHeraldCircuit(
        n_signal=n_signal,
        n_control=n_control,
        tmss_squeezing=flat_params.get("tmss_squeezing", []),
        us_params=flat_params.get("us_params"),
        uc_params=flat_params.get("uc_params"),
        disp_s=flat_params.get("disp_s"),
        disp_c=flat_params.get("disp_c"),
        mesh="rectangular",  # Default
        hbar=2.0,
    )

    # Set final ops
    if hasattr(circ, "final_squeezing_params"):
        circ.final_squeezing_params = final_sq
    if hasattr(circ, "final_displacement"):
        circ.final_displacement = final_disp

    circ.build()

    fig, ax = circ.plot_circuit()
    return fig


def compute_heralded_state(
    params: Dict[str, Any], cutoff: int = 10, **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Compute the heralded state vector and probability.
    Prefers JAX implementation handling full tree.
    """
    if JAX_AVAILABLE:
        try:
            return compute_state_with_jax(params, cutoff, **kwargs)
        except Exception:
            pass

    # Fallback to single/active leaf logic (Original behavior)
    flat_params = _extract_active_leaf_params(params)
    n_control = int(flat_params.get("n_control", 1))
    pnr = flat_params.get("pnr_outcome", [0] * n_control)

    # We need to rebuild circuit here because get_circuit_figure returns fig, not circ
    # Or refactor? For now, duplication is safer than complexity
    n_signal = int(flat_params.get("n_signal", 1))

    circ = GaussianHeraldCircuit(
        n_signal=n_signal,
        n_control=n_control,
        tmss_squeezing=flat_params.get("tmss_squeezing", []),
        us_params=flat_params.get("us_params"),
        uc_params=flat_params.get("uc_params"),
        disp_s=flat_params.get("disp_s"),
        disp_c=flat_params.get("disp_c"),
        mesh="rectangular",
        hbar=2.0,
    )

    final_sq = flat_params.get("final_squeezing_params")
    final_disp = flat_params.get("final_displacement")

    if hasattr(circ, "final_squeezing_params"):
        circ.final_squeezing_params = final_sq
    if hasattr(circ, "final_displacement"):
        circ.final_displacement = final_disp

    circ.build()

    psi, prob = circ.herald(pnr_outcome=pnr, signal_cutoff=cutoff, check_purity=True)
    return psi, float(prob)


def compute_wigner(psi: np.ndarray, xvec: np.ndarray, pvec: np.ndarray) -> np.ndarray:
    """
    Compute Wigner function for a state vector using QuTiP.
    """
    q_psi = qt.Qobj(psi)
    W = qt.wigner(q_psi, xvec, pvec)
    return W


def extract_genotype_index(p_data: Dict[str, Any], df_len: int = 0) -> int:
    """
    Robustly extract the genotype index from Plotly selection data.
    """

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
    
    Args:
        params: Decoded circuit parameters dict.
        
    Returns:
        (total_active_photons, max_active_pnr)
    """
    # 1. Get PNR array
    if "leaf_params" not in params:
        return 0.0, 0.0

    leaf_params = params["leaf_params"]
    if "pnr" not in leaf_params:
        return 0.0, 0.0

    pnr = np.array(leaf_params["pnr"]) # Shape (L, N_C)
    
    # 2. Get Active Flags
    if "leaf_active" in params:
        active = np.array(params["leaf_active"], dtype=bool)
    else:
        # Fallback to all active
        active = np.ones(pnr.shape[0], dtype=bool)
        
    # Ensure shapes match
    if active.shape[0] != pnr.shape[0]:
        active = np.ones(pnr.shape[0], dtype=bool)
        
    # 3. Filter
    active_pnr = pnr[active] # Shape (N_active, N_C)
    
    if active_pnr.size == 0:
        return 0.0, 0.0
        
    total_photons = float(np.sum(active_pnr))
    max_pnr = float(np.max(active_pnr)) if active_pnr.size > 0 else 0.0
    
    return total_photons, max_pnr
