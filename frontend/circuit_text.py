from typing import Dict, Any


def describe_preparation_circuit(params: Dict[str, Any], genotype_name="A") -> str:
    """
    Generates a textual description of the state preparation circuit based on decoded parameters.
    Handles 'Design A' (Tree structure).
    """

    lines = []
    lines.append(f"### Circuit Description (Genotype {genotype_name})")

    # Global Params
    if "homodyne_x" in params:
        from frontend import utils  # Ensure import if moving scope

        hx_val = params["homodyne_x"]
        if hasattr(hx_val, "__len__") and not isinstance(hx_val, str):
            # It's a vector (Design 0)
            try:
                hx_str = str(list(hx_val))
            except Exception:
                hx_str = str(hx_val)
            # Show abbreviated if too long?
            if len(hx_str) > 50:
                hx_str = hx_str[:47] + "..."
            lines.append(
                f"- **Homodyne Measurement**: X=[Vector], Window={utils.to_scalar(params.get('homodyne_window', 0.0)):.4f}"
            )
        else:
            hx = utils.to_scalar(hx_val)
            win = utils.to_scalar(params.get("homodyne_window", 0.0))
            lines.append(f"- **Homodyne Measurement**: X={hx:.4f}, Window={win:.4f}")

    # Helper to describe a leaf
    def describe_leaf(idx):
        active = params["leaf_active"][idx]
        status = "✅ ACTIVE" if active else "🔴 INACTIVE"

        # Leaf Params
        # Access arrays directly assuming shape (L, ...)
        def get_val(key, i):
            arr = params["leaf_params"][key]
            if hasattr(arr, "ndim") and arr.ndim > 0:
                return arr[i]
            return arr[i]

        from frontend import utils

        # Squeezing: Try 'r' first (General Gaussian), then 'tmss_r' (Legacy)
        r_list = get_val("r", idx)
        if r_list is not None and hasattr(r_list, "__len__"):
            try:
                r_vals = [float(x) for x in r_list]
                r_desc = f"{[round(x, 3) for x in r_vals]}"
            except Exception:
                r_desc = str(r_list)
            r_str = f"r={r_desc}"
        else:
            try:
                r_val = utils.to_scalar(get_val("tmss_r", idx))
                r_str = f"r={r_val:.2f}"
            except (KeyError, TypeError):
                r_str = "r=?"

        n_ctrl = int(utils.to_scalar(get_val("n_ctrl", idx)))

        # PNR: Slice by n_ctrl
        try:
            raw_pnr = get_val("pnr", idx)
            if hasattr(raw_pnr, "tolist"):
                pnr_list = raw_pnr.tolist()
            elif isinstance(raw_pnr, list):
                pnr_list = raw_pnr
            else:
                pnr_list = [raw_pnr]

            pnr_list = [int(utils.to_scalar(x)) for x in pnr_list]

            if n_ctrl > 0:
                final_pnr = pnr_list[:n_ctrl]
                pnr_str = str(final_pnr)
            else:
                pnr_str = "[]"
        except (KeyError, TypeError):
            pnr_str = "?"

        # Phases (Passive Unitary) - N^2 phases for Clements decomposition
        try:
            phases = get_val("phases", idx)
            if phases is not None and hasattr(phases, "__len__"):
                phases_list = [float(x) for x in phases]
                n_phases = len(phases_list)
                # N² = n_phases, so N = sqrt(n_phases)
                import math

                # Use effective N based on n_ctrl (backend physics)
                # n_ctrl=0 -> 1 mode, n_ctrl=1 -> 2 modes
                N = n_ctrl + 1

                if N * N <= n_phases:
                    # Truncate phases to match backend usage (first N*N params)
                    phases_list = phases_list[: N * N]
                    n_bs_params = N * (N - 1)

                    bs_params = phases_list[:n_bs_params]
                    final_phases = phases_list[n_bs_params:]

                    # Clements mesh structure - iterate to find BS mode indices
                    # Same logic as jax_clements_unitary
                    bs_strs = []
                    param_idx = 0
                    for s in range(N):  # N layers
                        start = 0 if (s % 2 == 0) else 1
                        for a in range(start, N - 1, 2):
                            if param_idx + 1 < len(bs_params):
                                theta = bs_params[param_idx]
                                phi = bs_params[param_idx + 1]
                                # Subscript showing which modes are connected
                                bs_strs.append(
                                    f"BS_{a}{a + 1}(θ={theta:.2f}, φ={phi:.2f})"
                                )
                                param_idx += 2

                    # Format final phases with mode labels
                    final_strs = [f"φ_{i}={p:.2f}" for i, p in enumerate(final_phases)]
                    final_str = ", ".join(final_strs)

                    phases_str = f"Passive Unitary (N={N} modes):\n"
                    phases_str += f"    {', '.join(bs_strs)}\n"
                    phases_str += f"    Final: [{final_str}]"
                else:
                    # Fallback: just show raw values
                    phases_fmt = [round(x, 3) for x in phases_list]
                    phases_str = f"phases={phases_fmt}"
            else:
                phases_str = "phases=[]"
        except (KeyError, TypeError):
            phases_str = "phases=?"

        # Displacements - complex valued
        try:
            disp_vec = get_val("disp", idx)
            if disp_vec is not None:
                if hasattr(disp_vec, "__len__"):
                    # Complex array - show all values
                    disp_list = []
                    for d in disp_vec:
                        try:
                            c = complex(d)
                            if abs(c.imag) < 1e-6:
                                disp_list.append(f"{c.real:.3f}")
                            else:
                                disp_list.append(f"{c.real:.3f}{c.imag:+.3f}j")
                        except Exception:
                            disp_list.append(str(d))
                    disp_str = f"disp=[{', '.join(disp_list)}]"
                else:
                    disp_str = f"disp={disp_vec}"
            else:
                # Legacy
                disp_s = utils.to_scalar(get_val("disp_s", idx))
                disp_str = f"disp_s={disp_s:.2f}"
        except (KeyError, TypeError):
            disp_str = "disp=?"

        # Build description with all parameters
        desc = f"**Leaf {idx}** [{status}]:\n"
        desc += f"  - {r_str}\n"
        desc += f"  - n_ctrl={n_ctrl}, PNR={pnr_str}\n"
        desc += f"  - {disp_str}\n"
        desc += f"  - {phases_str}"
        return desc

    lines.append("\n#### 1. Leaf States (Layer 0)")
    for i in range(8):
        lines.append(f"- {describe_leaf(i)}")

    # Describing Tree Mixing
    # Layer 1: 4 pairs (0,1), (2,3)...
    mix_params = params["mix_params"]  # (7, 3)
    # mix_source removed

    # Tree Indices Logic
    # 0..3: Layer 1 (bottom)
    # 4..5: Layer 2
    # 6:    Layer 3 (root)

    def describe_mix(mix_idx, depth_idx, in_A, in_B):
        from frontend import utils

        theta = utils.to_scalar(mix_params[mix_idx][0])
        phi = utils.to_scalar(mix_params[mix_idx][1])

        # Action is implicit
        bs_desc = f"BS(θ={theta:.2f}, φ={phi:.2f})"
        lines.append(f"- **Node {mix_idx}** (Inputs {in_A}, {in_B}): {bs_desc}")

    lines.append("\n#### 2. Mixing Layers")

    curr_mix = 0
    # Layer 1
    lines.append("\n\n**Layer 1 (8 -> 4)**")
    for i in range(4):
        describe_mix(curr_mix, i, f"Leaf {2 * i}", f"Leaf {2 * i + 1}")
        curr_mix += 1

    # Layer 2
    lines.append("\n\n**Layer 2 (4 -> 2)**")
    for i in range(2):
        describe_mix(
            curr_mix, i, f"Node {2 * i} (Layer 1)", f"Node {2 * i + 1} (Layer 1)"
        )
        curr_mix += 1

    # Layer 3
    lines.append("\n\n**Layer 3 (2 -> 1)**")
    describe_mix(curr_mix, 0, "Node 4 (Layer 2)", "Node 5 (Layer 2)")

    # Final Gaussian
    if "final_gauss" in params:
        fg = params["final_gauss"]
        lines.append("\n#### 3. Final Gaussian Ops")
        lines.append(f"- Squeezing: {fg}")  # Improve formatting if needed

    return "\n".join(lines)
