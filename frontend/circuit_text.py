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
            # General Gaussian r is a vector (N modes). Signal mode usually first or we show Max?
            # Or show vector.
            # Assuming N=3 (1 sig + 2 ctrl).
            # For brevity, let's show max r or all.
            # Convert to simple list
            try:
                r_vals = [float(x) for x in r_list]
                r_desc = f"{r_vals}"
            except Exception:
                r_desc = str(r_list)
            r_str = f"r={r_desc}"
        else:
            # Legacy fallback
            try:
                r_val = utils.to_scalar(get_val("tmss_r", idx))
                r_str = f"r={r_val:.2f}"
            except (KeyError, TypeError):
                r_str = "r=?"

        n_ctrl = int(utils.to_scalar(get_val("n_ctrl", idx)))

        # PNR: Slice by n_ctrl
        try:
            raw_pnr = get_val("pnr", idx)  # array
            # Convert to list of ints
            if hasattr(raw_pnr, "tolist"):
                pnr_list = raw_pnr.tolist()
            elif isinstance(raw_pnr, list):
                pnr_list = raw_pnr
            else:
                pnr_list = [raw_pnr]

            # Format elements
            pnr_list = [int(utils.to_scalar(x)) for x in pnr_list]

            # Take first n_ctrl elements
            if n_ctrl > 0:
                final_pnr = pnr_list[:n_ctrl]
                pnr_str = str(final_pnr)
            else:
                pnr_str = "[]"
        except (KeyError, TypeError):
            pnr_str = "?"

        # Displacements
        # Try 'disp' (General Gaussian) -> shape (2N,) usually.
        # Legacy 'disp_s' (scalar for signal).
        try:
            disp_vec = get_val("disp", idx)
            if disp_vec is not None:
                # Show first 2 elements (Signal Re, Im)
                if hasattr(disp_vec, "__len__") and len(disp_vec) >= 2:
                    d_re = float(disp_vec[0])
                    d_im = float(disp_vec[1])
                    disp_str = f"Disp=({d_re:.2f}, {d_im:.2f})"
                else:
                    disp_str = f"Disp={disp_vec}"
            else:
                # Legacy
                disp_s = utils.to_scalar(get_val("disp_s", idx))
                disp_str = f"DispS={disp_s:.2f}"
        except (KeyError, TypeError):
            disp_str = "Disp=?"

        desc = f"**Leaf {idx}** [{status}]: {r_str}, n_ctrl={n_ctrl}, PNR={pnr_str}, {disp_str}"
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
