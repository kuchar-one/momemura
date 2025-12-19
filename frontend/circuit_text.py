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

        hx = utils.to_scalar(params["homodyne_x"])
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

        r = utils.to_scalar(get_val("tmss_r", idx))
        n_ctrl = int(utils.to_scalar(get_val("n_ctrl", idx)))

        # PNR: Slice by n_ctrl
        raw_pnr = get_val("pnr", idx)  # array
        # Convert to list of ints
        if hasattr(raw_pnr, "tolist"):
            pnr_list = raw_pnr.tolist()
        elif isinstance(raw_pnr, list):
            pnr_list = raw_pnr
        else:
            pnr_list = [raw_pnr]

        # Take first n_ctrl elements
        if n_ctrl > 0:
            final_pnr = pnr_list[:n_ctrl]
            pnr_str = str(final_pnr)
        else:
            pnr_str = "[]"

        # Displacements
        disp_s = utils.to_scalar(get_val("disp_s", idx))

        desc = f"**Leaf {idx}** [{status}]: TMSS(r={r:.2f}), n_ctrl={n_ctrl}, PNR={pnr_str}, DispS={disp_s:.2f}"
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
