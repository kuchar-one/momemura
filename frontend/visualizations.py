import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_global_pareto(df: pd.DataFrame) -> go.Figure:
    """
    Plots the global Pareto front: Expectation (y) vs Probability (x).
    Color varies by Complexity.
    Expectation is minimized. LogProb is -log10(P), minimized.
    So x-axis = LogProb, y-axis = Expectation.
    """
    if df.empty:
        return go.Figure()

    hover_data = [
        "Desc_TotalPhotons",
        "Desc_MaxPNR",
        "Desc_Complexity",
        "genotype_idx",
        "Expectation",
        "LogProb",
    ]

    # Check for Global Dominance column
    symbol_args = {}
    if "GlobalDominant" in df.columns:
        hover_data.append("GlobalDominant")
        # Use symbol to distinguish dominant vs non-dominant
        # True -> Circle/Star, False -> X/Cross?
        # Or Size?

        # We need to map boolean to string for discrete symbolic mapping usually
        df["IsDominant"] = df["GlobalDominant"].apply(
            lambda x: "Global Pareto" if x else "Dominated"
        )

        symbol_args = dict(
            symbol="IsDominant",
            symbol_sequence=[
                "star",
                "circle-open",
            ],  # First is usually top of alphabetic? D comes before G?
            # 'Dominated' (circle-open/x), 'Global Pareto' (star)
            # Actually map takes strings.
            symbol_map={"Global Pareto": "star", "Dominated": "circle-open"},
        )

    fig = px.scatter(
        df,
        x="LogProb",
        y="Expectation",
        color="Complexity",
        hover_data=hover_data,
        title="Global Pareto Front: Expectation vs Probability",
        labels={
            "LogProb": "Negative Log10 Probability (Minimize)",
            "Expectation": "Expectation Value (Minimize)",
            "Complexity": "Complexity",
            "IsDominant": "Dominance Status",
        },
        color_continuous_scale="Viridis",
        **symbol_args,
    )
    fig.update_layout(clickmode="event+select")
    return fig


def plot_best_expectation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of Best Expectation Value per (Total Photons, Complexity) cell.
    """
    if df.empty:
        return go.Figure()

    # Aggregate best (minimum) Expectation for each cell
    # Cell is defined by (Desc_Complexity, Desc_TotalPhotons)
    # Convert descriptors to int for binning

    # We want a dense grid.
    max_complex = int(df["Desc_Complexity"].max()) if not df.empty else 10
    max_photons = int(df["Desc_TotalPhotons"].max()) if not df.empty else 25

    # Create grid
    grid_data = (
        df.groupby(["Desc_Complexity", "Desc_TotalPhotons"])["Expectation"]
        .min()
        .reset_index()
    )

    # Pivot for heatmap
    pivot_table = grid_data.pivot(
        index="Desc_Complexity", columns="Desc_TotalPhotons", values="Expectation"
    )

    # Fill missing with None (for transparent) or NaN
    # Reindex to ensure full grid range
    all_complex = np.arange(max_complex + 1)
    all_photons = np.arange(max_photons + 1)

    pivot_table = pivot_table.reindex(index=all_complex, columns=all_photons)

    fig = px.imshow(
        pivot_table,
        labels=dict(x="Total Photons", y="Complexity", color="Expectation"),
        x=all_photons,
        y=all_complex,
        color_continuous_scale="Viridis_r",  # Reversed so lower (better) is brighter/yellow
        title="Best Expectation Value per Cell",
        aspect="auto",  # Ensure it fills the container and isn't squashed
        origin="lower",  # 0 at bottom
    )
    fig.update_xaxes(side="bottom")
    fig.update_layout(clickmode="event+select")

    return fig


def plot_wigner_function(
    wigner_grid: np.ndarray,
    xvec: np.ndarray,
    pvec: np.ndarray,
    title: str = "Wigner Function",
) -> go.Figure:
    """
    Plots the Wigner function using a Heatmap with a two-slope plateau colormap.
    Mimics matplotlib's PlateauTwoSlopeNorm.
    """
    z_min = np.min(wigner_grid)
    z_max = np.max(wigner_grid)
    vcenter = 0.0

    # Calculate plateau size.
    # If not specified by user, we infer a reasonable default.
    # Wigner usually has range around [-0.3, 0.3].
    # A plateau of 0.02 seems reasonable to hide vacuum noise around 0.
    plateau_size = 0.02

    # Ensure vcenter is within range or expand range
    if z_min > vcenter:
        z_min = vcenter - 0.1
    if z_max < vcenter:
        z_max = vcenter + 0.1

    # Construct Custom Colorscale
    # We want Red-White-Blue (RdBu) but with a white plateau around vcenter
    # RdBu in Plotly: 0=Red, 0.5=White, 1=Blue (Actually it's usually Blue=0, Red=1 or vice versa)
    # Plotly's 'RdBu' is Red (high) to Blue (low)? No, usually Red=Negative, Blue=Positive in Physics?
    # Actually matplotlib 'RdBu' is Red (low) to Blue (high) or vice versa.
    # Wigner convention: Positive (Red/Yellow), Negative (Blue).
    # Let's use Red for positive, Blue for negative.

    # Let's manually define colors:
    # 0.0 (Min): Blue (#053061)
    # Plateau Start: White (#ffffff)
    # Plateau End: White (#ffffff)
    # 1.0 (Max): Red (#67001f)

    # Normalize points to [0, 1]
    def norm(v):
        return (v - z_min) / (z_max - z_min)

    plateau_low = vcenter - plateau_size / 2
    plateau_high = vcenter + plateau_size / 2

    p_low_norm = norm(plateau_low)
    p_high_norm = norm(plateau_high)

    # Clamp
    p_low_norm = max(0.0, min(1.0, p_low_norm))
    p_high_norm = max(0.0, min(1.0, p_high_norm))

    # Define colors
    c_min = "rgb(5, 48, 97)"  # Dark Blue
    c_mid = "rgb(255, 255, 255)"  # White
    c_max = "rgb(103, 0, 31)"  # Dark Red

    # If range is fully positive or fully negative, we might need adjustment,
    # but Wigner is usually centered.

    # Scale:
    # 0.0 -> Min Color
    # ... -> ...
    # p_low -> White
    # p_high -> White
    # 1.0 -> Max Color

    # BUT, to get a linear gradient from Min to Plateau_Low, and Plateau_High to Max,
    # we need to be careful if z_min/z_max are asymmetric.
    # The user's TwoSlopeNorm ensures linearity relative to value, not index.
    # Plotly interpolates linearly between stops.
    # So we simply place the stops at the calculated normalized positions.

    colorscale = [[0.0, c_min], [p_low_norm, c_mid], [p_high_norm, c_mid], [1.0, c_max]]

    # Sort just in case (though math guarantees order if z_min < z_max)
    colorscale.sort(key=lambda x: x[0])

    fig = go.Figure(
        data=go.Heatmap(
            z=wigner_grid,
            x=xvec,
            y=pvec,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="W(x, p)"),
            zmin=z_min,
            zmax=z_max,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="p",
        autosize=True,
    )
    return fig


def plot_tree_circuit(params: dict, genotype_name: str = "A") -> go.Figure:
    """
    Visualizes the binary tree preparation circuit for Genotype A.
    Colors active paths and annotates nodes with parameters.
    Implements implicit mixing logic:
    - Active + Active -> Mix
    - Active + Inactive -> Pass Active
    - Inactive + Inactive -> Inactive
    """
    import networkx as nx
    from frontend import utils

    # Tree Layout Constants
    leaves = 8

    # Graphs
    G = nx.DiGraph()
    pos = {}

    # Layers: 0 (Leaves) -> 1 (4 Mixers) -> 2 (2 Mixers) -> 3 (Root)

    # Add Nodes
    # Layer 0: Leaves 0..7
    for i in range(leaves):
        node_id = f"L{i}"
        G.add_node(node_id, layer=0, label=f"Leaf {i}")
        pos[node_id] = (i, 0)

    # Layer 1: Mixers 0..3
    for i in range(4):
        node_id = f"M1_{i}"
        G.add_node(node_id, layer=1, label=f"Mix {i}")
        pos[node_id] = (2 * i + 0.5, 1)

    # Layer 2: Mixers 0..1
    for i in range(2):
        node_id = f"M2_{i}"
        G.add_node(node_id, layer=2, label=f"Mix {i}")
        pos[node_id] = (4 * i + 1.5, 2)

    # Layer 3: Root
    node_id = "Root"
    G.add_node(node_id, layer=3, label="Output")
    pos[node_id] = (3.5, 3)

    # Parameters
    mix_params = params.get("mix_params", np.zeros((7, 3)))
    # mix_source removed/ignored
    leaf_active = params.get("leaf_active", np.zeros(8, dtype=bool))
    leaf_p = params.get("leaf_params", {})

    # --- Step 1: Determine Activity of Leaves (Bottom-Up) ---
    # A leaf is 'Active' if it is NON-TRIVIAL.
    # We use a helper to check triviality (Vac noise)

    node_active_status = {}  # Map node_id -> bool (Is physically active?)

    def get_leaf_val(idx, key, default, scalar=True):
        arr = leaf_p.get(key, default)
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        if isinstance(arr, list) and len(arr) > idx:
            val = arr[idx]
            if scalar:
                return utils.to_scalar(val)
            return val
        return default

    def is_leaf_active(idx):
        # 1. Genome Active Flag
        flag = leaf_active[idx] if idx < len(leaf_active) else False
        if not flag:
            return False  # Explicitly turned off

        # 2. Check content (Vacuum detection)
        # tmss_r is scalar in legacy, but 'r' is (L, N) list/array in General Gaussian
        # Try 'r' first
        r_list = get_leaf_val(idx, "r", None, scalar=False)
        if r_list is not None:
            # General Gaussian: r is list of squeezings. Active if any distinct from 0?
            if hasattr(r_list, "__len__"):
                r_val = float(np.max(np.abs(r_list)))
            else:
                r_val = abs(utils.to_scalar(r_list))
        else:
            # Fallback to legacy
            r_val = get_leaf_val(idx, "tmss_r", 0.0, scalar=True)

        # Simplified: If r active is True but state is vacuum, treat as Inactive for mixing?
        # Yes, "Vacuum Exclusion" logic implies ignoring vacuum.
        if abs(r_val) < 0.01:
            return False

        return True

    # 1. Leaves
    for i in range(leaves):
        node_active_status[f"L{i}"] = is_leaf_active(i)

    # 2. Mixers (Bottom-Up)
    # Generic Helper
    def calc_mixer_activity(mid, inp1, inp2):
        act1 = node_active_status[inp1]
        act2 = node_active_status[inp2]
        # Active if EITHER is active
        return act1 or act2

    # Layer 1
    for i in range(4):
        node_active_status[f"M1_{i}"] = calc_mixer_activity(
            f"M1_{i}", f"L{2 * i}", f"L{2 * i + 1}"
        )

    # Layer 2
    for i in range(2):
        node_active_status[f"M2_{i}"] = calc_mixer_activity(
            f"M2_{i}", f"M1_{2 * i}", f"M1_{2 * i + 1}"
        )

    # Layer 3 (Root)
    node_active_status["Root"] = calc_mixer_activity("Root", "M2_0", "M2_1")

    # --- Step 2: Trace Active Edges (Top-Down) ---
    # Which edges are CARRYING signal?
    # Start at Root. If Root is Active, check inputs.

    edge_status = {}  # (u, v) -> bool (Is Edge Active?)

    def trace_down(node_id, inp1, inp2):
        if not node_active_status[node_id]:
            # I am inactive. My inputs didn't contribute active signal (or were blocked).
            edge_status[(inp1, node_id)] = False
            edge_status[(inp2, node_id)] = False
            return

        # I am Active. Who contributed?
        act1 = node_active_status[inp1]
        act2 = node_active_status[inp2]

        if act1 and act2:
            # Both Active -> MIX -> Both Edges Active
            edge_status[(inp1, node_id)] = True
            edge_status[(inp2, node_id)] = True
        elif act1 and not act2:
            # Only 1 Active -> PASS 1 -> Edge 1 Active, Edge 2 Inactive
            edge_status[(inp1, node_id)] = True
            edge_status[(inp2, node_id)] = False
        elif not act1 and act2:
            # Only 2 Active -> PASS 2 -> Edge 1 Inactive, Edge 2 Active
            edge_status[(inp1, node_id)] = False
            edge_status[(inp2, node_id)] = True
        else:
            # Neither Active (Impossible if I am active?)
            # Valid if I passed vacuum? But effectively Inactive.
            edge_status[(inp1, node_id)] = False
            edge_status[(inp2, node_id)] = False

    # Trace
    trace_down("Root", "M2_0", "M2_1")
    trace_down("M2_0", "M1_0", "M1_1")
    trace_down("M2_1", "M1_2", "M1_3")
    for i in range(4):
        trace_down(f"M1_{i}", f"L{2 * i}", f"L{2 * i + 1}")

    # Also build Graph Edges for plotting
    for i in range(4):
        G.add_edge(f"L{2 * i}", f"M1_{i}")
        G.add_edge(f"L{2 * i + 1}", f"M1_{i}")
    for i in range(2):
        G.add_edge(f"M1_{2 * i}", f"M2_{i}")
        G.add_edge(f"M1_{2 * i + 1}", f"M2_{i}")
    G.add_edge("M2_0", "Root")
    G.add_edge("M2_1", "Root")

    # --- Step 3: Plot ---

    COLOR_ACTIVE = "#2ca02c"
    COLOR_INACTIVE = "#d62728"
    EDGE_ACTIVE = "#2ca02c"
    EDGE_INACTIVE = "#ffcccc"  # Light Red

    x_active, y_active = [], []
    x_inactive, y_inactive = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        isActive = edge_status.get((u, v), False)

        if isActive:
            x_active.extend([x0, x1, None])
            y_active.extend([y0, y1, None])
        else:
            x_inactive.extend([x0, x1, None])
            y_inactive.extend([y0, y1, None])

    traces = []
    traces.append(
        go.Scatter(
            x=x_inactive,
            y=y_inactive,
            mode="lines",
            line=dict(color=EDGE_INACTIVE, width=1),
            hoverinfo="none",
        )
    )
    traces.append(
        go.Scatter(
            x=x_active,
            y=y_active,
            mode="lines",
            line=dict(color=EDGE_ACTIVE, width=4),
            hoverinfo="none",
        )
    )

    # Nodes
    node_x, node_y, node_text, node_hover, node_color, node_line = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        isActive = node_active_status.get(node, False)

        # Details
        label = node
        hover = ""

        if node.startswith("L"):
            idx = int(node[1:])
            idx = int(node[1:])
            # Prioritize General Gaussian 'r'
            r_list = get_leaf_val(idx, "r", None, scalar=False)
            if r_list is not None:
                if hasattr(r_list, "__len__"):
                    r_val = float(np.max(np.abs(r_list)))
                else:
                    r_val = abs(utils.to_scalar(r_list))
            else:
                r_val = get_leaf_val(idx, "tmss_r", 0.0, scalar=True)
            n_val = int(get_leaf_val(idx, "n_ctrl", 1, scalar=True))

            # --- Detailed Params for Visualization ---
            # US Phase (Signal Phase) or General Phases
            # Legacy: us_phase. General: phases (N^2 list).
            us_ph = get_leaf_val(idx, "us_phase", None, scalar=True)
            if us_ph is None:
                # Try phases
                phs = get_leaf_val(idx, "phases", [], scalar=False)
                if hasattr(phs, "__len__") and len(phs) > 0:
                    us_ph_scalar = utils.to_scalar(phs[0])  # Proxy
                else:
                    us_ph_scalar = 0.0
            else:
                us_ph_scalar = utils.to_scalar(us_ph)

            # PNR - Detected Photons
            # PNR is usually a list of length n_ctrl.
            # get_leaf_val might return the whole list for this leaf if it's (L, N_ctrl)
            # pnr is VECTOR (don't scalarize)
            raw_pnr = get_leaf_val(idx, "pnr", [0], scalar=False)
            # If it's a list (N_ctrl), join it.
            if isinstance(raw_pnr, (list, np.ndarray)):
                pnr_str = ",".join([str(int(utils.to_scalar(x))) for x in raw_pnr])
            else:
                pnr_str = str(int(utils.to_scalar(raw_pnr)))

            # UC Params or General Phases
            uc_th_val = 0.0
            uc_th_raw = get_leaf_val(idx, "uc_theta", [], scalar=False)
            if hasattr(uc_th_raw, "__len__") and len(uc_th_raw) > 0:
                uc_th_val = utils.to_scalar(uc_th_raw[0])

            # Label
            if us_ph is None and get_leaf_val(idx, "r", None, scalar=False) is not None:
                # General Gaussian Label
                label = f"<b>L{idx}</b><br>GG<br>r_max={r_val:.2f}<br>PNR=[{pnr_str}]"
                hover = (
                    f"Leaf {idx} (General Gaussian)<br>Active: {isActive}<br>"
                    f"<b>Max Squeezing:</b> r={r_val:.2f}<br>"
                    f"<b>PNR:</b> [{pnr_str}]"
                )
            else:
                # Legacy Label
                label = f"<b>L{idx}</b><br>r={r_val:.2f}<br>PNR=[{pnr_str}]"
                hover = (
                    f"Leaf {idx}<br>Active: {isActive}<br>"
                    f"<b>Squeezing:</b> r={r_val:.2f}<br>"
                    f"<b>Signal:</b> Phase={us_ph_scalar:.2f}<br>"
                    f"<b>Control:</b> n={n_val}, PNR=[{pnr_str}]<br>"
                    f"<b>Unitary (1st):</b> Th={uc_th_val:.2f}"
                )
        elif node.startswith("M") or node == "Root":
            # Need parameter index
            midx = (
                6
                if node == "Root"
                else (
                    4 + int(node.split("_")[1])
                    if node.startswith("M2")
                    else int(node.split("_")[1])
                )
            )
            # Param
            if midx < len(mix_params):
                theta = utils.to_scalar(mix_params[midx][0])
                phi = utils.to_scalar(mix_params[midx][1])
                label = f"<b>{node.split('_')[0]}</b><br>θ={theta:.2f}"
                hover = f"{node}<br>Active: {isActive}<br>Theta={theta:.2f}<br>Phi={phi:.2f}"

        node_text.append(label)
        node_hover.append(hover)
        node_color.append(COLOR_ACTIVE if isActive else COLOR_INACTIVE)
        node_line.append("black")

    traces.append(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=35, color=node_color, line=dict(width=2, color="black")),
            text=node_text,
            textfont=dict(
                size=10,
                color="white"
                if any(c == COLOR_ACTIVE for c in node_color)
                else "black",
            ),
            hovertext=node_hover,
            hoverinfo="text",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Preparation Circuit Tree (Genotype {genotype_name})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig
