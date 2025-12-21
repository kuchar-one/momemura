import streamlit as st
import numpy as np

# pandas removed
import visualizations as viz
import utils
import plotly.express as px

st.set_page_config(page_title="Momemura Visualizer", layout="wide")

st.title("Momemura Optimization Results")

# Sidebar: Run Selection
st.sidebar.header("Configuration")
runs = utils.list_runs()
if not runs:
    st.warning("No runs found in 'output/' directory.")
    st.stop()

selected_run_dir = st.sidebar.selectbox("Select Run", runs)
run_path = f"output/{selected_run_dir}"


@st.cache_resource
def load_data(path):
    return utils.load_run(path)


def render_solution_details(row, result_obj, key_suffix=""):
    genotype_idx = int(row["genotype_idx"])

    # Metrics
    # Metrics (Safe Formatting)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Expectation", f"{utils.to_scalar(row['Expectation']):.4f}")
    m_col2.metric("LogProb", f"{utils.to_scalar(row['LogProb']):.4f}")
    m_col3.metric("Complexity", f"{utils.to_scalar(row['Complexity']):.2f}")
    m_col4.metric("Total Photons", f"{utils.to_scalar(row['TotalPhotons']):.2f}")

    # Reconstruct Circuit
    params = result_obj.get_circuit_params(genotype_idx)

    # Circuit Plot
    st.subheader("Active Leaf Schematic (Topology)")
    fig_circ = utils.get_circuit_figure(params)
    st.pyplot(fig_circ)

    # Wigner Function
    st.subheader("Wigner Function (Signal Mode)")
    # st.subheader("Wigner Function (Signal Mode)") # This subheader is now inside col_viz2

    with st.spinner("Computing Wigner function..."):
        try:
            # Compute state
            # Compute state
            cutoff = result_obj.config.get("cutoff", 12)  # Use config or default
            pnr_max = int(result_obj.config.get("pnr_max", 3))  # Use config or default

            params["pnr_outcome"] = params.get(
                "pnr_outcome", [0] * int(params.get("n_control", 1))
            )  # Ensure pnr present
            psi, prob = utils.compute_heralded_state(
                params, cutoff=cutoff, pnr_max=pnr_max
            )
            exp_val = row["Expectation"]  # Get expectation for Wigner title

            # --- VISUALIZATION ROW ---
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                st.subheader("State Preparation (Tree)")
                # Display Text Description
                from frontend.circuit_text import describe_preparation_circuit

                run_config = result_obj.config  # Assuming result_obj has config
                desc = describe_preparation_circuit(
                    params, genotype_name=run_config.get("genotype", "A")
                )
                st.markdown(desc)

                # Display Tree Plot
                from frontend.visualizations import plot_tree_circuit

                fig_tree = plot_tree_circuit(
                    params, genotype_name=run_config.get("genotype", "A")
                )
                st.plotly_chart(fig_tree, use_container_width=True)

            # Ensure exp_val and prob are scalar floats
            exp_val = utils.to_scalar(row["Expectation"])
            prob = utils.to_scalar(prob)

            # Calculate Simulated LogProb for comparison
            if prob > 0:
                sim_log_prob = -np.log10(prob)
            else:
                sim_log_prob = np.inf

            stored_log_prob = utils.to_scalar(row["LogProb"])

            st.text(
                f"Herald Probability (Simulated): {prob:.4e}  => LogProb: {sim_log_prob:.4f}"
            )
            st.text(f"Stored LogProb (Backend):       {stored_log_prob:.4f}")

            if abs(sim_log_prob - stored_log_prob) > 1.0:
                st.warning(
                    "⚠️ Mismatch between stored metadata and re-simulated state. "
                    "This likely due to dynamic limits or different cutoff settings. "
                    "The Wigner function shown is correct for the re-simulated parameters."
                )

            with col_viz2:
                st.subheader("Output Wigner Function")
                grid_size = 100
                xvec = np.linspace(-5, 5, grid_size)
                pvec = np.linspace(-5, 5, grid_size)
                W = utils.compute_wigner(psi, xvec, pvec)

                # Title with Simulated values
                fig_wig = viz.plot_wigner_function(
                    W,
                    xvec,
                    pvec,
                    title=f"Exp: {exp_val:.4f} | Prob: {prob:.2e}",
                )
                st.plotly_chart(
                    fig_wig,
                    use_container_width=True,
                    key=f"wigner_{genotype_idx}_{key_suffix}",
                )

        except Exception as e:
            st.error(f"Error computing Wigner: {e}")


try:
    result = load_data(run_path)
    # --- Main Analysis ---

    # Calculate Stats
    try:
        stats = result.get_experiment_stats()
        # Display Stats in Sidebar or top of Main? Sidebar is good for summary.
        # But we are already past sidebar code block (usually sidebar is top).
        # Let's add it to sidebar now using st.sidebar again.

        st.sidebar.markdown("---")
        st.sidebar.subheader("Experiment Statistics")
        st.sidebar.markdown(f"**Total Solutions:** {stats['total_solutions']}")
        st.sidebar.markdown(f"**Global Non-Dominated:** {stats['total_nondominated']}")
        st.sidebar.markdown(f"**Total Generations:** {stats['total_generations']}")
        st.sidebar.markdown(f"**Total Evaluations:** {stats['total_evaluations']}")
    except Exception as e:
        st.sidebar.error(f"Could not calculate stats: {e}")

    df = result.get_pareto_front()
except Exception as e:
    st.error(f"Failed to load run data: {e}")
    st.stop()

if df.empty:
    st.warning("No valid Pareto front solutions found.")
    st.stop()

# --- Main Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Global Pareto Front")
    # Interactive scatter
    fig_pareto = viz.plot_global_pareto(df)

    # If Aggregated, we have GlobalDominant column.
    # If so, maybe update plot traces?
    # Actually, plot_global_pareto is in `visualizations.py`. We should update IT.
    # But we can pass color/symbol args via `hover_data` or similar if we modify it.
    # Let's inspect viz.plot_global_pareto first?
    # Assuming standard scatter.

    # Override logic: IF 'GlobalDominant' in df, specific styling?
    # Let's trust visualizations.py handles generic DF or we update visualizations.py.
    # I'll update frontend/visualizations.py in next step.

    pareto_selection = st.plotly_chart(
        fig_pareto, use_container_width=True, on_select="rerun", key="pareto_plot"
    )

with col2:
    st.subheader("Best Expectation per Cell")
    # Interactive heatmap
    fig_heat = viz.plot_best_expectation_heatmap(df)
    heat_selection = st.plotly_chart(
        fig_heat, use_container_width=True, on_select="rerun", key="heat_plot"
    )

st.divider()

# --- Drill Down Logic ---

# 1. Point Selection from Pareto Plot
selected_points = pareto_selection.get("selection", {}).get("points", [])
# 2. Cell Selection from Heatmap
selected_cells = heat_selection.get("selection", {}).get("points", [])

if selected_points:
    st.header("Selected Solution Details")
    # Handle multiple selection - just take the first one or list them
    # For simplicity, focus on the last selected point or list tabs

    # We can use tabs for multiple points
    # We can use tabs for multiple points
    # Use pointIndex (Streamlit < 1.35) or point_index (Streamlit >= 1.35) or pointNumber (Plotly native)
    # The selection state structure can vary.
    # Let's inspect the keys if needed, but safe access is best.

    # Helper to get index
    # def get_point_index(p):
    #     return p.get("pointIndex", p.get("point_index", p.get("pointNumber")))

    # tabs = st.tabs([f"Point {get_point_index(p)}" for p in selected_points])

    # We need a label for each tab. extract_genotype_index returns the database index (genotype_idx),
    # but for label we might prefer the plot point index if available, or just the db index.
    # Let's use genotype_idx as the definitive label if possible, or fallback.

    def get_tab_label(p):
        idx = utils.extract_genotype_index(p, df_len=len(df))
        if idx is not None:
            return str(idx)  # Use genotype_idx as label
        # Fallback to visual index for label if DB lookup fails (unlikely given extract logic)
        return p.get("pointIndex", "?")

    tabs = st.tabs([f"Genotype {get_tab_label(p)}" for p in selected_points])

    for tab, p_data in zip(tabs, selected_points):
        with tab:
            # Robust retrieval using shared utility
            # customdata order matches hover_data in visualizations.py

            genotype_idx = utils.extract_genotype_index(p_data, df_len=len(df))
            row = None

            if genotype_idx is not None:
                row = df.iloc[genotype_idx]

            if row is None:
                st.error(f"Could not identify solution. Selection data: {p_data}")
            # Verify consistency if possible
            if (
                genotype_idx is not None
                and row is not None
                and genotype_idx != int(row["genotype_idx"])
            ):
                st.warning(
                    f"Index mismatch detected: {genotype_idx} vs {row['genotype_idx']}. Using data from customdata."
                )

            # Pass tab label as suffix to ensure uniqueness across tabs
            render_solution_details(row, result, key_suffix=f"tab_{tab._index}")


elif selected_cells:
    st.header("Selected Cell Details")
    # Heatmap selection gives x (TotalPhotons) and y (Complexity)
    # p_data['x'] and p_data['y']

    p_data = selected_cells[0]
    sel_photons = p_data["x"]
    sel_complex = p_data["y"]

    st.write(f"Cell: Complexity={sel_complex}, Total Photons={sel_photons}")

    # Filter DF for this cell
    # Note: Heatmap uses 'Desc_Complexity' and 'Desc_TotalPhotons' (floats usually, but binned/int)
    # The heatmap aggregation might have resulted in floats.
    # df columns are floats.
    # We explicitly cast to int for matching if grid uses ints

    # Let's check matching.
    # We accept small tolerance or exact match if integers.
    cell_df = df[
        (df["Desc_Complexity"].astype(int) == int(sel_complex))
        & (df["Desc_TotalPhotons"].astype(int) == int(sel_photons))
    ]

    if cell_df.empty:
        st.warning("No solutions in this cell (interpolated?).")
    else:
        st.write(f"Found {len(cell_df)} solutions in this cell.")

        # Plot local Pareto
        fig_local = px.scatter(
            cell_df,
            x="LogProb",
            y="Expectation",
            hover_data=["Expectation", "LogProb", "genotype_idx"],
            title="Local Pareto Front (In Cell)",
            labels={"LogProb": "Neg Log10 Prob", "Expectation": "Expectation"},
        )
        st.plotly_chart(fig_local, use_container_width=True, key="local_pareto_plot")

        # Show table
        st.dataframe(cell_df[["Expectation", "LogProb", "Complexity", "TotalPhotons"]])

else:
    st.info(
        "Select a point on the scatter plot or a cell on the heatmap to view details."
    )

# Best Results Summary
st.divider()
st.header("Best Results by Category")

best_exp_idx = df["Expectation"].idxmin()
best_prob_idx = df["LogProb"].idxmin()  # Minimized LogProb = Max Prob

cols = st.columns(2)
with cols[0]:
    st.subheader("Best Expectation")
    best_exp_row = df.loc[best_exp_idx]
    st.dataframe(best_exp_row.to_frame().T)
    if st.button("View Best Expectation Details"):
        render_solution_details(best_exp_row, result, key_suffix="btn_best_exp")

with cols[1]:
    st.subheader("Best Probability")
    best_prob_row = df.loc[best_prob_idx]
    st.dataframe(best_prob_row.to_frame().T)
    if st.button("View Best Probability Details"):
        render_solution_details(best_prob_row, result, key_suffix="btn_best_prob")
