import streamlit as st
import numpy as np

# pandas removed
import visualizations as viz
import utils
import plotly.express as px
from src.utils.result_manager import SimpleRepertoire

# Pickle Compatibility Hack:
# Old pickles expect SimpleRepertoire to be in __main__.
# Since this script runs as __main__ in Streamlit, aliasing it here works.
globals()["SimpleRepertoire"] = SimpleRepertoire

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


def render_gbs_optimization(st, gauss_res, genotype_idx, key_suffix, go):
    """Run and display the Hanamura (PRX 2026) two-step optimization of the
    GBS generator equivalent to the selected solution."""
    import numpy as np

    st.divider()
    st.subheader("Optimized GBS Architecture (Hanamura PRX 16, 021034)")
    st.markdown(
        "Treating the circuit above as a multimode non-Gaussian state generator "
        "$(C,\\beta,n)$, we (1) **reduce the detected photon numbers** via wave-form "
        "matching and (2) **maximize the success probability** via the damping "
        "transform — both modifying only the *Gaussian* part while preserving the "
        "heralded output state (up to a Gaussian unitary, absorbed by the final "
        "Gaussian operation on the signal)."
    )

    cov = gauss_res.get("cov")
    mu = gauss_res.get("mu")
    control_idx = gauss_res.get("control_idx") or []
    signal_idx = gauss_res.get("signal_idx")
    pnr = gauss_res.get("pnr_outcomes") or []

    if cov is None or signal_idx is None or len(control_idx) == 0:
        st.warning("No control (PNR) modes available to optimize for this solution.")
        return
    if sum(int(x) for x in pnr) == 0:
        st.info("All control modes detect 0 photons (Gaussian output) — nothing to reduce.")
        return

    factor = st.slider(
        "Photon-reduction factor (target ≈ n / factor, parity preserved)",
        min_value=1.5, max_value=5.0, value=3.0, step=0.5,
        key=f"gbs_factor_{genotype_idx}_{key_suffix}",
        help="Higher factor = more aggressive photon-number reduction. Output "
             "fidelity typically drops and required squeezing rises as the "
             "reduction becomes more aggressive.",
    )

    cache_key = f"gbs_opt_{genotype_idx}_{key_suffix}_{factor}"
    if cache_key not in st.session_state:
        with st.spinner("Optimizing GBS architecture (control-parameter method)…"):
            st.session_state[cache_key] = go.optimize_gbs_architecture(
                np.asarray(cov), np.asarray(mu), int(signal_idx),
                list(control_idx), list(pnr),
                reduction_factor=float(factor), verify=True,
            )
    res = st.session_state[cache_key]

    # --- headline metrics --------------------------------------------------
    ver = res.get("verification", {})
    fid = ver.get("output_fidelity")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected photons", f"{res['total_photons_after']}",
              delta=f"{res['total_photons_after'] - res['total_photons_before']} vs {res['total_photons_before']}")
    gain = res["prob_gain"]
    c2.metric("Success probability", f"{res['prob_after']:.2e}",
              delta=f"×{gain:.3g}" if np.isfinite(gain) else "↑")
    if fid is not None:
        c3.metric("Output fidelity (up to Gaussian U)", f"{fid * 100:.2f}%")
    else:
        c3.metric("Output fidelity", "n/a")

    st.caption(
        f"Success probability: {res['prob_before']:.2e} → "
        f"{res['prob_after_step1']:.2e} (after photon reduction) → "
        f"{res['prob_after']:.2e} (after probability maximization)."
    )

    # --- non-Gaussian control parameters ----------------------------------
    import pandas as pd
    rows = []
    for m in range(len(control_idx)):
        pb = res["params_before"][m]
        pa = res["params_after"][m]
        rows.append({
            "Control mode": gauss_res["modes"][control_idx[m]]["id"]
            if control_idx[m] < len(gauss_res["modes"]) else f"mode {m}",
            "n → n'": f"{res['n_before'][m]} → {res['n_after'][m]}",
            "s₀ → s₀'": f"{pb['s0']:.3f} → {pa['s0']:.3f}",
            "|δ₀| → |δ₀'|": f"{abs(pb['delta0']):.3f} → {abs(pa['delta0']):.3f}",
        })
    st.markdown("**Non-Gaussian control parameters** (per control mode)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- optimized architecture -------------------------------------------
    arch = res["architecture"]
    st.markdown("**Optimized generator** — vacuum → squeezers → interferometer → "
                "displacements → PNR")
    sq_str = ", ".join(
        f"r={r:.3f} ({db:.2f} dB)"
        for r, db in zip(arch["squeezings_r"], arch["squeezings_db"])
    )
    st.info(f"**Required squeezings:** {sq_str}")
    st.caption(
        f"PNR detection pattern: {arch['pnr_outcomes']} "
        f"(was {res['n_before']}). Max squeezing {arch['max_squeezing_db']:.2f} dB — "
        "this is the experimental cost of the probability gain; lower the "
        "reduction factor for a milder trade-off."
    )

    with st.expander("Interferometer & displacements of the optimized generator"):
        U = arch["U_passive"]
        try:
            import plotly.express as px
            fig_U = px.imshow(
                np.abs(U),
                labels=dict(x="Input (squeezer)", y="Output mode", color="|U|"),
                color_continuous_scale="Blues",
                title="Optimized interferometer amplitudes |U|",
            )
            st.plotly_chart(fig_U, use_container_width=True,
                            key=f"gbs_U_{genotype_idx}_{key_suffix}_{factor}")
        except Exception:
            st.write(np.round(np.abs(U), 4))
        st.markdown("**Displacements $(\\mu_x, \\mu_p)$ per mode:**")
        for i, (mx, mp) in enumerate(arch["displacements"]):
            role = "signal" if i in arch["signal_idx"] else "control"
            st.text(f"Mode {i} ({role}): x={mx:.4f}, p={mp:.4f}")

    # --- verification ------------------------------------------------------
    with st.expander("Verification", expanded=True):
        if fid is not None:
            ok = fid > 0.99
            (st.success if ok else st.warning)(
                f"Heralded output state preserved with fidelity **{fid*100:.3f}%** "
                f"(up to a Gaussian unitary), simulated at Fock cutoff "
                f"{ver.get('herald_cutoff')}."
            )
        elif ver.get("fidelity_skipped"):
            st.info("Exact output-fidelity simulation skipped (too many modes): "
                    + ver["fidelity_skipped"])
        st.markdown(
            f"- Step 2 output invariance: {ver.get('step2_output_invariant')}\n"
            f"- Optimized generator is a valid Gaussian state "
            f"(C ≥ iΩ): **{ver.get('optimized_generator_valid')}**\n"
            f"- Damping parameters t = {np.round(res['damping']['t'], 3).tolist()}"
        )


def render_solution_details(row, result_obj, key_suffix=""):
    genotype_idx = int(row["genotype_idx"])

    # Metrics
    # Reconstruct Circuit first to access params
    params = result_obj.get_circuit_params(genotype_idx)

    # Recalculate metrics for display (Active Leaves Only)
    # The stored 'row' metrics might count inactive leaves depending on optimizing code.
    # We enforce frontend correctness here.
    active_total_photons, active_max_pnr = utils.compute_active_metrics(params)

    # Metrics (Safe Formatting)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Expectation", f"{utils.to_scalar(row['Expectation']):.4f}")
    m_col2.metric("LogProb", f"{utils.to_scalar(row['LogProb']):.4f}")
    m_col3.metric("Complexity", f"{utils.to_scalar(row['Complexity']):.2f}")
    m_col4.metric("Total Photons (Active)", f"{active_total_photons:.2f}")

    # Circuit Plot
    # st.subheader("Active Leaf Schematic (Topology)")
    # fig_circ = utils.get_circuit_figure(params)
    # st.pyplot(fig_circ)

    # Wigner Function
    st.subheader("Wigner Function (Signal Mode)")
    # st.subheader("Wigner Function (Signal Mode)") # This subheader is now inside col_viz2

    with st.spinner("Computing Wigner function..."):
        try:
            # Compute state
            # Compute state
            # Determine Simulation Cutoff (Dynamic Limits Check)
            # Backend may use correction_cutoff for internal simulation to reduce leakage, then truncate.
            # To match stored probability, we must simulate at correction_cutoff.
            base_cutoff = int(result_obj.config.get("cutoff", 12))
            corr_cutoff = result_obj.config.get("correction_cutoff")

            sim_cutoff = base_cutoff
            if corr_cutoff is not None:
                cc = int(corr_cutoff)
                if cc > base_cutoff:
                    sim_cutoff = cc

            cutoff = base_cutoff
            pnr_max = int(result_obj.config.get("pnr_max", 3))  # Use config or default

            params["n_control"] = params.get(
                "n_control", 3
            )  # Default to 3 (standard GG) if missing
            params["pnr_outcome"] = params.get(
                "pnr_outcome", [0] * int(params.get("n_control", 1))
            )  # Ensure pnr present
            psi, prob = utils.compute_heralded_state(
                params, cutoff=sim_cutoff, pnr_max=pnr_max
            )
            # Note: psi will be size sim_cutoff (e.g. 30), not base_cutoff (12).
            # Wigner and Plotting tools handle this fine.
            exp_val = row["Expectation"]  # Get expectation for Wigner title

            # --- VISUALIZATION ROW ---
            # Use 3 columns: Tree | Output Wigner | Target Wigner
            col_viz1, col_viz2, col_viz3 = st.columns(3)

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

            stored_log_prob_val = utils.to_scalar(row["LogProb"])
            # Stored LogProb is likely NegLog10(P) (positive number) already if coming from fitness/metrics post-processing.
            # Empirical evidence: Val=17.59.
            stored_neg_log_prob = stored_log_prob_val

            st.text(
                f"Herald Probability (Simulated): {prob:.4e}  => NegLog10 P: {sim_log_prob:.4f}"
            )
            st.text(
                f"Stored LogProb (Backend):       {stored_log_prob_val:.4f} => NegLog10 P: {stored_neg_log_prob:.4f}"
            )

            if abs(sim_log_prob - stored_neg_log_prob) > 1.0:
                st.warning(
                    "⚠️ Mismatch between stored metadata and re-simulated state. "
                    "This likely due to dynamic limits or different cutoff settings. "
                    "The Wigner function shown is correct for the re-simulated parameters."
                )

            # Common Wigner Grid
            # User requested "high resolution"
            grid_size = 200
            limit = 5
            xvec = np.linspace(-limit, limit, grid_size)
            pvec = np.linspace(-limit, limit, grid_size)

            with col_viz2:
                st.subheader("Output Wigner Function")
                W = utils.compute_wigner(psi, xvec, pvec)

                # Title with Simulated values
                fig_wig = viz.plot_wigner_function(
                    W,
                    xvec,
                    pvec,
                    title=f"Output State<br>Exp: {exp_val:.4f} | Prob: {prob:.2e}",
                )
                st.plotly_chart(
                    fig_wig,
                    use_container_width=True,
                    key=f"wigner_{genotype_idx}_{key_suffix}",
                )

            with col_viz3:
                st.subheader("Target Ground State")

                # Retrieve Target Params
                try:
                    from src.utils.gkp_operator import construct_gkp_operator
                    import re

                    # Strategy: Try Config first, then Folder Name override?
                    # User said "should be in folder name yk" implying folder name is authoritative or convenient.

                    # Defaults
                    t_alpha_str = str(run_config.get("target_alpha", "2.0"))
                    t_beta_str = str(run_config.get("target_beta", "0.0"))

                    # Parse from Folder Name
                    # Expected pattern: ..._alpha_2.0_beta_0.0...
                    # or ..._alpha2.0_beta0.0...
                    # Regex for alpha
                    alpha_match = re.search(r"alpha_?([0-9\.]+)", selected_run_dir)
                    if alpha_match:
                        t_alpha_str = alpha_match.group(1)

                    # Regex for beta
                    beta_match = re.search(r"beta_?([0-9\.]+)", selected_run_dir)
                    if beta_match:
                        t_beta_str = beta_match.group(1)

                    st.caption(f"Target: α={t_alpha_str}, β={t_beta_str}")

                    # Safe parse complex
                    t_alpha = (
                        complex(t_alpha_str)
                        if "j" in t_alpha_str
                        else float(t_alpha_str)
                    )
                    t_beta = (
                        complex(t_beta_str) if "j" in t_beta_str else float(t_beta_str)
                    )

                    # Construct Operator
                    # Use same cutoff as simulation
                    target_cutoff = cutoff
                    op_matrix = construct_gkp_operator(
                        target_cutoff, t_alpha, t_beta, backend="thewalrus"
                    )

                    # Find Ground State (Lowest Eigenvalue of GKP Hamiltonian)
                    # Note: construct_gkp_operator returns the Hamiltonian matrix.
                    vals, vecs = np.linalg.eigh(op_matrix)
                    ground_state = vecs[:, 0]  # Lowest energy state

                    # Compute Wigner
                    W_target = utils.compute_wigner(ground_state, xvec, pvec)

                    fig_target = viz.plot_wigner_function(
                        W_target,
                        xvec,
                        pvec,
                        title=f"Target Ground State (Cutoff {cutoff})<br>Eigenvalue: {vals[0]:.4f}",
                    )
                    st.plotly_chart(
                        fig_target,
                        use_container_width=True,
                        key=f"target_wigner_{genotype_idx}_{key_suffix}",
                    )
                except Exception as e:
                    st.error(f"Error computing target Wigner: {e}")

            # --- Equivalent Gaussian Circuit ---
            try:
                import importlib
                import frontend.gaussian_decomposition
                importlib.reload(frontend.gaussian_decomposition)
                from frontend.gaussian_decomposition import compute_equivalent_gaussian
                
                gauss_res = compute_equivalent_gaussian(params)
                
                st.divider()
                st.subheader("Equivalent Gaussian Decomposition (GBS Equivalent)")
                st.markdown(
                    f"The active optical circuit exactly reduces to an equivalent **{gauss_res['num_final_modes']}-mode GBS circuit** applied to vacuum. "
                    f"Follow the sequential steps below to reproduce the state:"
                )
                
                # Step 1
                st.markdown("### Step 1: Initial Squeezing")
                st.markdown(f"Apply single-mode squeezers $S(r)$ to each of the {gauss_res['num_final_modes']} vacuum modes.")
                
                # Format squeezings in r and dB
                squeezings_formatted = []
                for r in gauss_res['squeezings_r']:
                    r_db = r * 10 * np.log10(np.exp(2))
                    squeezings_formatted.append(f"r={r:.3f} ({r_db:.2f} dB)")
                    
                st.info("**Required Squeezings:** " + ", ".join(squeezings_formatted))
                
                # Step 2
                st.markdown("### Step 2: Passive Interferometer")
                st.markdown("Apply the linear optical network defined by the unitary matrix $U$.")
                with st.expander("View Unitary Matrix $U$ & Amplitudes Heatmap"):
                    U = gauss_res["U_passive"]
                    import pandas as pd
                    formatted_U = []
                    for i in range(U.shape[0]):
                        row = []
                        for j in range(U.shape[1]):
                            z = U[i, j]
                            r = 0.0 if abs(z.real) < 1e-10 else z.real
                            i_val = 0.0 if abs(z.imag) < 1e-10 else z.imag
                            if i_val >= 0:
                                row.append(f"{r:.4f} + {i_val:.4f}j")
                            else:
                                row.append(f"{r:.4f} - {abs(i_val):.4f}j")
                        formatted_U.append(row)
                        
                    df_U = pd.DataFrame(formatted_U, 
                                        index=[f"Out {i}" for i in range(U.shape[0])],
                                        columns=[f"In {j}" for j in range(U.shape[1])])
                    st.dataframe(df_U, use_container_width=True)
                    
                    import plotly.express as px
                    amplitudes = np.abs(U)
                    fig_U = px.imshow(amplitudes, 
                                      labels=dict(x="Input Mode (from Squeezers)", y="Output Mode", color="Amplitude |U|"),
                                      x=[f"{j}" for j in range(U.shape[1])],
                                      y=[f"{i}" for i in range(U.shape[0])],
                                      color_continuous_scale="Blues",
                                      title="Interferometer Amplitudes |U|")
                    st.plotly_chart(fig_U, use_container_width=True)

                # Step 3
                st.markdown("### Step 3: Displacements")
                st.markdown("Apply phase-space displacements to the output modes.")
                with st.expander("View Displacements per Mode"):
                    st.markdown("**Final Mode Displacements $(\\mu_x, \\mu_p)$:**")
                    for i, (mx, mp) in enumerate(gauss_res['final_mu']):
                        mode_info = gauss_res['modes'][i]
                        st.text(f"Mode {i} ({mode_info['id']}): x={mx:.4f}, p={mp:.4f}")
                        
                # Step 4
                st.markdown("### Step 4: PNR Measurements")
                st.markdown("Perform Photon-Number Resolving (PNR) measurements on the control modes.")
                signal_mode = None
                for i, mode_info in enumerate(gauss_res['modes']):
                    if mode_info['type'] == 'control':
                        st.success(f"Mode {i} ({mode_info['id']}): Detect **{mode_info.get('pnr_val', '?')}** photons")
                    else:
                        signal_mode = f"Mode {i} ({mode_info['id']})"
                        
                # Output
                st.markdown("### Output: Heralded State")
                st.info(f"The remaining unmeasured mode is the output state: **{signal_mode}**")

                # --- Optimized GBS Architecture (Hanamura PRX 2026) ---
                try:
                    import frontend.gbs_optimizer as gbs_optimizer
                    importlib.reload(gbs_optimizer)
                    render_gbs_optimization(
                        st, gauss_res, genotype_idx, key_suffix, gbs_optimizer
                    )
                except Exception as opt_err:
                    import traceback
                    st.divider()
                    st.subheader("Optimized GBS Architecture (Hanamura)")
                    st.error(f"Could not optimize GBS architecture: {opt_err}")
                    st.code(traceback.format_exc())

            except Exception as e:
                import traceback
                st.error(f"Error computing Gaussian decomposition: {e}")
                st.code(traceback.format_exc())

            st.divider()

            # --- Independent Verification (Multi-Cutoff) ---
            # Use session_state to persist results across Streamlit reruns
            iv_key = f"iv_result_{genotype_idx}_{key_suffix}"

            # Determine the three cutoffs
            cutoff_opt = cutoff  # optimization cutoff
            cutoff_ver = sim_cutoff  # verification cutoff (correction_cutoff)
            cutoff_ext = sim_cutoff + 15  # extended cutoff
            # Deduplicate (e.g. if no correction_cutoff, opt == ver)
            cutoff_levels = list(dict.fromkeys([cutoff_opt, cutoff_ver, cutoff_ext]))
            cutoff_labels = {
                cutoff_opt: "Optimization",
                cutoff_ver: "Verification",
                cutoff_ext: "Extended (+5)",
            }

            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                if st.button(
                    f"🔬 Verify at {len(cutoff_levels)} cutoffs: {cutoff_levels}",
                    key=f"verify_btn_{genotype_idx}_{key_suffix}",
                ):
                    with st.spinner(
                        "Running independent verification at multiple cutoffs..."
                    ):
                        try:
                            from frontend.independent_verifier import verify_circuit

                            results_per_cutoff = {}
                            for c in cutoff_levels:
                                label = cutoff_labels.get(c, f"c={c}")
                                iv_result = verify_circuit(
                                    params, cutoff=c, pnr_max=pnr_max
                                )
                                iv_state = iv_result["state"]
                                psi_c, prob_c = utils.compute_heralded_state(
                                    params, cutoff=c, pnr_max=pnr_max
                                )
                                overlap = float(np.abs(np.vdot(psi_c, iv_state)) ** 2)
                                W_jax_c = utils.compute_wigner(psi_c, xvec, pvec)
                                W_iv_c = utils.compute_wigner(iv_state, xvec, pvec)

                                results_per_cutoff[c] = {
                                    "label": label,
                                    "fidelity": overlap,
                                    "jax_prob": float(prob_c),
                                    "iv_prob": iv_result["probability"],
                                    "W_jax": W_jax_c,
                                    "W_iv": W_iv_c,
                                    "report": iv_result["report"],
                                }

                            st.session_state[iv_key] = {
                                "cutoffs": cutoff_levels,
                                "results": results_per_cutoff,
                                "exp_val": exp_val,
                            }
                        except Exception as e:
                            import traceback

                            st.session_state[iv_key] = {
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                            }

            with btn_col2:
                if iv_key in st.session_state and st.button(
                    "✕ Clear",
                    key=f"clear_iv_{genotype_idx}_{key_suffix}",
                ):
                    del st.session_state[iv_key]
                    st.rerun()

            # Display stored results
            if iv_key in st.session_state:
                iv_data = st.session_state[iv_key]

                if "error" in iv_data:
                    st.error(f"Independent verification failed: {iv_data['error']}")
                    st.code(iv_data["traceback"])
                else:
                    cutoffs = iv_data["cutoffs"]
                    res = iv_data["results"]

                    # Summary metrics row
                    st.subheader("Cutoff Convergence Summary")
                    summary_cols = st.columns(len(cutoffs))
                    for col, c in zip(summary_cols, cutoffs):
                        r = res[c]
                        fid = r["fidelity"]
                        if fid > 0.99:
                            fid_icon = "✅"
                        elif fid > 0.90:
                            fid_icon = "⚠️"
                        else:
                            fid_icon = "❌"
                        col.metric(
                            f"{r['label']} (c={c})",
                            f"{fid_icon} F={fid:.6f}",
                        )
                        col.caption(
                            f"JAX: {r['jax_prob']:.3e} | IV: {r['iv_prob']:.3e}"
                        )

                    # Per-cutoff Wigner comparisons in tabs
                    wigner_tabs = st.tabs(
                        [f"c={c} ({res[c]['label']})" for c in cutoffs]
                    )
                    for w_tab, c in zip(wigner_tabs, cutoffs):
                        r = res[c]
                        with w_tab:
                            iv_col1, iv_col2 = st.columns(2)
                            with iv_col1:
                                st.caption(f"JAX (cutoff={c})")
                                fig_jax = viz.plot_wigner_function(
                                    r["W_jax"],
                                    xvec,
                                    pvec,
                                    title=f"JAX c={c}<br>P={r['jax_prob']:.3e}",
                                )
                                st.plotly_chart(
                                    fig_jax,
                                    use_container_width=True,
                                    key=f"iv_jax_{genotype_idx}_{c}_{key_suffix}",
                                )
                            with iv_col2:
                                st.caption(f"Independent (cutoff={c})")
                                fig_iv = viz.plot_wigner_function(
                                    r["W_iv"],
                                    xvec,
                                    pvec,
                                    title=f"Indep c={c}<br>P={r['iv_prob']:.3e}",
                                )
                                st.plotly_chart(
                                    fig_iv,
                                    use_container_width=True,
                                    key=f"iv_indep_{genotype_idx}_{c}_{key_suffix}",
                                )

                    # Diagnostics for highest cutoff
                    highest_c = cutoffs[-1]
                    iv_report = res[highest_c]["report"]
                    with st.expander(f"📋 Diagnostics (cutoff={highest_c})"):
                        st.write("**Per-Leaf Results:**")
                        for leaf in iv_report["leaves"]:
                            status = "✅" if leaf["active"] else "🔴"
                            st.text(
                                f"  Leaf {leaf['index']} [{status}]: "
                                f"norm={leaf['state_norm']:.6f}, "
                                f"prob={leaf['prob']:.4e}"
                                + (
                                    f", PNR={leaf.get('pnr', [])}"
                                    if leaf["active"]
                                    else ""
                                )
                            )

                        st.write("**Mixing Nodes:**")
                        for node in iv_report["mixing_nodes"]:
                            st.text(
                                f"  Node {node['node']} (L{node['layer']}): "
                                f"BS(θ={node['theta']:.3f}, φ={node['phi']:.3f}), "
                                f"x={node['hx']:.3f}, "
                                f"p_hom={node['p_homodyne']:.4e}, "
                                f"out_norm={node['output_norm']:.6f}"
                            )

                        st.write(
                            f"**Final state norm:** {iv_report['final_state_norm']:.6f}"
                        )

                        if iv_report["warnings"]:
                            st.write("**Warnings:**")
                            for w in iv_report["warnings"]:
                                st.warning(w)

            # Rank-Matched Plot Row
            if active_total_photons > 1:
                st.subheader(
                    f"Rank-Matched Target (Dimension {int(active_total_photons)})"
                )
                with st.spinner("Computing Rank-Matched Target..."):
                    try:
                        rank_cutoff = int(active_total_photons)
                        # Ensure valid dimension (at least 2)
                        rank_cutoff = max(2, rank_cutoff)

                        # Construct Operator
                        op_rank = construct_gkp_operator(
                            rank_cutoff, t_alpha, t_beta, backend="thewalrus"
                        )

                        # Ground State
                        vals_r, vecs_r = np.linalg.eigh(op_rank)
                        ground_state_r = vecs_r[:, 0]

                        # Wigner
                        W_rank = utils.compute_wigner(ground_state_r, xvec, pvec)

                        fig_rank = viz.plot_wigner_function(
                            W_rank,
                            xvec,
                            pvec,
                            title=f"Rank{rank_cutoff} Target<br>Dim: {rank_cutoff} | Eigenvalue: {vals_r[0]:.4f}",
                        )

                        # Center or full width?
                        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
                        with col_r2:
                            st.plotly_chart(
                                fig_rank,
                                use_container_width=True,
                                key=f"rank_wigner_{genotype_idx}_{key_suffix}",
                            )
                    except Exception as e:
                        st.warning(f"Could not compute Rank-Matched Target: {e}")

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

# Persist Plotly selection in session_state so button clicks don't lose it
if selected_points:
    # New scatter selection — store it (clears heatmap selection)
    resolved = []
    for p_data in selected_points:
        gidx = utils.extract_genotype_index(p_data, df_len=len(df))
        if gidx is not None:
            resolved.append(gidx)
    if resolved:
        st.session_state["selected_genotype_indices"] = resolved
elif selected_cells:
    # Heatmap selection — clear scatter selection
    st.session_state.pop("selected_genotype_indices", None)

# Use persisted selection for scatter points
active_indices = st.session_state.get("selected_genotype_indices", [])

if active_indices:
    st.header("Selected Solution Details")

    tabs = st.tabs([f"Genotype {idx}" for idx in active_indices])

    for tab_idx, (tab, genotype_idx) in enumerate(zip(tabs, active_indices)):
        with tab:
            row = df.iloc[genotype_idx]
            render_solution_details(row, result, key_suffix=f"tab_{tab_idx}")

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
        st.session_state["show_best_exp"] = not st.session_state.get(
            "show_best_exp", False
        )
    if st.session_state.get("show_best_exp", False):
        render_solution_details(best_exp_row, result, key_suffix="btn_best_exp")

with cols[1]:
    st.subheader("Best Probability")
    best_prob_row = df.loc[best_prob_idx]
    st.dataframe(best_prob_row.to_frame().T)
    if st.button("View Best Probability Details"):
        st.session_state["show_best_prob"] = not st.session_state.get(
            "show_best_prob", False
        )
    if st.session_state.get("show_best_prob", False):
        render_solution_details(best_prob_row, result, key_suffix="btn_best_prob")
