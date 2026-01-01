import streamlit as st
import numpy as np
import plotly.express as px
import jax.numpy as jnp
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Backend Components
try:
    from src.simulation.jax.runner import jax_get_heralded_state, jax_clements_unitary
    from src.simulation.jax.composer import (
        jax_compose_pair,
        jax_u_bs,
        jax_hermite_phi_matrix,
    )

    import frontend.visualizations as viz
    import frontend.utils as utils

    JAX_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import backend components: {e}")
    JAX_AVAILABLE = False


st.set_page_config(page_title="Experimentation Deck", layout="wide")

st.title("🧪 Experimentation Deck")
st.markdown("Interactive playground for quantum optical state preparation and mixing.")

if not JAX_AVAILABLE:
    st.stop()

# --- Sidebar Handlers ---
st.sidebar.header("Global Settings")
cutoff = st.sidebar.slider("Fock Cutoff", min_value=5, max_value=30, value=12)
hbar = st.sidebar.number_input("hbar", value=2.0, min_value=0.1, step=0.1)

# --- Helper Functions ---


def parse_fock_input(text_input: str, dim: int) -> np.ndarray:
    """Parse comma-separated complex coeffs."""
    try:
        parts = text_input.split(",")
        coeffs = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            coeffs.append(complex(p))

        vec = np.zeros(dim, dtype=complex)
        L = min(len(coeffs), dim)
        vec[:L] = coeffs[:L]

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec
    except Exception:
        return np.zeros(dim, dtype=complex)


def run_simulation(config_a: Dict, config_b: Dict, mix_config: Dict, cutoff_dim: int):
    """
    Run the full simulation using JAX backend.
    """

    # 1. Generate Source A
    if config_a["type"] == "Fock State":
        vec_a = parse_fock_input(config_a["fock_coeffs"], cutoff_dim)
        prob_a = 1.0
    else:
        p_a = config_a["params"]
        # Extract General Gaussian params
        n_ctrl = p_a["n_control"]
        modes = n_ctrl + 1

        leaf_params_a = {
            "n_ctrl": jnp.array(n_ctrl),
            "r": jnp.array(p_a["r"]),  # (N,)
            "phases": jnp.array(p_a["phases"]),  # (N^2,)
            "disp": jnp.array(p_a["disp"]),  # (N,) complex
            "pnr": jnp.array(p_a["pnr"]),  # (N-1,)
        }

        vec_a_jax, prob_a_jax, _, _, _, _ = jax_get_heralded_state(
            leaf_params_a, cutoff_dim, pnr_max=int(p_a.get("max_pnr", 3))
        )
        vec_a = np.array(vec_a_jax)
        prob_a = float(prob_a_jax)

    # 2. Generate Source B (Mirror A)
    if config_b["type"] == "Fock State":
        vec_b = parse_fock_input(config_b["fock_coeffs"], cutoff_dim)
        prob_b = 1.0
    else:
        p_b = config_b["params"]
        n_ctrl = p_b["n_control"]

        leaf_params_b = {
            "n_ctrl": jnp.array(n_ctrl),
            "r": jnp.array(p_b["r"]),
            "phases": jnp.array(p_b["phases"]),
            "disp": jnp.array(p_b["disp"]),
            "pnr": jnp.array(p_b["pnr"]),
        }
        vec_b_jax, prob_b_jax, _, _, _, _ = jax_get_heralded_state(
            leaf_params_b, cutoff_dim, pnr_max=int(p_b.get("max_pnr", 3))
        )
        vec_b = np.array(vec_b_jax)
        prob_b = float(prob_b_jax)

    # 3. Mixing / Pass
    mode = mix_config["mode"]

    vec_out = vec_a
    prob_joint = prob_a * prob_b

    if mode == "Pass A":
        vec_out = vec_a
        prob_out = 1.0
        prob_joint = prob_a
    elif mode == "Pass B":
        vec_out = vec_b
        prob_out = 1.0
        prob_joint = prob_b
    else:
        # Mix
        theta = mix_config["theta"]
        phi = mix_config["phi"]
        _hom_angle = mix_config["hom_angle"]
        hom_val = mix_config["hom_val"]  # Homodyne X

        state_A_jax = jnp.array(vec_a)
        state_B_jax = jnp.array(vec_b)

        U = jax_u_bs(theta, phi, cutoff_dim)

        hom_xs = jnp.atleast_1d(jnp.array(hom_val))
        phi_mat = jax_hermite_phi_matrix(hom_xs, cutoff_dim)
        phi_vec = phi_mat[:, 0]

        res_val = 1.0

        out_jax, prob_meas, joint_jax = jax_compose_pair(
            state_A_jax,
            state_B_jax,
            U,
            prob_a,
            prob_b,
            jnp.array(hom_val),
            0.0,
            res_val,
            phi_vec,
            None,
            None,
            cutoff_dim,
            homodyne_window_is_none=True,
            homodyne_x_is_none=False,
            homodyne_resolution_is_none=True,
            theta=theta,
            phi=phi,
        )

        vec_out = np.array(out_jax)
        prob_out = float(prob_meas)
        prob_joint = float(joint_jax)

    return {
        "state_a": vec_a,
        "prob_a": prob_a,
        "state_b": vec_b,
        "prob_b": prob_b,
        "state_out": vec_out,
        "prob_out": prob_out,
        "prob_joint": prob_joint,
    }


def create_gaussian_ui(key_suffix: str):
    """
    UI for General Gaussian Params (N modes).
    Returns params dict.
    """
    nc = st.number_input(f"N Control ({key_suffix})", 1, 5, 1, key=f"nc_{key_suffix}")
    modes = nc + 1

    st.markdown(f"**Total Modes: {modes} (1 Signal + {nc} Control)**")

    with st.expander("Squeezing (r)", expanded=True):
        # Allow comma separated input
        r_str = st.text_input(
            f"r vector (len {modes})",
            "0.0, " * (modes - 1) + "0.0",
            key=f"r_{key_suffix}",
        )
        try:
            r_vals = [float(x) for x in r_str.split(",")]
        except ValueError:
            r_vals = [0.0] * modes
        # Pad/Truncate
        if len(r_vals) < modes:
            r_vals += [0.0] * (modes - len(r_vals))
        r_vals = r_vals[:modes]

    with st.expander("Phases (Clements)", expanded=False):
        n_phases = modes * modes
        st.caption(f"Requires {n_phases} phases for U({modes})")
        ph_str = st.text_area(
            f"Phases (len {n_phases})",
            "0.0, " * (n_phases - 1) + "0.0",
            key=f"ph_{key_suffix}",
        )
        try:
            ph_vals = [float(x) for x in ph_str.replace("\n", ",").split(",")]
        except ValueError:
            ph_vals = [0.0] * n_phases
        if len(ph_vals) < n_phases:
            ph_vals += [0.0] * (n_phases - len(ph_vals))
        ph_vals = ph_vals[:n_phases]

    with st.expander("Displacement (alpha)", expanded=False):
        st.caption(f"Requires {modes} complex vals")
        dip_str = st.text_input(
            f"Disp (complex)", "0.0, " * (modes - 1) + "0.0", key=f"d_{key_suffix}"
        )
        try:
            d_vals = []
            for x in dip_str.split(","):
                d_vals.append(complex(x.strip()))
        except ValueError:
            d_vals = [0j] * modes
        if len(d_vals) < modes:
            d_vals += [0j] * (modes - len(d_vals))
        d_vals = d_vals[:modes]

    with st.expander("PNR Outcomes", expanded=True):
        st.caption(f"Requires {nc} integers (Control Modes 1..{nc})")
        pnr_str = st.text_input(f"PNR", "0, " * (nc - 1) + "0", key=f"pnr_{key_suffix}")
        try:
            pnr_vals = [int(x) for x in pnr_str.split(",")]
        except ValueError:
            pnr_vals = [0] * nc
        if len(pnr_vals) < nc:
            pnr_vals += [0] * (nc - len(pnr_vals))
        pnr_vals = pnr_vals[:nc]

    return {
        "n_control": nc,
        "r": r_vals,
        "phases": ph_vals,
        "disp": d_vals,
        "pnr": pnr_vals,
    }


# --- UI Construction ---

col_in_a, col_mid, col_in_b = st.columns([1, 0.8, 1])

# --- Source A ---
with col_in_a:
    st.subheader("Source A")
    type_a = st.radio("Type A", ["Gaussian Circuit", "Fock State"], key="type_a")

    config_a = {"type": type_a}

    if type_a == "Fock State":
        f_in = st.text_input("Fock Coeffs (0, 1...)", "0, 1", key="fock_a")
        config_a["fock_coeffs"] = f_in
    else:
        # General Gaussian UI
        config_a["params"] = create_gaussian_ui("A")

# --- Source B ---
with col_in_b:
    st.subheader("Source B")
    type_b = st.radio("Type B", ["Gaussian Circuit", "Fock State"], key="type_b")

    config_b = {"type": type_b}

    if type_b == "Fock State":
        f_in = st.text_input("Fock Coeffs (0, 1...)", "1", key="fock_b")
        config_b["fock_coeffs"] = f_in
    else:
        config_b["params"] = create_gaussian_ui("B")

# --- Mixing Section ---
with col_mid:
    st.subheader("Mixing / Operation")
    op_mode = st.radio("Operation", ["Mix (Beam Splitter)", "Pass A", "Pass B"])

    mix_config = {"mode": op_mode}

    if op_mode == "Mix (Beam Splitter)":
        bs_theta = st.slider("BS Theta", 0.0, np.pi, np.pi / 4, key="bs_th")
        bs_phi = st.slider("BS Phi", 0.0, 2 * np.pi, 0.0, key="bs_ph")

        st.markdown("---")
        st.markdown("**Measurement**")
        hom_angle = st.slider(
            "Homodyne Angle",
            0.0,
            2 * np.pi,
            0.0,
            key="hom_ang",
        )
        hom_val = st.number_input(
            "Homodyne X Outcome", value=0.0, step=0.1, key="hom_val"
        )

        mix_config.update(
            {
                "theta": bs_theta,
                "phi": bs_phi,
                "hom_angle": hom_angle,
                "hom_val": hom_val,
            }
        )

# --- Execution ---
st.divider()
if st.button("Run Experiment", type="primary"):
    with st.spinner("Simulating..."):
        try:
            res = run_simulation(config_a, config_b, mix_config, cutoff)

            # --- Results ---
            st.success("Simulation Complete!")

            # Probabilities
            c1, c2, c3 = st.columns(3)
            c1.metric("Prob A (Source)", f"{res['prob_a']:.4e}")
            c2.metric("Prob B (Source)", f"{res['prob_b']:.4e}")
            c3.metric("Joint Prob", f"{res['prob_joint']:.4e}")

            # --- Visualizations ---
            xvec = np.linspace(-5, 5, 200)
            pvec = np.linspace(-5, 5, 200)

            vrow1, vrow2, vrow3 = st.columns(3)

            def show_wig(psi, title, key, col):
                W = utils.compute_wigner(psi, xvec, pvec)
                fig = viz.plot_wigner_function(W, xvec, pvec, title=title)
                col.plotly_chart(fig, use_container_width=True, key=key)

            with vrow1:
                show_wig(res["state_a"], "Source A", "wig_a", st)
            with vrow2:
                show_wig(res["state_b"], "Source B", "wig_b", st)
            with vrow3:
                show_wig(res["state_out"], "Output State", "wig_out", st)

            # Fock Dist
            st.subheader("Output Fock Distribution")
            psi_out = res["state_out"]

            if psi_out.ndim == 1:
                n_states = len(psi_out)
                x = np.arange(n_states)
                re = np.real(psi_out)
                im = np.imag(psi_out)
                probs = np.abs(psi_out) ** 2

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Amplitudes (Real/Imag)", "Probability"),
                )
                fig.add_trace(
                    go.Bar(name="Real", x=x, y=re, marker_color="blue"), row=1, col=1
                )
                fig.add_trace(
                    go.Bar(name="Imag", x=x, y=im, marker_color="red"), row=1, col=1
                )
                fig.add_trace(
                    go.Bar(name="Prob", x=x, y=probs, marker_color="green"),
                    row=2,
                    col=1,
                )
                fig.update_layout(barmode="group", height=500)
                fig.update_xaxes(title_text="Fock State |n>", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

            else:
                probs = np.real(np.diag(psi_out))
                fig_fock = px.bar(
                    x=np.arange(len(probs)),
                    y=probs,
                    labels={"x": "Fock State |n>", "y": "Probability"},
                    title="Output Fock Distribution (Mixed State)",
                )
                st.plotly_chart(fig_fock, use_container_width=True)

            # Circuit Schematic (Placeholder)
            st.divider()
            st.subheader("Circuit Schematics")
            sc1, sc2 = st.columns(2)
            with sc1:
                if config_a["type"] == "Gaussian Circuit":
                    st.caption("Source A (General Gaussian)")
                    fig_circ = utils.get_circuit_figure(config_a["params"])
                    st.pyplot(fig_circ)
            with sc2:
                if config_b["type"] == "Gaussian Circuit":
                    st.caption("Source B (General Gaussian)")
                    fig_circ = utils.get_circuit_figure(config_b["params"])
                    st.pyplot(fig_circ)

        except Exception as e:
            st.error(f"Simulation Error: {e}")
            st.exception(e)
