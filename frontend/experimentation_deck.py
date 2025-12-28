import streamlit as st
import numpy as np
import plotly.express as px
import jax.numpy as jnp
from typing import Dict

# Add project root to path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Backend Components
try:
    from src.simulation.jax.runner import jax_get_heralded_state
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

    Returns:
        Result Dict containing:
        - 'state_a', 'prob_a'
        - 'state_b', 'prob_b'
        - 'state_out', 'prob_out', 'prob_joint'
    """

    # 1. Generate Source A
    if config_a["type"] == "Fock":
        vec_a = parse_fock_input(config_a["fock_coeffs"], cutoff_dim)
        prob_a = 1.0
    else:
        p_a = config_a["params"]
        n_c = p_a["n_control"]

        leaf_params_a = {
            "n_ctrl": jnp.array(n_c),  # Scalar (0-D)
            "tmss_r": jnp.array(p_a["r"]),  # Scalar (0-D)
            "us_phase": jnp.array([p_a["phase"]]),
            "uc_theta": jnp.array(p_a["uc_theta"]),
            "uc_phi": jnp.array(p_a["uc_phi"]),
            "uc_varphi": jnp.array(p_a["uc_varphi"]),
            "disp_s": jnp.array([0.0]),
            "disp_c": jnp.zeros(n_c),
            "pnr": jnp.array(p_a["pnr"]),  # Shape (N_C,)
        }

        vec_a_jax, prob_a_jax, _, _, _, _ = jax_get_heralded_state(
            leaf_params_a, cutoff_dim, pnr_max=int(p_a.get("max_pnr", 3))
        )
        vec_a = np.array(vec_a_jax)
        prob_a = float(prob_a_jax)

    # 2. Generate Source B (Mirror A)
    if config_b["type"] == "Fock":
        vec_b = parse_fock_input(config_b["fock_coeffs"], cutoff_dim)
        prob_b = 1.0
    else:
        p_b = config_b["params"]
        n_c = p_b["n_control"]
        leaf_params_b = {
            "n_ctrl": jnp.array(n_c),  # Scalar (0-D)
            "tmss_r": jnp.array(p_b["r"]),  # Scalar (0-D)
            "us_phase": jnp.array([p_b["phase"]]),
            "uc_theta": jnp.array(p_b["uc_theta"]),
            "uc_phi": jnp.array(p_b["uc_phi"]),
            "uc_varphi": jnp.array(p_b["uc_varphi"]),
            "disp_s": jnp.array([0.0]),
            "disp_c": jnp.zeros(n_c),
            "pnr": jnp.array(p_b["pnr"]),
        }
        vec_b_jax, prob_b_jax, _, _, _, _ = jax_get_heralded_state(
            leaf_params_b, cutoff_dim, pnr_max=int(p_b.get("max_pnr", 3))
        )
        vec_b = np.array(vec_b_jax)
        prob_b = float(prob_b_jax)

    # 3. Mixing / Pass
    mode = mix_config["mode"]

    # Defaults
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

        # Prepare JAX inputs
        state_A_jax = jnp.array(vec_a)
        state_B_jax = jnp.array(vec_b)

        # U_BS
        U = jax_u_bs(theta, phi, cutoff_dim)

        # Point Homodyne Vectors
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
            jnp.array(hom_val),  # hom_x
            0.0,  # win
            res_val,  # res
            phi_vec,
            None,
            None,  # V, dx
            cutoff_dim,
            homodyne_window_is_none=True,
            homodyne_x_is_none=False,
            homodyne_resolution_is_none=True,  # Just get density
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


def get_drawing_params(config_params):
    """Convert config params to circuit drawing params."""
    # config: r, phase, n_control, uc_theta/phi/varphi, pnr
    n_c = config_params["n_control"]

    # 1. TMSS
    # If n_c >= 1, r applies to (0, 1).
    tmss_sq = [config_params["r"]] if n_c >= 1 else []

    # 2. US
    # config phase is scalar.
    us_p = {"theta": [], "phi": [], "varphi": [config_params["phase"]]}

    # 3. UC
    # Split
    def ensure_list(x):
        return x if isinstance(x, list) else [x]

    uc_p = {
        "theta": ensure_list(config_params["uc_theta"]),
        "phi": ensure_list(config_params["uc_phi"]),
        "varphi": ensure_list(config_params["uc_varphi"]),
    }

    # 4. Displacements
    # Zero
    disp_s = []
    disp_c = [0.0] * n_c

    return {
        "n_signal": 1,
        "n_control": n_c,
        "tmss_squeezing": tmss_sq,
        "us_params": us_p,
        "uc_params": uc_p,
        "disp_s": disp_s,
        "disp_c": disp_c,
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
        # Gaussian Params
        r_a = st.slider("r (Squeezing)", -2.0, 2.0, 1.0, key="r_a")
        ph_a = st.slider("Signal Phase", 0.0, 2 * np.pi, 0.0, key="ph_a")
        nc_a = st.number_input("N Control", 1, 3, 1, key="nc_a")

        # UC Params
        # Expanders for complex setups
        with st.expander("Control Unitary Params", expanded=False):
            # We need n_pair parameters.
            n_pairs = (nc_a * (nc_a - 1)) // 2

            uc_th = []
            uc_phi = []
            if n_pairs > 0:
                st.markdown(f"**Pairs**: {n_pairs}")
                # Simple text input for lists or just one uniform value?
                # Let's do uniform for ease, with toggle?
                # Or parsing "0.1, 0.2"
                th_str = st.text_input("Thetas (comma sep)", "0.0", key="ucth_a")
                phi_str = st.text_input("Phis (comma sep)", "0.0", key="ucphi_a")

                # Parse
                try:
                    uc_th = [float(x) for x in th_str.split(",")]
                except ValueError:
                    uc_th = [0.0]
                try:
                    uc_phi = [float(x) for x in phi_str.split(",")]
                except ValueError:
                    uc_phi = [0.0]

                # Pad
                if len(uc_th) < n_pairs:
                    uc_th = uc_th + [uc_th[-1]] * (n_pairs - len(uc_th))
                if len(uc_phi) < n_pairs:
                    uc_phi = uc_phi + [uc_phi[-1]] * (n_pairs - len(uc_phi))

            # Varphis (N_Control)
            var_str = st.text_input("Varphis (Control Phases)", "0.0", key="ucvar_a")
            try:
                uc_var = [float(x) for x in var_str.split(",")]
            except ValueError:
                uc_var = [0.0]
            if len(uc_var) < nc_a:
                uc_var = uc_var + [uc_var[-1]] * (nc_a - len(uc_var))

        # PNR
        pnr_str = st.text_input("PNR Outcome (e.g. 1,0)", "1, 0", key="pnr_a")
        try:
            pnr_a = [int(x) for x in pnr_str.split(",")]
        except ValueError:
            pnr_a = [0]
        if len(pnr_a) < nc_a:
            pnr_a = pnr_a + [0] * (nc_a - len(pnr_a))

        config_a["params"] = {
            "r": r_a,
            "phase": ph_a,
            "n_control": nc_a,
            "uc_theta": uc_th,
            "uc_phi": uc_phi,
            "uc_varphi": uc_var,
            "pnr": pnr_a,
        }

# --- Source B ---
with col_in_b:
    st.subheader("Source B")
    type_b = st.radio("Type B", ["Gaussian Circuit", "Fock State"], key="type_b")

    config_b = {"type": type_b}

    if type_b == "Fock State":
        f_in = st.text_input("Fock Coeffs (0, 1...)", "1", key="fock_b")
        config_b["fock_coeffs"] = f_in
    else:
        # Gaussian Params
        r_b = st.slider("r (Squeezing)", -2.0, 2.0, 0.0, key="r_b")
        ph_b = st.slider("Signal Phase", 0.0, 2 * np.pi, 0.0, key="ph_b")
        nc_b = st.number_input("N Control", 1, 3, 1, key="nc_b")

        with st.expander("Control Unitary Params", expanded=False):
            n_pairs = (nc_b * (nc_b - 1)) // 2
            uc_th = []
            uc_phi = []
            if n_pairs > 0:
                st.markdown(f"**Pairs**: {n_pairs}")
                th_str = st.text_input("Thetas", "0.0", key="ucth_b")
                phi_str = st.text_input("Phis", "0.0", key="ucphi_b")
                try:
                    uc_th = [float(x) for x in th_str.split(",")]
                except ValueError:
                    uc_th = [0.0]
                try:
                    uc_phi = [float(x) for x in phi_str.split(",")]
                except ValueError:
                    uc_phi = [0.0]
                if len(uc_th) < n_pairs:
                    uc_th = uc_th + [uc_th[-1]] * (n_pairs - len(uc_th))
                if len(uc_phi) < n_pairs:
                    uc_phi = uc_phi + [uc_phi[-1]] * (n_pairs - len(uc_phi))

            var_str = st.text_input("Varphis", "0.0", key="ucvar_b")
            try:
                uc_var = [float(x) for x in var_str.split(",")]
            except ValueError:
                uc_var = [0.0]
            if len(uc_var) < nc_b:
                uc_var = uc_var + [uc_var[-1]] * (nc_b - len(uc_var))

        pnr_str = st.text_input("PNR Outcome", "0", key="pnr_b")
        try:
            pnr_b = [int(x) for x in pnr_str.split(",")]
        except ValueError:
            pnr_b = [0]
        if len(pnr_b) < nc_b:
            pnr_b = pnr_b + [0] * (nc_b - len(pnr_b))

        config_b["params"] = {
            "r": r_b,
            "phase": ph_b,
            "n_control": nc_b,
            "uc_theta": uc_th,
            "uc_phi": uc_phi,
            "uc_varphi": uc_var,
            "pnr": pnr_b,
        }

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
        # Homodyne
        hom_angle = st.slider(
            "Homodyne Angle (Unused?)",
            0.0,
            2 * np.pi,
            0.0,
            help="Rotation before X meas? Assuming standard X meas in basis.",
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

            # Helper for plot
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
            # Ensure 1D
            if psi_out.ndim > 1:
                # If DM, diag?
                probs = np.real(np.diag(psi_out))
            else:
                probs = np.abs(psi_out) ** 2

            fig_fock = px.bar(
                x=np.arange(len(probs)),
                y=probs,
                labels={"x": "Fock State |n>", "y": "Probability"},
            )
            st.plotly_chart(fig_fock, use_container_width=True)

            # Circuit Schematic
            st.divider()
            st.subheader("Circuit Schematics (Input)")

            sc1, sc2 = st.columns(2)
            with sc1:
                if config_a["type"] == "Gaussian Circuit":
                    st.caption("Source A Circuit")
                    d_params = get_drawing_params(config_a["params"])
                    fig_circ = utils.get_circuit_figure(d_params)
                    st.pyplot(fig_circ)
                else:
                    st.info("Source A is Fock State (No Circuit)")

            with sc2:
                if config_b["type"] == "Gaussian Circuit":
                    st.caption("Source B Circuit")
                    d_params = get_drawing_params(config_b["params"])
                    fig_circ = utils.get_circuit_figure(d_params)
                    st.pyplot(fig_circ)
                else:
                    st.info("Source B is Fock State (No Circuit)")

        except Exception as e:
            st.error(f"Simulation Error: {e}")
            st.exception(e)
