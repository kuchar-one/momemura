"""
Independent circuit verification using thewalrus + scipy.

Simulates the entire optical circuit from decoded parameters without
using the JAX backend, providing a cross-check of the optimization results.

Steps:
  1. Leaf heralding:  thewalrus.quantum.state_vector(mu, cov, post_select)
  2. Mixing tree:     scipy.linalg.expm for BS unitaries + scipy.special for Hermite
  3. Final Gaussian:  scipy.linalg.expm for squeeze/displacement
"""

import numpy as np
from scipy import linalg as sla
from scipy.special import hermite
from typing import Dict, Any, Tuple
import math

HBAR = 2.0


# ==============================================================
# 1. LEAF HERALDING (via thewalrus)
# ==============================================================


def _build_clements_unitary(phases: np.ndarray, N: int) -> np.ndarray:
    """
    Build N×N unitary from Clements decomposition.
    Replicates jax_clements_unitary exactly.
    """
    U = np.eye(N, dtype=complex)
    param_idx = 0

    for s in range(N):
        start = 0 if (s % 2 == 0) else 1
        for a in range(start, N - 1, 2):
            th = float(phases[param_idx])
            ph = float(phases[param_idx + 1])
            param_idx += 2

            # 2x2 BS matrix — must match jax_beamsplitter_2x2 exactly:
            # B = [[cos(θ), -e^{-iφ} sin(θ)],
            #      [e^{iφ} sin(θ),  cos(θ)    ]]
            t = np.cos(th)
            r = np.sin(th)
            B = np.array(
                [
                    [t, -np.exp(-1j * ph) * r],
                    [np.exp(1j * ph) * r, t],
                ],
                dtype=complex,
            )

            # Apply from left
            sub_U = U[a : a + 2, :]
            U[a : a + 2, :] = B @ sub_U

    # Final phases (varphis)
    varphis = phases[param_idx : param_idx + N]
    phases_diag = np.exp(1j * varphis)
    U = U * phases_diag[:, None]

    return U


def _build_gaussian_moments(r_vec, phases_vec, disp_vec, N_modes):
    """
    Build (mu, cov) for an N-mode Gaussian state using thewalrus.symplectic.

    Returns xp-ordered (x1,...,xN, p1,...,pN) mean and covariance.
    """
    import thewalrus.symplectic as symp

    # 1. Squeezing (per-mode)
    S_sq = np.eye(2 * N_modes)
    for i in range(N_modes):
        r = float(r_vec[i])
        # thewalrus.symplectic.squeezing returns 2x2 matrix
        S_single = symp.squeezing(r, phi=0.0)
        S_sq[i, i] = S_single[0, 0]
        S_sq[i, N_modes + i] = S_single[0, 1]
        S_sq[N_modes + i, i] = S_single[1, 0]
        S_sq[N_modes + i, N_modes + i] = S_single[1, 1]

    # 2. Passive unitary (Clements decomposition)
    U_pass = _build_clements_unitary(phases_vec[: N_modes * N_modes], N_modes)
    S_pass = symp.interferometer(U_pass)

    # 3. Total symplectic
    S_total = S_pass @ S_sq

    # 4. Displacement → mean vector
    scale = np.sqrt(2 * HBAR)
    alpha = np.array([complex(d) for d in disp_vec[:N_modes]])
    mu = np.zeros(2 * N_modes)
    mu[:N_modes] = scale * np.real(alpha)
    mu[N_modes:] = scale * np.imag(alpha)

    # 5. Covariance
    cov_vac = (HBAR / 2) * np.eye(2 * N_modes)
    cov = S_total @ cov_vac @ S_total.T

    return mu, cov


def _herald_leaf(r_vec, phases_vec, disp_vec, pnr_outcomes, n_ctrl, cutoff, pnr_max):
    """
    Compute heralded state for a single leaf using thewalrus.

    Returns:
        (state_vec, probability)
    """
    from thewalrus.quantum import state_vector

    N_modes = n_ctrl + 1  # signal + control modes

    # Build Gaussian moments using only N_modes modes
    r_eff = np.array(r_vec[:N_modes], dtype=float)
    phases_eff = np.array(phases_vec[: N_modes * N_modes], dtype=float)
    disp_eff = np.array(disp_vec[:N_modes])

    mu, cov = _build_gaussian_moments(r_eff, phases_eff, disp_eff, N_modes)

    if n_ctrl == 0:
        # Pure Gaussian, no heralding
        sv = state_vector(
            mu, cov, cutoff=cutoff, hbar=HBAR, normalize=False, check_purity=False
        )
        prob = float(np.sum(np.abs(sv) ** 2))
        if prob > 0:
            sv = sv / np.sqrt(prob)
        return sv, prob

    # Build post_select dict: {control_mode_index: pnr_outcome}
    # Signal is mode 0, control modes are 1..n_ctrl
    post_select = {}
    for i in range(n_ctrl):
        pnr_val = int(pnr_outcomes[i])
        post_select[i + 1] = pnr_val  # mode indices 1, 2, ...

    # Compute state vector using thewalrus reference implementation
    sv = state_vector(
        mu,
        cov,
        post_select=post_select,
        cutoff=cutoff,
        hbar=HBAR,
        normalize=False,
        check_purity=False,
    )

    # sv has shape (cutoff,) since 1 signal mode remains
    prob = float(np.sum(np.abs(sv) ** 2))
    if prob > 0:
        sv_norm = sv / np.sqrt(prob)
    else:
        sv_norm = np.zeros(cutoff, dtype=complex)

    return sv_norm, prob


# ==============================================================
# 2. MIXING TREE (via scipy)
# ==============================================================


def _fock_bs_unitary(theta: float, phi: float, cutoff: int) -> np.ndarray:
    """
    Build 2-mode beam splitter unitary in Fock basis using scipy.linalg.expm.

    BS = exp(theta * (exp(i*phi)*a†b - exp(-i*phi)*ab†))
    """
    # Annihilation operators
    a_single = np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), k=1)
    eye = np.eye(cutoff)

    # Two-mode operators
    a = np.kron(a_single, eye)  # a ⊗ I (mode 1)
    b = np.kron(eye, a_single)  # I ⊗ a (mode 2)

    a_dag = a.T
    b_dag = b.T

    # Generator
    gen = theta * (np.exp(1j * phi) * a_dag @ b - np.exp(-1j * phi) * a @ b_dag)

    U = sla.expm(gen)
    return U


def _hermite_phi(x: float, cutoff: int) -> np.ndarray:
    """
    Compute Fock-basis position wavefunction φ_n(x) for n = 0..cutoff-1.

    φ_n(x) = (1/sqrt(2^n n! sqrt(π ℏ))) * H_n(x/sqrt(ℏ)) * exp(-x²/(2ℏ))
    """
    phi_vec = np.zeros(cutoff)
    x_scaled = x / np.sqrt(HBAR)
    gauss = np.exp(-(x_scaled**2) / 2)

    for n in range(cutoff):
        Hn = hermite(n)
        norm = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi * HBAR))
        phi_vec[n] = norm * float(Hn(x_scaled)) * gauss

    return phi_vec


def _mix_pair(
    stateA: np.ndarray,
    stateB: np.ndarray,
    theta: float,
    phi: float,
    homodyne_x: float,
    cutoff: int,
) -> Tuple[np.ndarray, float]:
    """
    Mix two signal states on a beam splitter, then homodyne measure the second output.

    Returns:
        (output_state, homodyne_probability_density)
    """
    # Both inputs must be vectors (pure states)
    psi_in = np.kron(stateA, stateB)

    # Apply BS
    U = _fock_bs_unitary(theta, phi, cutoff)
    psi_out = U @ psi_in

    # Point homodyne on mode 2
    phi_vec = _hermite_phi(homodyne_x, cutoff)

    # Project: ⟨φ(x)|ψ⟩ over mode 2
    psi_2d = psi_out.reshape((cutoff, cutoff))
    v = psi_2d @ phi_vec  # shape (cutoff,)

    p_density = float(np.real(np.vdot(v, v)))

    if p_density > 0:
        v_norm = v / np.sqrt(p_density)
    else:
        v_norm = np.zeros(cutoff, dtype=complex)

    return v_norm, p_density


# ==============================================================
# 3. FINAL GAUSSIAN OPS (via scipy)
# ==============================================================


def _apply_final_gaussian(
    state: np.ndarray, final_gauss: Dict[str, Any], cutoff: int
) -> np.ndarray:
    """
    Apply final single-mode Gaussian operations: S(r,φ) · R(φ) · D(α)
    Uses scipy.linalg.expm.
    """
    r = float(final_gauss.get("r", 0.0))
    phi = float(final_gauss.get("phi", 0.0))
    varphi = float(final_gauss.get("varphi", 0.0))
    disp = complex(final_gauss.get("disp", 0.0))

    a = np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), k=1)
    adag = a.T

    # 1. Squeeze S(z) where z = r * exp(i * 2φ)
    term1 = np.exp(-2j * phi) * (a @ a)
    term2 = np.exp(2j * phi) * (adag @ adag)
    K_squeeze = (r / 2.0) * (term1 - term2)
    U_squeeze = sla.expm(K_squeeze)

    # 2. Rotation R(varphi)
    n_op = np.arange(cutoff)
    U_rot = np.diag(np.exp(1j * n_op * varphi))

    # 3. Displacement D(alpha)
    K_disp = disp * adag - np.conj(disp) * a
    U_disp = sla.expm(K_disp)

    U_final = U_disp @ U_rot @ U_squeeze
    return U_final @ state


# ==============================================================
# 4. MAIN VERIFICATION ENTRY POINT
# ==============================================================


def verify_circuit(
    params: Dict[str, Any], cutoff: int, pnr_max: int = 3
) -> Dict[str, Any]:
    """
    Independently simulate the full circuit from decoded parameters.

    Args:
        params: Decoded circuit parameters dict (same as used by describe_preparation_circuit)
        cutoff: Fock space cutoff dimension
        pnr_max: Maximum PNR outcome

    Returns:
        dict with:
            'state': final state vector (np.ndarray, shape (cutoff,))
            'probability': total herald probability
            'report': dict of diagnostic info
    """
    report = {"leaves": [], "mixing_nodes": [], "warnings": []}

    leaf_params = params["leaf_params"]
    leaf_active = params["leaf_active"]
    mix_params = params["mix_params"]

    # Helper to extract per-leaf arrays
    def get_leaf_val(key, idx):
        arr = leaf_params[key]
        if hasattr(arr, "__getitem__"):
            return arr[idx]
        return arr

    # -----------------------------------------------------------
    # Step 1: Compute heralded states for all 8 leaves
    # -----------------------------------------------------------
    leaf_states = []
    leaf_probs = []

    for i in range(8):
        is_active = bool(leaf_active[i])

        if not is_active:
            # Inactive leaf → vacuum state
            vac = np.zeros(cutoff, dtype=complex)
            vac[0] = 1.0
            leaf_states.append(vac)
            leaf_probs.append(1.0)
            report["leaves"].append(
                {"index": i, "active": False, "state_norm": 1.0, "prob": 1.0}
            )
            continue

        r_vec = np.array(get_leaf_val("r", i), dtype=float)
        phases_vec = np.array(get_leaf_val("phases", i), dtype=float)
        disp_vec = np.array(get_leaf_val("disp", i))
        pnr_raw = get_leaf_val("pnr", i)
        n_ctrl = int(get_leaf_val("n_ctrl", i))

        # Parse PNR outcomes
        if hasattr(pnr_raw, "tolist"):
            pnr_list = pnr_raw.tolist()
        elif isinstance(pnr_raw, list):
            pnr_list = pnr_raw
        else:
            pnr_list = [pnr_raw]
        pnr_outcomes = [int(p) for p in pnr_list[:n_ctrl]]

        try:
            sv, prob = _herald_leaf(
                r_vec, phases_vec, disp_vec, pnr_outcomes, n_ctrl, cutoff, pnr_max
            )
        except Exception as e:
            report["warnings"].append(f"Leaf {i} heralding failed: {e}")
            sv = np.zeros(cutoff, dtype=complex)
            sv[0] = 1.0
            prob = 0.0

        leaf_states.append(sv)
        leaf_probs.append(prob)
        report["leaves"].append(
            {
                "index": i,
                "active": True,
                "n_ctrl": n_ctrl,
                "pnr": pnr_outcomes,
                "state_norm": float(np.sum(np.abs(sv) ** 2)),
                "prob": prob,
            }
        )

    # -----------------------------------------------------------
    # Step 2: Mixing tree (7 nodes: 4 + 2 + 1)
    # -----------------------------------------------------------
    # Extract homodyne x values (vector, one per node)
    hx_raw = params.get("homodyne_x", 0.0)
    if hasattr(hx_raw, "__len__") and not isinstance(hx_raw, str):
        hx_values = [float(x) for x in hx_raw]
    else:
        hx_values = [float(hx_raw)] * 7

    # Pad to 7 if needed
    while len(hx_values) < 7:
        hx_values.append(0.0)

    # Layer 1: 4 pairs
    layer1_states = []
    layer1_probs = []
    layer1_active = []  # Track activity through the tree
    for j in range(4):
        mix_idx = j
        theta = float(mix_params[mix_idx][0])
        phi = float(mix_params[mix_idx][1])
        hx = hx_values[mix_idx]

        idxA = 2 * j
        idxB = 2 * j + 1

        sA = leaf_states[idxA]
        sB = leaf_states[idxB]

        activeA = bool(leaf_active[idxA])
        activeB = bool(leaf_active[idxB])

        if activeA and activeB:
            # Both active: mix on BS + homodyne
            out, p_hom = _mix_pair(sA, sB, theta, phi, hx, cutoff)
            combined_prob = leaf_probs[idxA] * leaf_probs[idxB]
            is_active = True
        elif activeA and not activeB:
            # Only A active: pass through (no BS)
            out = sA
            p_hom = 1.0
            combined_prob = leaf_probs[idxA]
            is_active = True
        elif not activeA and activeB:
            # Only B active: pass through (no BS)
            out = sB
            p_hom = 1.0
            combined_prob = leaf_probs[idxB]
            is_active = True
        else:
            # Both inactive: vacuum pass-through
            out = sA  # vacuum
            p_hom = 1.0
            combined_prob = 1.0
            is_active = False

        layer1_states.append(out)
        layer1_probs.append(combined_prob)
        layer1_active.append(is_active)

        report["mixing_nodes"].append(
            {
                "node": mix_idx,
                "layer": 1,
                "theta": theta,
                "phi": phi,
                "hx": hx,
                "p_homodyne": p_hom,
                "output_norm": float(np.sum(np.abs(out) ** 2)),
                "both_active": activeA and activeB,
            }
        )

    # Layer 2: 2 pairs
    layer2_states = []
    layer2_probs = []
    layer2_active = []
    for j in range(2):
        mix_idx = 4 + j
        theta = float(mix_params[mix_idx][0])
        phi = float(mix_params[mix_idx][1])
        hx = hx_values[mix_idx]

        sA = layer1_states[2 * j]
        sB = layer1_states[2 * j + 1]
        pA = layer1_probs[2 * j]
        pB = layer1_probs[2 * j + 1]
        actA = layer1_active[2 * j]
        actB = layer1_active[2 * j + 1]

        if actA and actB:
            out, p_hom = _mix_pair(sA, sB, theta, phi, hx, cutoff)
            combined_prob = pA * pB
            is_active = True
        elif actA and not actB:
            out = sA
            p_hom = 1.0
            combined_prob = pA
            is_active = True
        elif not actA and actB:
            out = sB
            p_hom = 1.0
            combined_prob = pB
            is_active = True
        else:
            out = sA
            p_hom = 1.0
            combined_prob = 1.0
            is_active = False

        layer2_states.append(out)
        layer2_probs.append(combined_prob)
        layer2_active.append(is_active)

        report["mixing_nodes"].append(
            {
                "node": mix_idx,
                "layer": 2,
                "theta": theta,
                "phi": phi,
                "hx": hx,
                "p_homodyne": p_hom,
                "output_norm": float(np.sum(np.abs(out) ** 2)),
                "both_active": actA and actB,
            }
        )

    # Layer 3: 1 pair (root)
    mix_idx = 6
    theta = float(mix_params[mix_idx][0])
    phi = float(mix_params[mix_idx][1])
    hx = hx_values[mix_idx]

    sA = layer2_states[0]
    sB = layer2_states[1]
    pA = layer2_probs[0]
    pB = layer2_probs[1]
    actA = layer2_active[0]
    actB = layer2_active[1]

    if actA and actB:
        final_state, p_hom = _mix_pair(sA, sB, theta, phi, hx, cutoff)
        total_prob = pA * pB
    elif actA and not actB:
        final_state = sA
        p_hom = 1.0
        total_prob = pA
    elif not actA and actB:
        final_state = sB
        p_hom = 1.0
        total_prob = pB
    else:
        final_state = sA
        p_hom = 1.0
        total_prob = 1.0

    report["mixing_nodes"].append(
        {
            "node": mix_idx,
            "layer": 3,
            "theta": theta,
            "phi": phi,
            "hx": hx,
            "p_homodyne": p_hom,
            "output_norm": float(np.sum(np.abs(final_state) ** 2)),
            "both_active": actA and actB,
        }
    )

    # -----------------------------------------------------------
    # Step 3: Final Gaussian operations
    # -----------------------------------------------------------
    if "final_gauss" in params and params["final_gauss"]:
        final_state = _apply_final_gaussian(final_state, params["final_gauss"], cutoff)
        report["final_gauss_applied"] = True
    else:
        report["final_gauss_applied"] = False

    report["final_state_norm"] = float(np.sum(np.abs(final_state) ** 2))

    return {
        "state": final_state,
        "probability": total_prob,
        "report": report,
    }
