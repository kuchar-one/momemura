import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Tuple

from src.circuits.jax_herald import (
    vacuum_covariance,
    passive_unitary_to_symplectic,
    two_mode_squeezer_symplectic,
    expand_mode_symplectic,
    complex_alpha_to_qp,
)
from src.circuits.jax_composer import jax_hermite_phi_matrix

# Constants from run_mome.py (should be shared ideally)
MAX_MODES = 3  # 1 signal + 2 control
MAX_SIGNAL = 1
MAX_CONTROL = 2
MAX_PNR = 3
MAX_SCHMIDT = 2


def jax_decode_genotype(g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
    """
    Decodes a single genotype into parameters for a Maximal Superblock (Depth 3).

    Genotype Layout (~200 params):
    - Homodyne: 2 params (x, window)
    - Mix Nodes (7): 4 params each (theta, phi, varphi, source) -> 28 params
    - Leaves (8): 18 params each (active, n_ctrl, tmss, us, uc, disp_s, disp_c, pnr) -> 144 params
    """
    # Pad g to safe length (e.g. 256)
    target_len = 256
    if g.shape[0] < target_len:
        g = jnp.pad(g, (0, target_len - g.shape[0]))

    idx = 0

    # 1. Homodyne
    hom_x_raw = g[idx]
    idx += 1
    hom_win_raw = g[idx]
    idx += 1

    homodyne_x = jnp.tanh(hom_x_raw) * 4.0
    homodyne_window = jnp.abs(jnp.tanh(hom_win_raw) * 2.0)

    # Rounding
    homodyne_x = jnp.round(homodyne_x * 1e6) / 1e6
    homodyne_window = jnp.round(homodyne_window * 1e6) / 1e6

    # 2. Mix Nodes (7)
    # 3 angles + 1 source
    n_mix = 7
    mix_params_flat = g[idx : idx + n_mix * 4]
    idx += n_mix * 4

    mix_params_reshaped = mix_params_flat.reshape((n_mix, 4))

    # Angles: tanh -> pi/2
    mix_angles = jnp.tanh(mix_params_reshaped[:, :3]) * (jnp.pi / 2)

    # Source: 0=Mix, 1=Left, 2=Right
    source_raw = mix_params_reshaped[:, 3]
    mix_source = jnp.zeros(n_mix, dtype=jnp.int32)
    mix_source = jnp.where(source_raw < -0.33, 1, mix_source)
    mix_source = jnp.where(source_raw > 0.33, 2, mix_source)

    # 3. Leaves (8)
    n_leaves = 8
    # Params per leaf:
    # 0: Active (bool)
    # 1: Num Controls (0, 1, 2)
    # 2-3: TMSS r (2)
    # 4: US phi (1) - 1 mode unitary is just phase
    # 5-8: UC params (4) - 2 mode unitary
    # 9-10: Disp S (2) - 1 mode
    # 11-14: Disp C (4) - 2 modes
    # 15-16: PNR (2)
    n_leaf_params = 17

    leaves_flat = g[idx : idx + n_leaves * n_leaf_params]
    # idx += n_leaves * n_leaf_params # Not strictly needed if we don't read more

    leaves_reshaped = leaves_flat.reshape((n_leaves, n_leaf_params))

    # Param 0: Active (bool)
    leaf_active = leaves_reshaped[:, 0] > 0.0

    # Param 1: Num Controls (0..2)
    # Map [-inf, inf] -> {0, 1, 2}
    # < -0.33 -> 0, -0.33..0.33 -> 1, > 0.33 -> 2
    n_ctrl_raw = leaves_reshaped[:, 1]
    leaf_n_ctrl = jnp.ones(n_leaves, dtype=jnp.int32)  # Default 1
    leaf_n_ctrl = jnp.where(n_ctrl_raw < -0.33, 0, leaf_n_ctrl)
    leaf_n_ctrl = jnp.where(n_ctrl_raw > 0.33, 2, leaf_n_ctrl)

    # Params 2-3: TMSS (2)
    tmss_r = jnp.tanh(leaves_reshaped[:, 2:4]) * 2.0

    # Param 4: US (1) - Phase only for 1 signal mode
    # We'll treat it as 'varphi' for 1 mode
    us_phase = jnp.tanh(leaves_reshaped[:, 4:5]) * (jnp.pi / 2)
    # We need to construct us_params dict, but jax_get_heralded_state expects arrays.
    # We'll pass the phase directly.

    # Params 5-8: UC (4) - 2 mode unitary
    uc_params = leaves_reshaped[:, 5:9]
    uc_theta = jnp.tanh(uc_params[:, 0:1]) * (jnp.pi / 2)
    uc_phi = jnp.tanh(uc_params[:, 1:2]) * (jnp.pi / 2)
    uc_varphi = jnp.tanh(uc_params[:, 2:4]) * (jnp.pi / 2)

    # Params 9-10: Disp S (2) - 1 mode (real, imag)
    disp_s_params = leaves_reshaped[:, 9:11]
    disp_s = (
        jnp.tanh(disp_s_params[:, 0]) * 3.0 + 1j * jnp.tanh(disp_s_params[:, 1]) * 3.0
    )
    # Reshape to (N, 1)
    disp_s = disp_s[:, None]

    # Params 11-14: Disp C (4) - 2 modes
    disp_c_params = leaves_reshaped[:, 11:15]
    disp_c_real = jnp.tanh(disp_c_params[:, 0::2]) * 3.0
    disp_c_imag = jnp.tanh(disp_c_params[:, 1::2]) * 3.0
    disp_c = disp_c_real + 1j * disp_c_imag

    # Params 15-16: PNR (2)
    pnr_raw = jnp.clip(leaves_reshaped[:, 15:17], 0.0, 1.0)
    pnr = jnp.round(pnr_raw * MAX_PNR).astype(jnp.int32)

    leaf_params = {
        "n_ctrl": leaf_n_ctrl,
        "tmss_r": tmss_r,
        "us_phase": us_phase,  # Changed from full params
        "uc_theta": uc_theta,
        "uc_phi": uc_phi,
        "uc_varphi": uc_varphi,
        "disp_s": disp_s,
        "disp_c": disp_c,
        "pnr": pnr,
    }

    return {
        "homodyne_x": homodyne_x,
        "homodyne_window": homodyne_window,
        "mix_params": mix_angles,
        "mix_source": mix_source,
        "leaf_active": leaf_active,
        "leaf_params": leaf_params,
    }


def jax_beamsplitter_2x2(theta: float, phi: float) -> jnp.ndarray:
    """
    2x2 complex beam-splitter matrix B(theta, phi).
    B = [[cosθ, -e^{-iφ} sinθ],
         [e^{iφ} sinθ, cosθ]].
    """
    t = jnp.cos(theta)
    r = jnp.sin(theta)
    exp_phi = jnp.exp(1j * phi)
    exp_neg_phi = jnp.exp(-1j * phi)

    row1 = jnp.array([t, -exp_neg_phi * r])
    row2 = jnp.array([exp_phi * r, t])
    return jnp.stack([row1, row2])


def jax_interferometer_unitary(
    theta: jnp.ndarray, phi: jnp.ndarray, varphi: jnp.ndarray, M: int
) -> jnp.ndarray:
    """
    Constructs MxM unitary using rectangular mesh.
    Unrolled for M=5 (max modes).
    """
    U = jnp.eye(M, dtype=jnp.complex_)

    # We can't use Python loops over dynamic M if M is traced.
    # But M is usually static (MAX_MODES or n_signal).
    # Since we extract MAX_MODES params, we should use M=MAX_MODES?
    # No, run_mome uses n_signal/n_control.
    # We can use `jax.lax.fori_loop` or unroll if M is small.
    # Let's assume M is static (passed as static arg to JIT).

    # Rectangular mesh logic:
    # for s in range(M):
    #   start = 0 if s%2==0 else 1
    #   for a in range(start, M-1, 2):
    #     apply BS(theta[k], phi[k])

    param_idx = 0

    # We need to carry U and param_idx?
    # No, we can just iterate.
    # But we need to slice theta/phi.

    # Since M is small (<=5), we can use Python loops for unrolling.
    for s in range(M):
        start = 0 if (s % 2 == 0) else 1
        for a in range(start, M - 1, 2):
            th = theta[param_idx]
            ph = phi[param_idx]
            param_idx += 1

            B = jax_beamsplitter_2x2(th, ph)

            # Update U rows [a, a+1]
            sub_U = U[a : a + 2, :]
            new_sub_U = B @ sub_U
            U = U.at[a : a + 2, :].set(new_sub_U)

    # Final phases
    # R = diag(exp(i varphi))
    phases = jnp.exp(1j * varphi)
    U = U * phases[:, None]  # Broadcast over columns (row-wise multiplication)
    return U


def jax_build_circuit(
    tmss_r: jnp.ndarray,
    us_theta: jnp.ndarray,
    us_phi: jnp.ndarray,
    us_varphi: jnp.ndarray,
    uc_theta: jnp.ndarray,
    uc_phi: jnp.ndarray,
    uc_varphi: jnp.ndarray,
    disp_s: jnp.ndarray,
    disp_c: jnp.ndarray,
    hbar: float = 2.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Builds the Gaussian state (mu, cov) for the herald circuit.
    """
    # Dimensions
    # Hardcoded for n_sig=1, n_ctrl=2 (MAX_CONTROL)
    # We use fixed size to ensure static shapes for JIT.
    n_sig = 1
    n_ctrl = 2
    N = n_sig + n_ctrl

    # 1. Vacuum
    cov = vacuum_covariance(N, hbar)
    mu = jnp.zeros(2 * N)

    # 2. TMSS
    # S_total starts as identity
    # We can apply symplectic matrices to cov/mu.
    # S_total = I.
    # For each r in tmss_r:
    #   S_sq = ...
    #   S_total = S_sq @ S_total

    # Since N is small, we can just multiply symplectic matrices.
    # Or better: construct S_total explicitly.

    # TMSS on modes (0, 1) -> (sig 0, ctrl 0)
    # If we have more pairs, they would be (sig 1, ctrl 1) etc.
    # But run_mome default is 1 signal, 2 control.
    # tmss_r has length 1 (schmidt_rank=1).

    # Let's assume 1 TMSS pair on (0, 1).
    r = tmss_r[0]
    S_sq_small = two_mode_squeezer_symplectic(r)
    S_sq = expand_mode_symplectic(S_sq_small, jnp.array([0, 1]), N)

    # Apply to state
    # cov = S @ cov @ S.T
    # mu = S @ mu
    # But mu is zero so far.

    # 3. Unitaries
    # U_s on signal modes (0)
    # U_c on control modes (1, 2)

    # U_s
    # If n_sig=1, U_s is just a phase shift.
    # jax_interferometer_unitary handles M=1 case?
    # It loops range(M). range(1) -> s=0. range(0, 0, 2) -> empty.
    # So just phases.
    U_s = jax_interferometer_unitary(us_theta, us_phi, us_varphi, n_sig)

    # U_c
    U_c = jax_interferometer_unitary(uc_theta, uc_phi, uc_varphi, n_ctrl)

    # Convert to symplectic
    S_us = passive_unitary_to_symplectic(U_s)
    S_uc = passive_unitary_to_symplectic(U_c)

    # Combine into block diagonal S_interf
    # S_interf = diag(S_us, S_uc)
    # S_us is 2x2 (for 1 mode). S_uc is 4x4 (for 2 modes).
    # Total 6x6.

    # We can use block_diag logic.
    # JAX doesn't have block_diag?
    # We can use jax.scipy.linalg.block_diag
    S_interf = jax.scipy.linalg.block_diag(S_us, S_uc)

    # Total Symplectic
    S_total = S_interf @ S_sq

    # Update cov, mu
    cov = S_total @ cov @ S_total.T
    mu = S_total @ mu

    # 4. Displacements
    # disp_s (n_sig,), disp_c (n_ctrl,)
    # Combine
    disp = jnp.concatenate([disp_s, disp_c])
    d_qp = complex_alpha_to_qp(disp, hbar)

    mu = mu + d_qp

    return mu, cov


def jax_get_heralded_state(params: Dict[str, jnp.ndarray], cutoff: int):
    """
    Computes the heralded state for a single block (1 Signal, up to 2 Controls).

    Args:
        params: Dict of params for ONE block.
        cutoff: Fock cutoff.

    Returns:
        (vec, prob, modes, max_pnr)
    """
    # Extract params
    n_ctrl = params["n_ctrl"]  # (1,) int
    tmss_r = params["tmss_r"]  # (2,)
    us_phase = params["us_phase"]  # (1,)
    uc_theta = params["uc_theta"]  # (1,)
    uc_phi = params["uc_phi"]  # (1,)
    uc_varphi = params["uc_varphi"]  # (2,)
    disp_s = params["disp_s"]  # (1,)
    disp_c = params["disp_c"]  # (2,)
    pnr = params["pnr"]  # (2,)

    # Constants
    hbar = 2.0

    # 1. Gaussian State (Covariance & Mean)
    # Modes: 0 (Signal), 1 (Ctrl0), 2 (Ctrl1)
    N = MAX_SIGNAL + MAX_CONTROL  # 1 + 2 = 3
    mu = jnp.zeros(2 * N)
    cov = vacuum_covariance(N, hbar)

    S_total = jnp.eye(2 * N)

    # TMSS Logic
    # We apply TMSS sequentially on (Sig, Ctrl_i)

    # TMSS 0: (Sig, Ctrl0) -> (0, 1)
    # Only if n_ctrl >= 1
    # We use lax.cond or just apply identity if r=0?
    # Better to use mask or cond.
    # But structure must be static?
    # We can just apply it with r * mask?
    # n_ctrl is dynamic (traced).

    # Mask r values
    # r0 effective = r0 if n_ctrl >= 1 else 0
    r0 = jnp.where(n_ctrl >= 1, tmss_r[0], 0.0)
    r1 = jnp.where(n_ctrl >= 2, tmss_r[1], 0.0)

    S_tmss_0 = two_mode_squeezer_symplectic(r0, 0.0)
    S_big_0 = expand_mode_symplectic(S_tmss_0, jnp.array([0, 1]), N)
    S_total = S_big_0 @ S_total

    S_tmss_1 = two_mode_squeezer_symplectic(r1, 0.0)
    S_big_1 = expand_mode_symplectic(S_tmss_1, jnp.array([0, 2]), N)
    S_total = S_big_1 @ S_total

    # Interferometers
    # US on Signal (Mode 0) - Phase rotation
    # U = exp(i phi) -> Symplectic is Rotation(phi)
    # 1-mode rotation R(phi) = [[c, -s], [s, c]]
    # us_phase is (1,)
    phi_s = us_phase[0]
    cp = jnp.cos(phi_s)
    sp = jnp.sin(phi_s)
    R_s = jnp.array([[cp, -sp], [sp, cp]])
    # Expand to N modes (Mode 0)
    S_us = expand_mode_symplectic(R_s, jnp.array([0]), N)
    S_total = S_us @ S_total

    # UC on Controls (Mode 1, 2)
    # 2-mode unitary on (1, 2)
    # Only relevant if n_ctrl >= 2?
    # If n_ctrl < 2, Mode 2 is vacuum and unentangled.
    # But we can always apply it.

    # Construct U_c (2x2)
    ct = jnp.cos(uc_theta[0])
    st = jnp.sin(uc_theta[0])
    # BS
    U_bs = jnp.array(
        [[ct, -jnp.exp(-1j * uc_phi[0]) * st], [jnp.exp(1j * uc_phi[0]) * st, ct]]
    )
    # Phase
    U_ph = jnp.diag(jnp.exp(1j * uc_varphi))
    U_c = U_ph @ U_bs

    S_uc_small = passive_unitary_to_symplectic(U_c)
    S_uc = expand_mode_symplectic(S_uc_small, jnp.array([1, 2]), N)
    S_total = S_uc @ S_total

    # Apply Total Symplectic
    cov = S_total @ cov @ S_total.T
    mu = S_total @ mu  # mu starts at 0

    # Displacements
    # disp_s (1,), disp_c (2,)
    # alpha = [disp_s[0], disp_c[0], disp_c[1]]
    alpha = jnp.concatenate([disp_s, disp_c])
    r_disp = complex_alpha_to_qp(alpha, hbar)
    mu = mu + r_disp

    # Herald
    # We project Control modes (1, 2) onto Fock states |pnr>
    # pnr is (2,)
    # If n_ctrl < 2, we should project Mode 2 onto |0>?
    # Or just ignore it?
    # If we ignore it, we must trace it out -> Mixed state.
    # User wants PURE states.
    # So we MUST project all auxiliary modes.
    # If n_ctrl < 2, Mode 2 is vacuum (uncoupled). Projecting on |0> is correct/consistent.
    # So we enforce pnr[1] = 0 if n_ctrl < 2.

    pnr_effective = jnp.where(
        jnp.arange(2) < n_ctrl,
        pnr,
        0,  # Force 0 for inactive controls
    )

    # jax_pure_state_amplitude expects pnr tuple for ALL auxiliary modes?
    # No, it expects pnr for the modes being heralded.
    # Our `jax_pure_state_amplitude` implementation takes `pnr` which matches `n_aux`.
    # Here `n_aux` is 2 (Modes 1, 2).
    # So we pass pnr_effective (2,).

    # Note: jax_pure_state_amplitude needs static shape for pnr?
    # It takes `pnr: Tuple[int, ...]`.
    # But here pnr is dynamic array.
    # We need `jax_get_full_amplitudes` which computes all PNRs up to MAX_PNR.

    max_pnr_tuple = (MAX_PNR, MAX_PNR)

    # H_full: (cutoff, MAX_PNR+1, MAX_PNR+1)
    # Wait, jax_get_full_amplitudes returns (cutoff^n_sig, pnr_dims...)
    # n_sig = 1.

    from src.circuits.jax_herald import jax_get_full_amplitudes

    H_full = jax_get_full_amplitudes(mu, cov, max_pnr_tuple, cutoff, hbar)

    # Slice
    p0 = pnr_effective[0]
    p1 = pnr_effective[1]

    vec_slice = H_full[:, p0, p1]  # (cutoff,)

    # Prob
    prob_slice = jnp.sum(jnp.abs(vec_slice) ** 2)

    # Normalize
    vec_norm = jax.lax.cond(
        prob_slice > 0,
        lambda _: vec_slice / jnp.sqrt(prob_slice),
        lambda _: jnp.zeros_like(vec_slice),
        None,
    )

    # Return pure vector (cutoff,)
    return (
        vec_norm,
        prob_slice,
        1.0,
        jnp.max(pnr_effective).astype(jnp.float_),
        jnp.sum(pnr_effective).astype(jnp.float_),
    )


# -------------------------
# Batched Scoring
# -------------------------


@partial(jax.jit, static_argnames=("cutoff",))
def _score_batch_shard(
    genotypes: jnp.ndarray, cutoff: int, operator: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled scoring function for a single batch shard.
    This runs on one device.
    """

    def score_one(g):
        # Decode
        params = jax_decode_genotype(g, cutoff)

        # 1. Get Leaf States
        # vmap over 8 leaves
        leaf_vecs, leaf_probs, leaf_modes, leaf_max_pnrs, leaf_total_pnrs = jax.vmap(
            partial(jax_get_heralded_state, cutoff=cutoff)
        )(params["leaf_params"])

        # 2. Superblock
        hom_x = params["homodyne_x"]
        hom_win = params["homodyne_window"]

        # Homodyne matrices
        # Point homodyne for now
        phi_mat = jax_hermite_phi_matrix(jnp.array([hom_x]), cutoff)
        phi_vec = phi_mat[:, 0]

        # Dummy window params
        V_matrix = jnp.zeros((cutoff, 1))
        dx_weights = jnp.zeros(1)

        # Call superblock
        from src.circuits.jax_composer import jax_superblock

        final_state, _, joint_prob, is_active, max_pnr, active_modes = jax_superblock(
            leaf_vecs,
            leaf_probs,
            params["leaf_active"],
            leaf_max_pnrs,  # Use max PNRs for superblock propagation
            leaf_modes,
            params["mix_params"],
            params["mix_source"],
            hom_x,
            hom_win,
            0.0,  # resolution
            phi_vec,
            V_matrix,
            dx_weights,
            cutoff,
            True,  # window is none (we use point homodyne via phi_vec)
            False,  # x is none? (False because we use hom_x)
            True,  # resolution is none
        )

        # 3. Expectation Value
        # final_state is (cutoff,) pure state vector
        # exp_val = <psi|O|psi>
        # We use vdot: vdot(a, b) = a.conj() @ b
        # So vdot(psi, O @ psi)

        # Ensure final_state is vector
        # If it's density matrix (from mixed path), we use trace logic.
        # But we are in pure path now.

        op_psi = operator @ final_state
        exp_val = jnp.real(jnp.vdot(final_state, op_psi))

        # 4. Fitness & Descriptors
        prob_clipped = jnp.maximum(joint_prob, 1e-30)
        log_prob = -jnp.log10(prob_clipped)

        # Calculate Total Photons (sum of all PNRs in active leaves)
        # leaf_active is boolean (8,)
        # leaf_total_pnrs is (8,)
        total_photons = jnp.sum(jnp.where(params["leaf_active"], leaf_total_pnrs, 0.0))

        # Fitness: [-exp_val, -log_prob, -complexity, -total_photons]
        # Complexity = active_modes
        fitness = jnp.array([-exp_val, -log_prob, -active_modes, -total_photons])

        # Descriptor: [active_modes, max_pnr, total_photons]
        # D1: Complexity (Total Modes)
        # D2: Max PNR
        # D3: Total Photons
        descriptor = jnp.array([active_modes, max_pnr, total_photons])

        return fitness, descriptor

    fitnesses, descriptors = jax.vmap(score_one)(genotypes)
    return fitnesses, descriptors


def jax_scoring_fn_batch(
    genotypes: jnp.ndarray, cutoff: int, operator: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batched scoring function.
    Dispatches to multi-device pmap if devices > 1, else uses jit.
    """
    n_devices = jax.local_device_count()

    if n_devices <= 1:
        # Fallback to single-device JIT
        return _score_batch_shard(genotypes, cutoff, operator)

    # Multi-GPU Logic
    # 1. Pad genotypes to be divisible by n_devices
    batch_size = genotypes.shape[0]
    remainder = batch_size % n_devices
    padding = (n_devices - remainder) if remainder > 0 else 0

    if padding > 0:
        # Pad with zeros (or duplicates)
        # Since these are extra, their results will be discarded.
        # Zeros might cause numerical issues in eval?
        # Better to pad with the first element to ensure validity.
        pad_block = jnp.repeat(genotypes[:1], padding, axis=0)
        g_padded = jnp.concatenate([genotypes, pad_block], axis=0)
    else:
        g_padded = genotypes

    # 2. Reshape for pmap: (n_devices, shard_size, ...)
    shard_size = g_padded.shape[0] // n_devices
    g_sharded = g_padded.reshape((n_devices, shard_size, -1))

    # 3. pmap execution
    # Operator needs to be replicated? pmap can handle broadcast if not mapped.
    # We map axis 0 of g_sharded. operator is constant (broadcasted).
    # cutoff is static.
    # We need a pmapped function.
    # Note: pmap JIT compiles automatically.

    # We define pmapped function outside or cache it?
    # pmap is expensive to re-compile if partials change.
    # _score_batch_shard is JIT compressed.
    # We can use pmap(_score_batch_shard, in_axes=(0, None, None))

    # NOTE: pmap args: (shard, cutoff, operator)
    # in_axes=(0, None, None) -> shard divides axis 0, others replicated.

    pmapped_fn = jax.pmap(
        _score_batch_shard,
        in_axes=(0, None, None),
        static_broadcasted_argnums=(1,),  # cutoff is arg 1
    )

    # Replicate operator?
    # Actually, for pmap with None in_axes, the arg is broadcasted to all devices.
    # But for array args, it's efficient to do it implicitly.

    fitnesses_sharded, descriptors_sharded = pmapped_fn(g_sharded, cutoff, operator)

    # 4. Reshape back
    # fitnesses_sharded: (n_devices, shard_size, 4)
    # descriptors_sharded: (n_devices, shard_size, 3)

    fitnesses = fitnesses_sharded.reshape((-1, 4))
    descriptors = descriptors_sharded.reshape((-1, 3))

    # 5. Remove padding
    if padding > 0:
        fitnesses = fitnesses[:batch_size]
        descriptors = descriptors[:batch_size]

    return fitnesses, descriptors
