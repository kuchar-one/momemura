import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Tuple

from src.genotypes.genotypes import get_genotype_decoder

from src.simulation.jax.herald import (
    vacuum_covariance,
    two_mode_squeezer_symplectic,
    expand_mode_symplectic,
    passive_unitary_to_symplectic,
    complex_alpha_to_qp,
)
from src.simulation.jax.composer import jax_hermite_phi_matrix

# Constants from run_mome.py (should be shared ideally)
MAX_MODES = 3  # 1 signal + 2 control
MAX_SIGNAL = 1
MAX_CONTROL = 2
MAX_PNR = 3


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


def jax_creation_annihilation(cutoff: int):
    """Returns a and a_dag matrices for Fock size cutoff."""
    a = jnp.diag(jnp.sqrt(jnp.arange(1, cutoff)), k=1)
    return a, a.T


def jax_apply_final_gaussian(
    state_vec: jnp.ndarray, params: Dict[str, Any], cutoff: int
) -> jnp.ndarray:
    """
    Applies single-mode Gaussian unitary U_final to the state vector.
    U_final = D(alpha) @ R(varphi) @ S(r, phi)
    """
    r = params["r"]
    phi = params["phi"]
    varphi = params["varphi"]
    disp = params["disp"]  # complex alpha

    a, adag = jax_creation_annihilation(cutoff)

    # 1. Squeeze S(z) with z = r * exp(i * 2*phi)
    # S(z) = exp(0.5 * (z.conj() * a^2 - z * adag^2))
    # Using user formula: K = (r/2) * (exp(-2j*phi)*a^2 - exp(2j*phi)*adag^2)
    # This matches standard S(z) definition if z = r * exp(2j*phi).

    # We use jax.scipy.linalg.expm
    term1 = jnp.exp(-2j * phi) * (a @ a)
    term2 = jnp.exp(2j * phi) * (adag @ adag)
    K_squeeze = (r / 2.0) * (term1 - term2)
    U_squeeze = jax.scipy.linalg.expm(K_squeeze)

    # 2. Rotation R(varphi) = exp(i * n * varphi)
    n_op = jnp.arange(cutoff)
    U_rot = jnp.diag(jnp.exp(1j * n_op * varphi))

    # 3. Displacement D(alpha) = exp(alpha * adag - alpha.conj() * a)
    K_disp = disp * adag - jnp.conj(disp) * a
    U_disp = jax.scipy.linalg.expm(K_disp)

    # Total U
    # Order: U_final = U_disp @ R(varphi) @ U_squeeze (Applied right to left on ket)
    # This matches common decompositions (Squeeze then Rotate then Displace)
    U_final = U_disp @ U_rot @ U_squeeze

    return U_final @ state_vec


def jax_get_heralded_state(
    params: Dict[str, jnp.ndarray], cutoff: int, pnr_max: int = 3
):
    """
    Computes the heralded state for a single block (1 Signal, up to 2 Controls).

    Args:
        params: Dict containing leaf parameters
        cutoff: Fock cutoff dimension
        pnr_max: Maximum PNR value for amplitude tensor (static, must be known at trace time)
    """
    # Extract params
    n_ctrl_eff = params["n_ctrl"]  # (1,) int
    # tmss_r is now SCALAR (0-D array) after vmap slicing from (L,)
    tmss_r = params["tmss_r"]
    us_phase = params["us_phase"]
    uc_theta = params["uc_theta"]
    uc_phi = params["uc_phi"]
    uc_varphi = params["uc_varphi"]
    disp_s = params["disp_s"]
    disp_c = params["disp_c"]
    pnr = params["pnr"]

    # Determine dimensions dynamically
    # We infer N_C from pnr or disp_c shape.
    N_C = pnr.shape[-1]
    N = 1 + N_C  # 1 Signal + N_C Controls

    hbar = 2.0
    mu = jnp.zeros(2 * N)
    cov = vacuum_covariance(N, hbar)

    S_total = jnp.eye(2 * N)

    # 2. TMSS (Signal 0 + Control 0)
    # Only if N_C >= 1
    # n_ctrl_eff tells us how many controls are active
    r0 = jnp.where(n_ctrl_eff >= 1, tmss_r, 0.0)

    # We always have at least 1 signal. If N_C > 0:
    if N_C > 0:
        S_tmss_0 = two_mode_squeezer_symplectic(r0, 0.0)
        S_big_0 = expand_mode_symplectic(S_tmss_0, jnp.array([0, 1]), N)
        S_total = S_big_0 @ S_total

    # 3. Interferometers
    # US on Signal (Mode 0)
    phi_s = us_phase[0]
    cp = jnp.cos(phi_s)
    sp = jnp.sin(phi_s)
    R_s = jnp.array([[cp, -sp], [sp, cp]])
    S_us = expand_mode_symplectic(R_s, jnp.array([0]), N)
    S_total = S_us @ S_total

    # UC on Controls (Modes 1..N_C)
    # Construct unitary U_c for controls
    if N_C > 0:
        # Identity
        U_c = jnp.eye(N_C, dtype=jnp.complex64)

        pair_idx = 0
        # Iterate pairs (i, j) for i<j?
        for i in range(N_C):
            for j in range(i + 1, N_C):
                # Extract params
                th = uc_theta[pair_idx]
                ph = uc_phi[pair_idx]

                ct = jnp.cos(th)
                st = jnp.sin(th)

                # BS matrix on i, j
                row_i = U_c[i, :].copy()
                row_j = U_c[j, :].copy()

                U_c = U_c.at[i, :].set(ct * row_i - jnp.exp(-1j * ph) * st * row_j)
                U_c = U_c.at[j, :].set(jnp.exp(1j * ph) * st * row_i + ct * row_j)

                pair_idx += 1

        # Phases at output
        # uc_varphi has size N_C
        U_ph = jnp.diag(jnp.exp(1j * uc_varphi))
        U_c = U_ph @ U_c

        S_uc_small = passive_unitary_to_symplectic(U_c)
        ctrl_indices = jnp.arange(1, N)
        S_uc = expand_mode_symplectic(S_uc_small, ctrl_indices, N)
        S_total = S_uc @ S_total

    # Apply Total Symplectic
    cov = S_total @ cov @ S_total.T
    mu = S_total @ mu

    # 4. Displacements
    # disp_s is (1,), disp_c is (N_C,)
    alpha = jnp.concatenate([disp_s, disp_c])
    r_disp = complex_alpha_to_qp(alpha, hbar)
    mu = mu + r_disp

    # 5. Herald
    # Project auxiliary modes (1..N_C) onto PNR

    ctrl_indices_local = jnp.arange(N_C)  # 0..N_C-1
    # n_ctrl_eff is integer 0..N_C

    # Mask PNR: Use provided PNR for active controls, 0 for inactive
    pnr_effective = jnp.where(ctrl_indices_local < n_ctrl_eff, pnr, 0)

    # Use pnr_max parameter (passed as static argument)
    MAX_PNR_LOCAL = pnr_max

    from src.simulation.jax.herald import jax_get_full_amplitudes

    max_pnr_sequence = [MAX_PNR_LOCAL] * N_C
    max_pnr_tuple = tuple(max_pnr_sequence)

    H_full = jax_get_full_amplitudes(mu, cov, max_pnr_tuple, cutoff, hbar)

    H_flat_ctrl = H_full.reshape(cutoff, -1)

    # Calculate flat index from pnr_effective
    D_ctrl = MAX_PNR_LOCAL + 1

    flat_idx = 0
    # Iterate backwards?
    # H[c, i, j] -> linear [c, i*D + j]
    for i in range(N_C):
        # pnr index for control i
        p = pnr_effective[i]
        # power of D: (N_C - 1 - i)
        power = N_C - 1 - i
        stride = D_ctrl**power
        flat_idx += p * stride

    vec_slice = H_flat_ctrl[:, flat_idx]

    # Calculate probability of heralding
    prob_slice = jnp.sum(jnp.abs(vec_slice) ** 2)

    # Normalize
    vec_norm = jax.lax.cond(
        prob_slice > 0,
        lambda _: vec_slice / jnp.sqrt(prob_slice),
        lambda _: jnp.zeros_like(vec_slice),
        None,
    )

    return (
        vec_norm,
        prob_slice,
        1.0,
        jnp.max(pnr_effective).astype(jnp.float_),
        jnp.sum(pnr_effective).astype(jnp.float_),  # Total PNR sum
        1.0,  # Count as 1 leaf (not modes per leaf) - summed to get active leaf count
    )


@partial(
    jax.jit,
    static_argnames=(
        "cutoff",
        "genotype_name",
        "genotype_config",
        "correction_cutoff",
        "pnr_max",
    ),
)
def _score_batch_shard(
    genotypes: jnp.ndarray,
    cutoff: int,
    operator: jnp.ndarray,
    genotype_name: str,
    genotype_config: Any = None,  # Can be dict or tuple(items)
    correction_cutoff: int = None,
    pnr_max: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled scoring function for a single batch shard.
    Supports dynamic limits via correction_cutoff simulation.
    """
    # Reconstruct config dict if passed as tuple
    if genotype_config is not None and isinstance(genotype_config, tuple):
        config_dict = dict(genotype_config)
    else:
        config_dict = genotype_config

    # Extract depth from config for tree structure (needed by decoder)
    depth = 3
    if config_dict is not None:
        depth = int(config_dict.get("depth", 3))

    decoder = get_genotype_decoder(genotype_name, depth=depth, config=config_dict)

    # Determine Simulation Cutoff
    use_correction = (correction_cutoff is not None) and (correction_cutoff > cutoff)
    sim_cutoff = correction_cutoff if use_correction else cutoff

    # pnr_max is now passed as argument (static)
    # This overrides config, or rather assumes config was used to set the arg.
    # Re-extracting from config is redundant but harmless unless they conflict.
    # We use the explicit argument.
    pass

    def score_one(g):
        # Decode using simulation cutoff (scales applied here if any)
        # Note: Decoder limits (r_scale) are in config/decoder, not passed here.
        params = decoder.decode(g, sim_cutoff)

        # 1. Get Leaf States (in sim_cutoff)
        leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes = jax.vmap(
            partial(jax_get_heralded_state, cutoff=sim_cutoff, pnr_max=pnr_max)
        )(params["leaf_params"])

        # 2. Superblock (in sim_cutoff)
        hom_x = params["homodyne_x"]
        hom_win = params["homodyne_window"]

        # Point homodyne
        # Handle scalar or vector hom_x
        hom_xs = jnp.atleast_1d(hom_x)
        # jax_hermite_phi_matrix returns (cutoff, N) for input (N,)
        phi_mat = jax_hermite_phi_matrix(hom_xs, sim_cutoff)

        # If scalar original, we want (cutoff,)
        if jnp.ndim(hom_x) == 0:
            phi_vec = phi_mat[:, 0]
        else:
            # If vector, we want (N, cutoff) for compatibility with jax_superblock broadcasting
            phi_vec = phi_mat.T

        V_matrix = jnp.zeros((sim_cutoff, 1))
        dx_weights = jnp.zeros(1)

        from src.simulation.jax.composer import jax_superblock

        (
            final_state,
            _,
            joint_prob,
            is_active,
            max_pnr,
            total_sum_pnr,
            active_modes,
        ) = jax_superblock(
            leaf_vecs,
            leaf_probs,
            params["leaf_active"],
            leaf_max_pnrs,
            leaf_total_pnrs,
            leaf_modes,
            params["mix_params"],
            hom_x,
            hom_win,
            hom_win,  # homodyne_resolution = window size
            phi_vec,
            V_matrix,
            dx_weights,
            sim_cutoff,
            True,  # homodyne_window_is_none=True (Point Mode)
            False,
            False,  # homodyne_resolution_is_none=False (Apply Scaling)
        )

        # 3. Apply Final Global Gaussian (in sim_cutoff)
        final_state_transformed = jax_apply_final_gaussian(
            final_state, params["final_gauss"], sim_cutoff
        )

        # --- Dynamic Limits Logic ---
        leakage_penalty = 0.0

        if use_correction:
            # Calculate Leakage: Probability mass outside 'cutoff'
            # final_state_transformed is vector in sim_cutoff
            probs = jnp.abs(final_state_transformed) ** 2
            prob_total = jnp.sum(probs)
            # Safe normalize if tiny?

            # Mass in [0, cutoff]
            mass_in = jnp.sum(probs[:cutoff])
            leakage = 1.0 - (mass_in / jnp.maximum(prob_total, 1e-12))

            # Truncate state for evaluation
            # Simply slice [0:cutoff]
            state_trunc = final_state_transformed[:cutoff]

            # Renormalize truncated state
            norm_trunc = jnp.linalg.norm(state_trunc)
            state_eval = jax.lax.cond(
                norm_trunc > 1e-9, lambda x: x / norm_trunc, lambda x: x, state_trunc
            )

            # Leakage Penalty
            # If leakage > 0.05 (5%), apply heavy penalty?
            # Or continuous penalty?
            # Continuous: penalty = leakage * 100.0
            leakage_penalty = leakage * 2.0

        else:
            state_eval = final_state_transformed

        # 4. Expectation Value (using original operator in 'cutoff')
        # Operator shape: (cutoff, cutoff)
        # state_eval shape: (cutoff,)
        op_psi = operator @ state_eval
        raw_exp_val = jnp.real(jnp.vdot(state_eval, op_psi))

        # Penalize very low probability states (hard cutoff at 10^-40)
        # This allows exploration down to 10^-40 while preserving exp_val interpretation
        exp_val = jnp.where(joint_prob > 1e-40, raw_exp_val, jnp.inf)

        # Add Leakage Penalty
        exp_val += leakage_penalty

        # 5. Fitness & Descriptors
        prob_clipped = jnp.maximum(joint_prob, 1e-45)
        log_prob = -jnp.log10(prob_clipped)
        total_photons = total_sum_pnr

        # Fitness: [-exp_val, -log_prob, -complexity, -total_photons]
        fitness = jnp.array([-exp_val, -log_prob, -active_modes, -total_photons])

        # Descriptor: [active_modes, max_pnr, total_photons]
        descriptor = jnp.array([active_modes, max_pnr, total_photons])

        return fitness, descriptor

    fitnesses, descriptors = jax.vmap(score_one)(genotypes)
    return fitnesses, descriptors


def jax_scoring_fn_batch(
    genotypes: jnp.ndarray,
    cutoff: int,
    operator: jnp.ndarray,
    genotype_name: str = "A",
    genotype_config: Dict = None,
    correction_cutoff: int = None,
    pnr_max: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batched scoring function for QDax.
    Dispatches to multi-device pmap if devices > 1, else uses jit.
    """
    n_devices = jax.local_device_count()

    # Debug Log (only printed during tracing/exec)
    # Ideally should be controlled by a debug flag, but simple print here is fine for now
    # as this function is called once per chunk usually.
    # Actually it's called every iteration if we put it inside loop, but here it's inside `score_batch`.
    # `jax_scoring_fn_batch` is called by `scoring_fn` in `run_mome.py`.
    # It constructs the graph.
    # We want runtime logging.
    # Using jax.debug.print is better for runtime.

    def log_debug(msg, *args):
        # Fallback if host_callback/debug.print is tricky?
        # Standard print happens at trace time.
        pass

    if n_devices <= 1:
        # Fallback to single-device JIT
        # Ensure config is hashable (dict -> tuple of items)
        if genotype_config is not None and isinstance(genotype_config, dict):
            config_hashable = tuple(sorted(genotype_config.items()))
        else:
            config_hashable = genotype_config

        return _score_batch_shard(
            genotypes,
            cutoff,
            operator,
            genotype_name,
            config_hashable,
            correction_cutoff,
            pnr_max,  # Pass pnr_max
        )

    # Multi-GPU Logic
    # 1. Pad genotypes to be divisible by n_devices
    # print(f"DEBUG: Using {n_devices} JAX devices.")
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

    # NOTE: pmap args: (shard, cutoff, operator, genotype_name)
    # in_axes=(0, None, None, None) -> shard divides axis 0, others replicated.

    # score_shard args: (genotypes, cutoff, operator, genotype_name, genotype_config, correction_cutoff)
    # Argnum Map:
    # 0: genotypes (Sharded)
    # 1: cutoff (Static)
    # 2: operator (Broadcasted)
    # 3: genotype_name (Static)
    # 4: genotype_config (Static)
    # 5: correction_cutoff (Static)

    # operator is array, so we BROADCAST it (None in in_axes).
    # Cutoff, name, config, correction are constants -> STATIC.

    pmapped_fn = jax.pmap(
        _score_batch_shard,
        in_axes=(0, None, None, None, None, None, None),
        static_broadcasted_argnums=(
            1,
            3,
            4,
            5,
            6,
        ),  # cutoff(1), name(3), config(4), correction(5), pnr_max(6) static
    )

    # Ensure config is hashable for pmap static arg
    if genotype_config is not None and isinstance(genotype_config, dict):
        config_hashable = tuple(sorted(genotype_config.items()))
    else:
        config_hashable = genotype_config

    # Explicitly log start of pmap execution
    # jax.debug.print("DEBUG: Starting pmap on {n} devices, batch={b}", n=n_devices, b=batch_size)

    fitnesses_sharded, descriptors_sharded = pmapped_fn(
        g_sharded,
        cutoff,
        operator,
        genotype_name,
        config_hashable,
        correction_cutoff,
        pnr_max,
    )

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
