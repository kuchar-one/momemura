import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Tuple

from src.genotypes.genotypes import get_genotype_decoder

from src.simulation.jax.herald import (
    vacuum_covariance,
    passive_unitary_to_symplectic,
    complex_alpha_to_qp,
)
from src.simulation.jax.composer import jax_hermite_phi_matrix

# Updated Constants
MAX_MODES = 6  # 1 Signal + 5 Controls
# These are max limits for static array sizing if needed, but JAX code below is mostly dynamic
# except for pnr masking which uses 'pnr' array size.


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


def jax_clements_unitary(phases: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Constructs NxN unitary using Clements decomposition (Rectangular Mesh).
    Requires N^2 phases.
    The phases vector is assumed to contain [theta_1, phi_1, ..., theta_K, phi_K, varphi_1, ..., varphi_N]
    Wait, Clements uses N(N-1) parameters + N phases = N^2.

    Structure: M layers of beam splitters.
    For N=3:
       L0: BS(0,1)
       L1: BS(1,2)
       L2: BS(0,1)

    We simply iterate through the rectangular mesh similar to `jax_interferometer_unitary` but
    ensure we consume exactly (N^2 - N)/2 pairs of (theta, phi) plus N varphis.
    Actually, standard Clements/Reck parameterization:
    N(N-1)/2 beam splitters, each has theta, phi. Total N(N-1).
    Plus N phases at input/output.
    Total params = N^2 - N + N = N^2.

    Input `phases` has length N^2.
    """
    U = jnp.eye(N, dtype=jnp.complex_)

    # Check if N is static or traced? Usually static.
    # If N is small (<=6), we can unroll.

    param_idx = 0

    # Iterate Rectangular Mesh
    for s in range(N):
        start = 0 if (s % 2 == 0) else 1
        for a in range(start, N - 1, 2):
            # Each BS consumes 2 phases: theta, phi
            th = phases[param_idx]
            ph = phases[param_idx + 1]
            param_idx += 2

            B = jax_beamsplitter_2x2(th, ph)

            # Update rows [a, a+1]
            # U = B @ U  (Apply from left? Or right? Usually left builds up from output?)
            # Standard Clements formulation: U = D * T_N * ... * T_1
            # Let's apply B to the sub-block of U.
            sub_U = U[a : a + 2, :]
            new_sub_U = B @ sub_U
            U = U.at[a : a + 2, :].set(new_sub_U)

    # Final Phases (Diagonal)
    # Are we sure we consumed exactly N^2 - N params?
    # Mesh size logic: s in 0..N-1 (N layers).
    # Odd N=3:
    # s=0: a=0 (0,1). Limit 2. Range 0,2 no. Just 0.
    # s=1: a=1 (1,2). Limit 2. Just 1.
    # s=2: a=0 (0,1).
    # Total BS: 3. Params 6.
    # N^2 = 9. Varphis = 3. Total 9. Correct.

    # Last N params are varphis
    varphis = phases[param_idx : param_idx + N]
    phases_diag = jnp.exp(1j * varphis)
    U = U * phases_diag[:, None]  # Broadcast over columns

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

    # 1. Squeeze S(z)
    term1 = jnp.exp(-2j * phi) * (a @ a)
    term2 = jnp.exp(2j * phi) * (adag @ adag)
    K_squeeze = (r / 2.0) * (term1 - term2)
    U_squeeze = jax.scipy.linalg.expm(K_squeeze)

    # 2. Rotation R(varphi)
    n_op = jnp.arange(cutoff)
    U_rot = jnp.diag(jnp.exp(1j * n_op * varphi))

    # 3. Displacement D(alpha)
    K_disp = disp * adag - jnp.conj(disp) * a
    U_disp = jax.scipy.linalg.expm(K_disp)

    U_final = U_disp @ U_rot @ U_squeeze

    return U_final @ state_vec


def jax_get_gaussian_moments(
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute (mu, cov) for a General Gaussian state defined by params.
    """
    r_vec = params["r"]  # (N,)
    phases_vec = params["phases"]  # (N^2,)
    disp_vec = params["disp"]  # (N,) complex

    N = r_vec.shape[0]
    hbar = 2.0

    # 1. Squeezing Symplectic
    # S_sq = diag(e^-r, e^r) (x-squeezing)
    exp_minus_r = jnp.exp(-r_vec)
    exp_plus_r = jnp.exp(r_vec)
    S_sq = jnp.diag(jnp.concatenate([exp_minus_r, exp_plus_r]))

    # 2. Passive Unitary Symplectic
    U_pass = jax_clements_unitary(phases_vec, N)
    S_pass = passive_unitary_to_symplectic(U_pass)

    # 3. Total Symplectic
    S_total = S_pass @ S_sq

    # Moments calculated above via jax_get_gaussian_moments

    # 5. Mask PNR (Effective PNR vector)

    # Displacement
    # alpha vector to mu (xp)
    mu = complex_alpha_to_qp(disp_vec, hbar)

    # Covariance
    cov_vac = vacuum_covariance(N, hbar)
    cov = S_total @ cov_vac @ S_total.T

    return mu, cov


def jax_get_heralded_state(
    params: Dict[str, jnp.ndarray], cutoff: int, pnr_max: int = 3
):
    """
    Computes Herald State from General Gaussian Leaf parameters.

    The number of modes is determined by n_ctrl:
    - n_ctrl=0: 1-mode Gaussian (signal only), no heralding, prob=1.0
    - n_ctrl=1: 2-mode Gaussian, herald on 1 control mode
    - n_ctrl=2: 3-mode Gaussian, herald on 2 control modes
    """
    from src.simulation.jax.herald import (
        jax_get_full_amplitudes,
        vacuum_covariance,
        passive_unitary_to_symplectic,
        complex_alpha_to_qp,
    )

    n_ctrl_eff = params["n_ctrl"]
    pnr_vec = params["pnr"]
    r_vec = params["r"]
    phases_vec = params["phases"]
    disp_vec = params["disp"]
    hbar = 2.0

    # Ensure params are padded to support max modes (3) for jax.lax.switch branches
    # This prevents shape mismatches if the genotype was configured with fewer modes
    # but the static graph traces the n_ctrl=2 branch.
    MAX_MODES = 3

    pad_r = MAX_MODES - r_vec.shape[0]
    if pad_r > 0:
        r_vec = jnp.pad(r_vec, (0, pad_r), constant_values=0.0)

    pad_phases = (MAX_MODES * MAX_MODES) - phases_vec.shape[0]
    if pad_phases > 0:
        phases_vec = jnp.pad(phases_vec, (0, pad_phases), constant_values=0.0)

    pad_disp = MAX_MODES - disp_vec.shape[0]
    if pad_disp > 0:
        disp_vec = jnp.pad(disp_vec, (0, pad_disp), constant_values=0.0)

    def _build_gaussian_moments(N_modes):
        """Build Gaussian moments for N_modes using first N_modes params."""
        # Squeezing symplectic
        r_N = r_vec[:N_modes]
        exp_minus_r = jnp.exp(-r_N)
        exp_plus_r = jnp.exp(r_N)
        S_sq = jnp.diag(jnp.concatenate([exp_minus_r, exp_plus_r]))

        # Passive unitary (extract N^2 phases for NxN unitary)
        phases_N = phases_vec[: N_modes * N_modes]
        U_pass = jax_clements_unitary(phases_N, N_modes)
        S_pass = passive_unitary_to_symplectic(U_pass)

        # Total symplectic
        S_total = S_pass @ S_sq

        # Displacement
        disp_N = disp_vec[:N_modes]
        mu = complex_alpha_to_qp(disp_N, hbar)

        # Covariance
        cov_vac = vacuum_covariance(N_modes, hbar)
        cov = S_total @ cov_vac @ S_total.T

        return mu, cov

    def _compute_n0(_):
        """n_ctrl=0: 1-mode Gaussian, no heralding, prob=1.0"""
        mu, cov = _build_gaussian_moments(1)
        # For N=1, use recurrence with empty pnr tuple
        H = jax_get_full_amplitudes(mu, cov, (), cutoff, hbar)
        # H has shape (cutoff,) - just the signal amplitudes
        prob = jnp.sum(jnp.abs(H) ** 2)
        vec_norm = jax.lax.cond(
            prob > 0,
            lambda _: H / jnp.sqrt(prob),
            lambda _: jnp.zeros(cutoff, dtype=H.dtype),
            None,
        )
        # For 1-mode pure Gaussian, prob should be 1.0
        return vec_norm, prob, jnp.float32(0.0), jnp.float32(0.0)

    def _compute_n1(_):
        """n_ctrl=1: 2-mode Gaussian, herald on 1 control mode"""
        mu, cov = _build_gaussian_moments(2)
        pnr_tuple = (pnr_max,)
        H = jax_get_full_amplitudes(mu, cov, pnr_tuple, cutoff, hbar)
        # H shape: (cutoff, pnr_max+1) - index with pnr[0]
        pnr_idx = jnp.clip(pnr_vec[0], 0, pnr_max).astype(jnp.int32)
        vec = H[:, pnr_idx]
        prob = jnp.sum(jnp.abs(vec) ** 2)
        vec_norm = jax.lax.cond(
            prob > 0,
            lambda _: vec / jnp.sqrt(prob),
            lambda _: jnp.zeros(cutoff, dtype=vec.dtype),
            None,
        )
        max_pnr = pnr_vec[0].astype(jnp.float32)
        total_pnr = pnr_vec[0].astype(jnp.float32)
        return vec_norm, prob, max_pnr, total_pnr

    def _compute_n2(_):
        """n_ctrl=2: 3-mode Gaussian, herald on 2 control modes"""
        mu, cov = _build_gaussian_moments(3)
        pnr_tuple = (pnr_max, pnr_max)
        H = jax_get_full_amplitudes(mu, cov, pnr_tuple, cutoff, hbar)
        # H shape: (cutoff, pnr_max+1, pnr_max+1)
        pnr_0 = jnp.clip(pnr_vec[0], 0, pnr_max).astype(jnp.int32)
        pnr_1 = jnp.clip(pnr_vec[1], 0, pnr_max).astype(jnp.int32)
        vec = H[:, pnr_0, pnr_1]
        prob = jnp.sum(jnp.abs(vec) ** 2)
        vec_norm = jax.lax.cond(
            prob > 0,
            lambda _: vec / jnp.sqrt(prob),
            lambda _: jnp.zeros(cutoff, dtype=vec.dtype),
            None,
        )
        max_pnr = jnp.maximum(pnr_vec[0], pnr_vec[1]).astype(jnp.float32)
        total_pnr = (pnr_vec[0] + pnr_vec[1]).astype(jnp.float32)
        return vec_norm, prob, max_pnr, total_pnr

    # Dispatch based on n_ctrl
    # n_ctrl_eff is 0, 1, or 2 (clipped to valid range)
    n_ctrl_clamped = jnp.clip(n_ctrl_eff, 0, 2).astype(jnp.int32)

    vec_out, prob_out, max_pnr_out, total_pnr_out = jax.lax.switch(
        n_ctrl_clamped,
        [_compute_n0, _compute_n1, _compute_n2],
        None,
    )

    return (
        vec_out,
        prob_out,
        1.0,
        max_pnr_out.astype(jnp.float32),
        total_pnr_out.astype(jnp.float32),
        1.0,
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
    genotype_config: Any = None,
    correction_cutoff: int = None,
    pnr_max: int = 3,
    gs_eig: float = -4.0,  # Default safe-ish value if not passed
    gaussian_limit: float = 2.0 / 3.0,  # Gaussian achievable limit (2/3 for GKP)
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    JIT-compiled scoring function for a single batch shard.
    Returns: (fitnesses, descriptors, extras)
    extras contains:
      - gradients: Gradient of expectation value w.r.t genotype
      - leakage: 1 - norm_squared inside cutoff
      - raw_expectation: Unpenalized expectation value
      - joint_probability: Probability of the heralded event
      - pnr_cost: Total pnr cost
    """
    if genotype_config is not None and isinstance(genotype_config, tuple):
        config_dict = dict(genotype_config)
    else:
        config_dict = genotype_config

    depth = 3
    if config_dict is not None:
        depth = int(config_dict.get("depth", 3))

    decoder = get_genotype_decoder(genotype_name, depth=depth, config=config_dict)

    use_correction = (correction_cutoff is not None) and (correction_cutoff > cutoff)
    sim_cutoff = correction_cutoff if use_correction else cutoff

    def loss_fn(g):
        """
        Returns (Loss, Aux)
        Loss = ExpVal (to be minimized, so we return it directly as JAX grad computes grad of this)
        Aux = (fitness_rest, descriptor, extra_metrics)
        """
        params = decoder.decode(g, sim_cutoff)

        # 1. Get Leaf States
        # Conditionally compute heralded states only for active leaves
        # Inactive leaves return dummy values that will be ignored by mixing logic
        leaf_active = params["leaf_active"]

        # Use JAX's dynamic complex dtype (matches herald.py internal dtype)
        complex_dtype = (jnp.zeros(1) + 1j).dtype
        # JAX uses float64 by default for Python floats (like 1.0)
        float_dtype = jnp.float64

        def conditional_herald(leaf_p, is_active):
            """Compute heralded state only if active, otherwise return dummy."""

            def compute(_):
                return jax_get_heralded_state(
                    leaf_p, cutoff=sim_cutoff, pnr_max=pnr_max
                )

            def skip(_):
                # Return dummy values with matching dtypes
                dummy_vec = jnp.zeros(sim_cutoff, dtype=complex_dtype)
                return (
                    dummy_vec,
                    jnp.array(0.0, dtype=float_dtype),
                    jnp.array(1.0, dtype=float_dtype),
                    jnp.array(0.0, dtype=jnp.float32),  # max_pnr is float32
                    jnp.array(0.0, dtype=jnp.float32),  # total_pnr is float32
                    jnp.array(0.0, dtype=float_dtype),
                )

            return jax.lax.cond(is_active, compute, skip, None)

        # vmap over leaves with active flag
        leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes = jax.vmap(
            conditional_herald
        )(params["leaf_params"], leaf_active)

        # --- Leaf Upper-Mass Penalty ---
        # Check leaf states at BASE cutoff (not sim_cutoff) for truncation issues
        # This catches high-photon leaf states that will cause divergent evolution
        # through the mixing tree due to BS unitary truncation
        leaf_penalty = 0.0
        if use_correction:
            # Compute leaf states at base cutoff
            def compute_base_leaf(leaf_p, is_active):
                def compute(_):
                    vec, _, _, _, _, _ = jax_get_heralded_state(
                        leaf_p, cutoff=cutoff, pnr_max=pnr_max
                    )
                    return vec

                def skip(_):
                    return jnp.zeros(cutoff, dtype=complex_dtype)

                return jax.lax.cond(is_active, compute, skip, None)

            base_leaf_vecs = jax.vmap(compute_base_leaf)(
                params["leaf_params"], leaf_active
            )

            # Compute upper mass for each active leaf at base cutoff
            half_cutoff = cutoff // 2

            def leaf_upper_mass(vec, is_active):
                probs = jnp.abs(vec) ** 2
                upper = jnp.sum(probs[half_cutoff:])
                # Only count for active leaves
                return jnp.where(is_active, upper, 0.0)

            leaf_upper_masses = jax.vmap(leaf_upper_mass)(base_leaf_vecs, leaf_active)
            max_leaf_upper = jnp.max(leaf_upper_masses)

            # Penalty: heavy penalty for any leaf with >5% upper mass
            # max_leaf_upper=0.10 -> penalty ~0.25
            # max_leaf_upper=0.20 -> penalty ~0.60
            # max_leaf_upper=0.40 -> penalty ~2.0
            # DEACTIVATED: Now relying on conservative hx_scale instead
            # leaf_penalty = max_leaf_upper * 1.0 + (max_leaf_upper**2) * 10.0
            leaf_penalty = 0.0  # Deactivated

        # 2. Superblock
        hom_x = params["homodyne_x"]
        hom_win = params["homodyne_window"]

        # Handle hom_x shape (scalar vs vector)
        hom_xs = jnp.atleast_1d(hom_x)
        phi_mat = jax_hermite_phi_matrix(hom_xs, sim_cutoff)

        if jnp.ndim(hom_x) == 0:
            phi_vec = phi_mat[:, 0]
        else:
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
            hom_win,
            phi_vec,
            V_matrix,
            dx_weights,
            sim_cutoff,
            True,
            False,
            False,
        )

        # 3. Final Global Gaussian
        final_state_transformed = jax_apply_final_gaussian(
            final_state, params["final_gauss"], sim_cutoff
        )

        # --- Dynamic Limits Logic ---
        leakage_penalty = 0.0
        leakage_val = 0.0
        structure_penalty = 0.0

        if use_correction:
            probs = jnp.abs(final_state_transformed) ** 2
            prob_total = jnp.sum(probs)
            mass_in = jnp.sum(probs[:cutoff])
            leakage_val = 1.0 - (mass_in / jnp.maximum(prob_total, 1e-12))

            state_trunc = final_state_transformed[:cutoff]
            norm_trunc = jnp.linalg.norm(state_trunc)
            # Re-normalize if meaningful norm exists
            state_eval = jax.lax.cond(
                norm_trunc > 1e-9, lambda x: x / norm_trunc, lambda x: x, state_trunc
            )
            leakage_penalty = leakage_val * 2.0

            # --- Structural Fidelity Penalty ---
            # Run superblock at BASE cutoff and check native upper_mass.
            # This catches divergence from intermediate mixing/homodyne operations
            # that produce high-n states even when input leaves are fine.
            # Example: Solution 511 has leaves with mean_n<5, but after BS+homodyne
            # mixing, one node has mean_n=10.14 and 14% upper mass.

            # Compute leaf states at base cutoff
            def compute_base_leaf_for_structure(leaf_p, is_active):
                def compute(_):
                    return jax_get_heralded_state(
                        leaf_p, cutoff=cutoff, pnr_max=pnr_max
                    )

                def skip(_):
                    return (
                        jnp.zeros(cutoff, dtype=complex_dtype),
                        jnp.array(0.0, dtype=float_dtype),
                        jnp.array(1.0, dtype=float_dtype),
                        jnp.array(0.0, dtype=jnp.float32),
                        jnp.array(0.0, dtype=jnp.float32),
                        jnp.array(0.0, dtype=float_dtype),
                    )

                return jax.lax.cond(is_active, compute, skip, None)

            (
                base_leaf_vecs_s,
                base_leaf_probs_s,
                _,
                base_leaf_max_pnrs_s,
                base_leaf_total_pnrs_s,
                base_leaf_modes_s,
            ) = jax.vmap(compute_base_leaf_for_structure)(
                params["leaf_params"], leaf_active
            )

            # Compute phi_vec at base cutoff
            hom_xs_base = jnp.atleast_1d(hom_x)
            phi_mat_base = jax_hermite_phi_matrix(hom_xs_base, cutoff)
            if jnp.ndim(hom_x) == 0:
                phi_vec_base = phi_mat_base[:, 0]
            else:
                phi_vec_base = phi_mat_base.T

            V_matrix_base = jnp.zeros((cutoff, 1))
            dx_weights_base = jnp.zeros(1)

            (base_final_state, _, _, _, _, _, _) = jax_superblock(
                base_leaf_vecs_s,
                base_leaf_probs_s,
                params["leaf_active"],
                base_leaf_max_pnrs_s,
                base_leaf_total_pnrs_s,
                base_leaf_modes_s,
                params["mix_params"],
                hom_x,
                hom_win,
                hom_win,
                phi_vec_base,
                V_matrix_base,
                dx_weights_base,
                cutoff,
                True,
                False,
                False,
            )

            # Check upper mass of state BEFORE final Gaussian
            # The final Gaussian can hide divergence (e.g., squeezing transforms
            # a high-n state to appear low-n). We must catch the issue at the
            # mixing tree output, not after the final transformation.
            half_cutoff = cutoff // 2
            probs_base = jnp.abs(base_final_state) ** 2
            danger_mass = jnp.sum(probs_base[half_cutoff:])

            # Combined linear + quadratic penalty
            # DEACTIVATED: Now relying on conservative hx_scale instead
            # structure_penalty = danger_mass * 1.0 + (danger_mass**2) * 10.0
            structure_penalty = 0.0  # Deactivated
        else:
            state_eval = final_state_transformed

        # 4. Expectation Value
        # Gradient Target: minimize <Psi|Op|Psi>
        # We need real( <Psi | Op | Psi> )
        op_psi = operator @ state_eval
        raw_exp_val = jnp.real(jnp.vdot(state_eval, op_psi))

        # Apply penalty for invalid states (prob ~ 0)
        # Gradient should flow through raw_exp_val where valid
        exp_val = jnp.where(joint_prob > 1e-40, raw_exp_val, 0.0)

        # 5. Fitness & Descriptors
        # CLIP PROBABILITY at 1.0 to prevent explosion incentive
        # If prob > 1.0, log10(prob) > 0, which increases fitness (since fitness uses log10(p)).
        # We must clip it to stop the optimizer from exploiting numerical errors.
        prob_safe = jnp.maximum(joint_prob, 1e-45)
        prob_capped = jnp.minimum(prob_safe, 1.0)

        # Calculate Negative Log Likelihood (NLL)
        log_prob = -jnp.log10(prob_capped)

        # Add massive penalty if prob > 1.0 + epsilon (tolerance for float errors)
        # We add this to log_prob (which is minimized/penalized part of loss, or negative of fitness)
        violation = jnp.maximum(joint_prob - 1.0, 0.0)
        penalty = jnp.where(violation > 1e-4, jnp.inf, 0.0)

        log_prob = log_prob + penalty

        # --- Physics-Based Artifact Penalty ---
        # If expectation < gaussian_limit AND no photons detected, this is an artifact.
        # You cannot beat the Gaussian limit without non-Gaussian resources (PNR detection).
        # This catches cases where numerical artifacts create false sub-Gaussian states.
        # Penalty on BOTH objectives: exp_val=inf (bad expectation) and log_prob=inf (zero probability)
        is_below_gaussian = exp_val < gaussian_limit
        no_photons_detected = total_sum_pnr < 0.5  # Effectively 0
        is_artifact = jnp.logical_and(is_below_gaussian, no_photons_detected)
        artifact_penalty = jnp.where(is_artifact, jnp.inf, 0.0)
        exp_val = exp_val + artifact_penalty  # Make expectation infinitely bad
        log_prob = log_prob + artifact_penalty  # Make probability zero (inf NLL)

        total_photons = total_sum_pnr

        # Loss Function for Gradient Methods
        # Default weights: 1.0 Expectation, 0.0 Probability
        w_exp = 1.0
        w_prob = 0.0
        if config_dict is not None:
            w_exp = float(config_dict.get("alpha_expectation", 1.0))
            w_prob = float(config_dict.get("alpha_probability", 0.0))

        # Weighted Sum -> Augmented Tchebycheff
        # Note: We minimize loss.
        # Utopia points
        # Expectation: gs_eig - eps
        # Probability: -eps (Ideal -logP = 0)
        eps = 0.01
        z_star_exp = gs_eig - eps
        z_star_prob = -eps
        rho = 0.01

        # Terms (Minimize distance to Utopia)
        # We want to minimize (exp_val - z*) and (log_prob - z*)
        # Since z* is strictly lower bound, value - z* is positive.
        d_exp = w_exp * jnp.abs(exp_val - z_star_exp)
        d_prob = w_prob * jnp.abs(log_prob - z_star_prob)

        loss_val = (
            jnp.maximum(d_exp, d_prob)
            + rho * (d_exp + d_prob)
            + leakage_penalty
            + structure_penalty
            + leaf_penalty
        )

        # Construct Fitness: [-Exp, -LogP, -Complex, -Photons]
        # Use computed loss_val which includes penalty? or separate?
        # MOME usually uses penalized expectation.
        final_exp = (
            jnp.where(joint_prob > 1e-40, raw_exp_val, jnp.inf)
            + leakage_penalty
            + structure_penalty
            + leaf_penalty
        )

        # Fitnesses for QDax (Maximization)
        fitness_rest = jnp.array([-log_prob, -active_modes, -total_photons])

        # We stick -final_exp into the auxiliary return so we can reconstruct full fitness
        # But loss_fn returns 'loss' to minimize.

        descriptor = jnp.array([active_modes, max_pnr, total_photons])

        aux = {
            "fitness_0": -final_exp,  # Maximize this
            "fitness_rest": fitness_rest,
            "descriptor": descriptor,
            "leakage": leakage_val,
            "structure_penalty": structure_penalty,
            "leaf_penalty": leaf_penalty,
            "raw_expectation": raw_exp_val,
            "joint_probability": joint_prob,
            "pnr_cost": total_photons,
        }

        return jnp.real(loss_val), aux

    # vmap value_and_grad over batch
    # value_and_grad returns (loss, aux), grad
    # internal tuple structure: ((loss, aux), grad)
    # vmap output: ((loss_batch, aux_batch), grad_batch)
    (loss_batch, aux_batch), grads_batch = jax.vmap(
        jax.value_and_grad(loss_fn, has_aux=True)
    )(genotypes)

    # Reconstruct Fitnesses
    # fitness_0 shape (N,)
    f0 = aux_batch["fitness_0"][:, None]  # (N, 1)
    f_rest = aux_batch["fitness_rest"]  # (N, 3)
    fitnesses = jnp.concatenate([f0, f_rest], axis=1)

    descriptors = aux_batch["descriptor"]

    extras = {
        "gradients": grads_batch,
        "leakage": aux_batch["leakage"],
        "raw_expectation": aux_batch["raw_expectation"],
        "joint_probability": aux_batch["joint_probability"],
        "pnr_cost": aux_batch["pnr_cost"],
    }

    return fitnesses, descriptors, extras


def jax_scoring_fn_batch(
    genotypes: jnp.ndarray,
    cutoff: int,
    operator: jnp.ndarray,
    genotype_name: str = "A",
    genotype_config: Dict = None,
    correction_cutoff: int = None,
    pnr_max: int = 3,
    gs_eig: float = -4.0,
    gaussian_limit: float = 2.0 / 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Batched scoring function for QDax.
    Returns: (fitnesses, descriptors, extras)
    """
    n_devices = jax.local_device_count()

    if n_devices <= 1:
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
            pnr_max,
            gs_eig,
            gaussian_limit,
        )

    # Multi-GPU Logic
    batch_size = genotypes.shape[0]
    remainder = batch_size % n_devices
    padding = (n_devices - remainder) if remainder > 0 else 0

    if padding > 0:
        pad_block = jnp.repeat(genotypes[:1], padding, axis=0)
        g_padded = jnp.concatenate([genotypes, pad_block], axis=0)
    else:
        g_padded = genotypes

    shard_size = g_padded.shape[0] // n_devices
    g_sharded = g_padded.reshape((n_devices, shard_size, -1))

    # pmap needs to handle pytree return (tuple of arrays/dicts)
    # _score_batch_shard returns (fit, desc, extras)
    pmapped_fn = jax.pmap(
        _score_batch_shard,
        in_axes=(0, None, None, None, None, None, None, None, None),
        static_broadcasted_argnums=(1, 3, 4, 5, 6, 7, 8),
    )

    if genotype_config is not None and isinstance(genotype_config, dict):
        config_hashable = tuple(sorted(genotype_config.items()))
    else:
        config_hashable = genotype_config

    fitnesses_sharded, descriptors_sharded, extras_sharded = pmapped_fn(
        g_sharded,
        cutoff,
        operator,
        genotype_name,
        config_hashable,
        correction_cutoff,
        pnr_max,
        gs_eig,
        gaussian_limit,
    )

    # Reshape results
    # fitnesses: (Devices, Shard, 4) -> (Batch, 4)
    fitnesses = fitnesses_sharded.reshape((-1, 4))
    descriptors = descriptors_sharded.reshape((-1, 3))

    # Extras is a Dict of sharded arrays. Need to reshape each value.
    # extras_sharded = {"gradients": (Dev, Shard, D), ...}
    def reshape_extra(x):
        return x.reshape((-1, *x.shape[2:]))

    extras = jax.tree_util.tree_map(reshape_extra, extras_sharded)

    if padding > 0:
        fitnesses = fitnesses[:batch_size]
        descriptors = descriptors[:batch_size]
        extras = jax.tree_util.tree_map(lambda x: x[:batch_size], extras)

    return fitnesses, descriptors, extras
