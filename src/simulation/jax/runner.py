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
    U = jnp.eye(N, dtype=jnp.complex64)

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


def jax_get_heralded_state(
    params: Dict[str, jnp.ndarray], cutoff: int, pnr_max: int = 3
):
    """
    Computes Herald State from General Gaussian Leaf parameters.
    Leaf Params:
      - n_ctrl: scalar int
      - r: (N,) squeezing
      - phases: (N^2,) unitary phases
      - disp: (N,) complex displacement
      - pnr: (N-1,) pnr outcomes

    Logic:
      1. Construct S_total = S_pass @ S_sq
      2. Construct mu = disp_vec_real_imag
      3. Covariance sigma = 0.5 * S S^T
      4. Herald on last N-1 modes
    """
    # Extract params
    n_ctrl_eff = params["n_ctrl"]  # (1,) or scalar
    r_vec = params["r"]  # (N,)
    phases_vec = params["phases"]  # (N^2,)
    disp_vec = params["disp"]  # (N,) complex
    pnr_vec = params["pnr"]  # (Nc_max,)

    # Infer dimension N from r_vec
    N = r_vec.shape[0]
    N_C = N - 1

    hbar = 2.0

    # 1. Squeezing Symplectic
    # r_vec contains squeeze params.
    # ordering x1..xN, p1..pN ?
    # S_sq = diag(e^-r, e^r) if standard.
    # Note: `two_mode_squeezer` usually is r, -r.
    # Let's assume standard x-squeezing: var(x) = e^{-2r}.
    # Diag should be [e^-r1 ... e^-rN, e^r1 ... e^rN] for Block Basis.
    # Or interleaved?
    # `src/simulation/jax/herald.py` usually assumes Block Basis (x...p...).
    # Let's verify vacuum_covariance.
    # It returns identity.

    # Construct diagonal S_sq
    exp_minus_r = jnp.exp(-r_vec)
    exp_plus_r = jnp.exp(r_vec)
    S_sq = jnp.diag(jnp.concatenate([exp_minus_r, exp_plus_r]))

    # 2. Passive Unitary Symplectic
    # Construct U_pass from phases
    U_pass = jax_clements_unitary(phases_vec, N)
    S_pass = passive_unitary_to_symplectic(U_pass)

    # 3. Total Symplectic
    S_total = S_pass @ S_sq

    # 4. Covariance & Mean
    # Vacuum cov = (hbar/2) * I?
    # Usually herald.py `vacuum_covariance` returns `(hbar/2) * I`
    cov_vac = vacuum_covariance(N, hbar)
    # sigma = S cov_vac S^T
    # If cov_vac is proportional to I, then S cov S^T = (hbar/2) S S^T
    cov = S_total @ cov_vac @ S_total.T

    # Displacement
    # mu_disp needs real representation: [Re(d), Im(d)] (Block Basis)

    # Helper `complex_alpha_to_qp` takes alpha and returns [q, p] * sqrt(2*hbar).
    r_disp = complex_alpha_to_qp(disp_vec, hbar)

    # Rotated displacement?
    # User plan: D @ U @ S |0>.
    # Displaced state has mean = r_disp
    # Covariance is independent of displacement.
    mu = r_disp

    # 5. Herald
    # Control modes indices: 1..N-1 (0 is signal)
    # PNR Masking
    # Map pnr_vec to effective pnr
    # pnr_vec has size N-1? Or fixed large size?
    # Usually Genotype decodes to fixed max size (e.g. 5 controls).
    # But N might vary per run config?
    # Here N is determined by r_vec.

    ctrl_indices_local = jnp.arange(N_C)
    pnr_effective = jnp.where(ctrl_indices_local < n_ctrl_eff, pnr_vec[:N_C], 0)

    # Heralding Function
    # We use jax_get_full_amplitudes
    from src.simulation.jax.herald import jax_get_full_amplitudes

    # We need to pass max_pnr_tuple.
    # Since we can't iterate dynamically for tuple creation in JIT if N_C is dynamic,
    # we assume N_C is static (defined by genotype config `modes`).
    # `jax_get_heralded_state` handles static shapes usually.

    max_pnr_sequence = [pnr_max] * N_C
    max_pnr_tuple = tuple(max_pnr_sequence)

    # Return Amplitudes tensor H (cutoff, pnr_max+1, ..., pnr_max+1)
    H_full = jax_get_full_amplitudes(mu, cov, max_pnr_tuple, cutoff, hbar)

    # Extract the slice corresponding to pnr_effective
    # H_full shape: (cutoff, D_ctrl, D_ctrl, ...) where D_ctrl = pnr_max+1
    H_flat = H_full.reshape(cutoff, -1)

    D_ctrl = pnr_max + 1

    # Compute flat index for [pnr_0, pnr_1, ...]
    # Stride calculation
    # Index = sum( p[i] * D^(N_C - 1 - i) ) ?
    # Note: `jax_get_full_amplitudes` returns indices in order of modes 1..N.
    # So H[k, n1, n2...]

    flat_idx = 0
    for i in range(N_C):
        p = pnr_effective[i]
        power = N_C - 1 - i
        stride = D_ctrl**power
        flat_idx += p * stride

    vec_slice = H_flat[:, flat_idx]

    # Probability
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
        jnp.max(pnr_effective).astype(jnp.float32),
        jnp.sum(pnr_effective).astype(jnp.float32),
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled scoring function for a single batch shard.
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

    def score_one(g):
        params = decoder.decode(g, sim_cutoff)

        # 1. Get Leaf States
        # vmap over leaf_params
        leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes = jax.vmap(
            partial(jax_get_heralded_state, cutoff=sim_cutoff, pnr_max=pnr_max)
        )(params["leaf_params"])

        # 2. Superblock
        hom_x = params["homodyne_x"]
        hom_win = params["homodyne_window"]

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

        if use_correction:
            probs = jnp.abs(final_state_transformed) ** 2
            prob_total = jnp.sum(probs)
            mass_in = jnp.sum(probs[:cutoff])
            leakage = 1.0 - (mass_in / jnp.maximum(prob_total, 1e-12))

            state_trunc = final_state_transformed[:cutoff]
            norm_trunc = jnp.linalg.norm(state_trunc)
            state_eval = jax.lax.cond(
                norm_trunc > 1e-9, lambda x: x / norm_trunc, lambda x: x, state_trunc
            )
            leakage_penalty = leakage * 2.0
        else:
            state_eval = final_state_transformed

        # 4. Expectation Value
        op_psi = operator @ state_eval
        raw_exp_val = jnp.real(jnp.vdot(state_eval, op_psi))

        exp_val = jnp.where(joint_prob > 1e-40, raw_exp_val, jnp.inf)
        exp_val += leakage_penalty

        # 5. Fitness & Descriptors
        prob_clipped = jnp.maximum(joint_prob, 1e-45)
        log_prob = -jnp.log10(prob_clipped)
        total_photons = total_sum_pnr

        fitness = jnp.array([-exp_val, -log_prob, -active_modes, -total_photons])
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

    pmapped_fn = jax.pmap(
        _score_batch_shard,
        in_axes=(0, None, None, None, None, None, None),
        static_broadcasted_argnums=(1, 3, 4, 5, 6),
    )

    if genotype_config is not None and isinstance(genotype_config, dict):
        config_hashable = tuple(sorted(genotype_config.items()))
    else:
        config_hashable = genotype_config

    fitnesses_sharded, descriptors_sharded = pmapped_fn(
        g_sharded,
        cutoff,
        operator,
        genotype_name,
        config_hashable,
        correction_cutoff,
        pnr_max,
    )

    fitnesses = fitnesses_sharded.reshape((-1, 4))
    descriptors = descriptors_sharded.reshape((-1, 3))

    if padding > 0:
        fitnesses = fitnesses[:batch_size]
        descriptors = descriptors[:batch_size]

    return fitnesses, descriptors
