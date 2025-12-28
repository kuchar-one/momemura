import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


def jax_compose_pair(
    stateA: jnp.ndarray,
    stateB: jnp.ndarray,
    U: jnp.ndarray,
    pA: float,
    pB: float,
    homodyne_x: float,  # Pass 0.0 if None
    homodyne_window: float,  # Pass 0.0 if None
    homodyne_resolution: float,  # Pass 0.0 if None
    phi_vec: jnp.ndarray,  # Precomputed phi vector or None (pass zeros if None)
    V_matrix: jnp.ndarray,  # Precomputed V matrix or None
    dx_weights: jnp.ndarray,  # Precomputed weights or None
    cutoff: int,  # Implicit in shapes, but maybe needed for reshape
    homodyne_window_is_none: bool,
    homodyne_x_is_none: bool,
    homodyne_resolution_is_none: bool,
    theta: float = 0.0,  # Added for optimization
    phi: float = 0.0,  # Added for optimization
) -> Tuple[jnp.ndarray, float, float]:
    is_vec_A = stateA.ndim == 1
    is_vec_B = stateB.ndim == 1

    # --- Pure-state path ---
    # We can use python control flow for is_vec checks because shapes are static?
    # Yes, ndim is static.

    if is_vec_A and is_vec_B and homodyne_window_is_none:
        psi_in = jnp.kron(stateA, stateB)

        # Optimization: Use block-diagonal application if U is not provided (or we can reconstruct it)
        # We assume if theta/phi are passed, we can use them.
        # But U is always passed by jax_superblock.
        # We can check if U is None? No, JIT.
        # We can check if U is a dummy?
        # Let's just use jax_apply_bs_vec if we are in this path.
        # We trust theta/phi are correct.

        psi_out = jax_apply_bs_vec(psi_in, theta, phi, cutoff)

        if homodyne_x_is_none:
            # Partial trace
            c = cutoff
            psi2 = psi_out.reshape((c, c))
            rho_red = jnp.einsum("ij,kj->ik", psi2, psi2.conj())

            tr = jnp.real(jnp.trace(rho_red))
            rho_red = jax.lax.cond(
                tr != 0, lambda _: rho_red / tr, lambda _: rho_red, None
            )
            joint = pA * pB
            return rho_red, 1.0, joint

        # Homodyne point
        # phi_vec must be provided
        psi2d = psi_out.reshape((cutoff, cutoff))
        v = psi2d @ phi_vec
        p_x_density = jnp.real(jnp.vdot(v, v))

        if homodyne_resolution_is_none:
            p_measure = p_x_density
        else:
            p_measure = p_x_density * homodyne_resolution
            p_measure = jnp.minimum(p_measure, 1.0)

        vec_cond = jax.lax.cond(
            p_x_density > 0,
            lambda _: v / jnp.sqrt(p_x_density),
            lambda _: jnp.zeros_like(v),
            None,
        )

        joint = pA * pB * p_measure
        return vec_cond, p_measure, joint

    # --- Mixed-state path ---
    def ensure_dm(s):
        if s.ndim == 2:
            return s
        else:
            return jnp.outer(s, s.conj())

    rhoA = ensure_dm(stateA)
    rhoB = ensure_dm(stateB)

    rho_in = jnp.kron(rhoA, rhoB)
    # U @ rho @ U.dag
    # U is (N, N). rho is (N, N).
    # But here N is c^2.
    rho_out = U @ rho_in @ U.conj().T

    if homodyne_x_is_none and homodyne_window_is_none:
        c = cutoff
        rho_t = rho_out.reshape((c, c, c, c))
        # Partial trace over mode 2 (indices 1 and 3)
        rho_red = jnp.einsum("ijkj->ik", rho_t)

        tr = jnp.real(jnp.trace(rho_red))
        rho_red = jax.lax.cond(tr != 0, lambda _: rho_red / tr, lambda _: rho_red, None)
        joint = pA * pB
        return rho_red, 1.0, joint

    # Homodyne
    if homodyne_window_is_none:
        # Point
        # phi_vec must be provided
        c = cutoff
        rho_reshaped = rho_out.reshape((c, c, c, c))
        new_rho = jnp.einsum("akbl,k,l->ab", rho_reshaped, phi_vec.conj(), phi_vec)

        p_x_density = jnp.real(jnp.trace(new_rho))

        if homodyne_resolution_is_none:
            p_measure = p_x_density
        else:
            p_measure = p_x_density * homodyne_resolution
            p_measure = jnp.minimum(p_measure, 1.0)

        rho_cond = jax.lax.cond(
            p_x_density > 0,
            lambda _: new_rho / p_x_density,
            lambda _: jnp.zeros_like(new_rho),
            None,
        )

        joint = pA * pB * p_measure
        return rho_cond, p_measure, joint

    # Window
    # V_matrix and dx_weights must be provided
    c = cutoff
    rho_reshaped = rho_out.reshape((c, c, c, c))

    rho_cond_stack = jnp.einsum(
        "aubv,vi,ui->abi", rho_reshaped, V_matrix, V_matrix.conj()
    )

    p_xs = jnp.real(jnp.einsum("aai->i", rho_cond_stack))

    Pwin = jnp.sum(p_xs * dx_weights)
    Pwin = jnp.minimum(Pwin, 1.0)

    rho_cond_integrated = jnp.sum(rho_cond_stack * dx_weights[None, None, :], axis=2)

    rho_cond = jax.lax.cond(
        Pwin > 0,
        lambda _: rho_cond_integrated / Pwin,
        lambda _: jnp.zeros((c, c), dtype=jnp.complex_),
        None,
    )

    joint = pA * pB * Pwin
    return rho_cond, Pwin, joint


def jax_hermite_phi_matrix(
    xs: jnp.ndarray, cutoff: int, hbar: float = 2.0
) -> jnp.ndarray:
    """
    Computes the Hermite basis functions phi_n(x) for n in [0, cutoff-1] and x in xs.
    Returns matrix of shape (cutoff, len(xs)).

    phi_n(x) = (1/sqrt(2^n n! sqrt(pi hbar))) * H_n(x/sqrt(hbar)) * exp(-x^2/(2 hbar))
    """
    xs = jnp.asarray(xs)

    # Precompute constants
    norm_0 = (jnp.pi * hbar) ** (-0.25)

    # phi_0
    phi_0 = norm_0 * jnp.exp(-(xs**2) / (2 * hbar))

    # Initialize matrix list
    phis = [phi_0]

    if cutoff > 1:
        # phi_1
        phi_1 = jnp.sqrt(2 / hbar) * xs * phi_0
        phis.append(phi_1)

        for n in range(1, cutoff - 1):
            # Recurrence:
            # phi_{n+1} = x * sqrt(2/((n+1)hbar)) * phi_n - sqrt(n/(n+1)) * phi_{n-1}

            p_n = phis[-1]
            p_nm1 = phis[-2]

            c1 = jnp.sqrt(2 / ((n + 1) * hbar))
            c2 = jnp.sqrt(n / (n + 1))

            p_np1 = c1 * xs * p_n - c2 * p_nm1
            phis.append(p_np1)

    return jnp.stack(phis)


@partial(jax.jit, static_argnames=("cutoff",))
def jax_u_bs(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    """
    Constructs BS unitary.
    Dispatches to efficient implementation based on cutoff.
    """
    if cutoff < 15:
        return _jax_u_bs_expm(theta, phi, cutoff)
    else:
        return _jax_u_bs_decomposed(theta, phi, cutoff)


def _jax_u_bs_expm(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    """Standard expm implementation for small cutoffs."""
    n = jnp.arange(1, cutoff)
    sqrt_n = jnp.sqrt(n)
    a_op = jnp.diag(sqrt_n, 1)
    identity = jnp.eye(cutoff)
    a_big = jnp.kron(a_op, identity)
    b_big = jnp.kron(identity, a_op)
    bdag_big = b_big.conj().T
    term = jnp.exp(-1j * phi) * a_big @ bdag_big
    G = theta * (term.conj().T - term)
    return jax.scipy.linalg.expm(G)


def _jax_u_bs_decomposed(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    """
    Decomposed implementation: U = D(-phi) @ exp(-2i theta Jy) @ D(phi).
    Uses real matrix exponentiation on block-diagonal Jy.
    Efficient for large cutoffs.
    """
    import numpy as np

    num_blocks = 2 * cutoff - 1

    # Static shapes for JIT
    Ns_np = np.arange(num_blocks)
    n_min_np = np.maximum(0, Ns_np - cutoff + 1)
    n_max_np = np.minimum(Ns_np, cutoff - 1)

    # 1. Construct Block-Diagonal Real Generator (-2i theta Jy)
    # We construct the blocks of -theta * (a^dag b - a b^dag)

    # Use JAX arrays for dynamic values
    Ns_jax = jnp.arange(num_blocks)
    n_min_jax = jnp.maximum(0, Ns_jax - cutoff + 1)
    n_max_jax = jnp.minimum(Ns_jax, cutoff - 1)

    ks = jnp.arange(cutoff)

    Ns_grid = Ns_jax[:, None]
    ks_grid = ks[None, :]
    n_min_grid = n_min_jax[:, None]
    n_max_grid = n_max_jax[:, None]

    n1 = n_min_grid + ks_grid
    valid_mask = (ks_grid >= 1) & (n1 <= n_max_grid)

    val = jnp.sqrt(jnp.maximum(0, n1 * (Ns_grid - n1 + 1)))

    # Generator elements
    term = -theta * val * valid_mask

    # Fill blocks (Real)
    blocks = jnp.zeros((num_blocks, cutoff, cutoff), dtype=term.dtype)
    for k in range(1, cutoff):
        blocks = blocks.at[:, k - 1, k].set(term[:, k])
        blocks = blocks.at[:, k, k - 1].set(-term[:, k])

    # 2. Compute Real Matrix Exponential
    Ry_blocks = jax.scipy.linalg.expm(blocks)

    # 3. Apply Diagonal Phases
    # m = (n1 - n2)/2 = n1 - N/2
    m_vals = n1 - Ns_grid / 2.0
    phases = jnp.exp(-1j * phi * m_vals)

    # U_block = phases.conj() * Ry_block * phases
    U_blocks_complex = phases.conj()[:, :, None] * Ry_blocks * phases[:, None, :]

    # 4. Scatter into full matrix
    # Use complex dtype matching phases
    U = jnp.zeros((cutoff**2, cutoff**2), dtype=phases.dtype)

    all_ii = []
    all_jj = []
    all_vals = []

    for N in range(num_blocks):
        nm = int(n_min_np[N])
        nM = int(n_max_np[N])
        dim = nM - nm + 1

        n1s = jnp.arange(nm, nM + 1)
        n2s = N - n1s
        indices = n1s * cutoff + n2s

        block = U_blocks_complex[N, :dim, :dim]

        ii, jj = jnp.meshgrid(indices, indices, indexing="ij")

        all_ii.append(ii.flatten())
        all_jj.append(jj.flatten())
        all_vals.append(block.flatten())

    total_ii = jnp.concatenate(all_ii)
    total_jj = jnp.concatenate(all_jj)
    total_vals = jnp.concatenate(all_vals)

    U = U.at[total_ii, total_jj].set(total_vals)

    return U


def jax_apply_bs_vec(
    vec: jnp.ndarray, theta: float, phi: float, cutoff: int
) -> jnp.ndarray:
    """
    Applies BS unitary to a vector state |psi> without constructing the full matrix.
    U = D(-phi) @ exp(-2i theta Jy) @ D(phi)

    Args:
        vec: (cutoff^2,) flattened state vector.
        theta, phi: BS parameters.
        cutoff: Fock cutoff.

    Returns:
        (cutoff^2,) transformed vector.
    """
    # 1. Apply D(phi) (Diagonal phases)
    # Basis indices |n1, n2>
    # m = (n1 - n2)/2
    # n1 = i // cutoff, n2 = i % cutoff
    indices = jnp.arange(cutoff * cutoff)
    n1 = indices // cutoff
    n2 = indices % cutoff
    m = (n1 - n2) / 2.0

    # D(phi) = exp(-i phi m)
    phases = jnp.exp(-1j * phi * m)

    # U = D(-phi) @ Ry @ D(phi)
    # D(phi) = diag(phases)
    # D(-phi) = diag(phases.conj())
    # U = diag(phases) @ Ry @ diag(phases.conj())
    # Wait, _jax_u_bs_decomposed says:
    # U_block = phases * Ry_block * phases.conj()
    # This implies U = diag(phases) @ Ry @ diag(phases.conj())
    # So (U v) = diag(phases) @ (Ry @ (diag(phases.conj()) @ v))

    # 1. Apply D(phi)
    vec = vec * phases

    # 2. Apply exp(-2i theta Jy) (Block Diagonal)
    # Jy couples |n, N-n> within subspaces of constant N = n1 + n2
    # We iterate over blocks N = 0 .. 2*cutoff - 2

    num_blocks = 2 * cutoff - 1

    # Precompute block parameters (same as _jax_u_bs_decomposed)
    Ns = jnp.arange(num_blocks)
    n_min = jnp.maximum(0, Ns - cutoff + 1)
    n_max = jnp.minimum(Ns, cutoff - 1)

    # We need to construct the blocks of the generator G = -2i theta Jy
    # Actually we want exp(G). Since G is real anti-symmetric (if we factor out i),
    # exp(-2i theta Jy) is a real rotation matrix (Ry) if we define it carefully.
    # _jax_u_bs_decomposed computes Ry_blocks = expm(blocks) where blocks is real anti-symmetric.

    # Let's reuse the block construction logic
    ks = jnp.arange(cutoff)
    Ns_grid = Ns[:, None]
    ks_grid = ks[None, :]
    n_min_grid = n_min[:, None]
    n_max_grid = n_max[:, None]

    n1_grid = n_min_grid + ks_grid
    valid_mask = (ks_grid >= 1) & (n1_grid <= n_max_grid)

    val = jnp.sqrt(jnp.maximum(0, n1_grid * (Ns_grid - n1_grid + 1)))
    term = -theta * val * valid_mask

    # Construct blocks
    blocks = jnp.zeros((num_blocks, cutoff, cutoff), dtype=term.dtype)

    # Vectorized block filling
    # We want blocks[N, k-1, k] = term[N, k]
    #         blocks[N, k, k-1] = -term[N, k]
    # for k in 1..cutoff-1

    # We can use .at with advanced indexing
    # Indices for k=1..cutoff-1
    # k_indices = jnp.arange(1, cutoff)

    # We need to broadcast over N
    # blocks[:, k-1, k]
    # We can loop over k (cutoff is small, 25) or vectorize fully.
    # Looping over k is fine and readable.
    for k in range(1, cutoff):
        blocks = blocks.at[:, k - 1, k].set(term[:, k])
        blocks = blocks.at[:, k, k - 1].set(-term[:, k])

    # Expm
    Ry_blocks = jax.scipy.linalg.expm(blocks)

    # Apply blocks to vector
    # We need to extract the relevant slice of 'vec' for each block N
    # and multiply by Ry_blocks[N]

    # This is tricky to vectorize efficiently because slices have different sizes.
    # However, we can use a masked approach or scan.
    # Since max block size is 'cutoff' (25), and there are ~50 blocks.
    # We can pad everything to 'cutoff' size.

    # Gather input vector segments
    # For each N, the indices are n1 * cutoff + (N - n1) for n1 in [n_min, n_max]
    # We can construct a gather map.

    # Grid of indices (num_blocks, cutoff)
    # n1 varies from n_min to n_max. We pad with 0 (or dummy)
    n1_indices = n_min_grid + ks_grid  # (num_blocks, cutoff)
    n2_indices = Ns_grid - n1_indices
    flat_indices = n1_indices * cutoff + n2_indices

    # Mask for valid indices within each block
    # dim = n_max - n_min + 1
    dims = n_max - n_min + 1
    mask = ks_grid < dims[:, None]

    # Gather
    # We use a safe gather (clamp indices) and then mask
    safe_indices = jnp.minimum(flat_indices, cutoff * cutoff - 1)
    vec_gathered = vec[safe_indices]  # (num_blocks, cutoff)
    vec_gathered = jnp.where(mask, vec_gathered, 0.0)

    # Matrix Multiply
    # Ry_blocks is (num_blocks, cutoff, cutoff)
    # vec_gathered is (num_blocks, cutoff)
    # We want (num_blocks, cutoff) output
    vec_transformed_blocks = jnp.einsum("nij,nj->ni", Ry_blocks, vec_gathered)

    # Scatter back
    # We need to add these back to a zero vector?
    # Or just set them? The blocks partition the space, so we can just set.
    # But scatter requires unique indices. flat_indices has duplicates (the padded parts).
    # We must mask the scatter.

    # Create output vector
    vec_out = jnp.zeros_like(vec)

    # We can flatten the batch and scatter
    flat_indices_all = flat_indices.flatten()
    vals_all = vec_transformed_blocks.flatten()
    mask_all = mask.flatten()

    # Only scatter valid
    # jax.ops.index_update or .at
    # We can use where to filter indices? No, indices must be static or bounded.
    # But we can scatter everything, with invalid indices pointing to a dummy location?
    # Or just add 0s to dummy locations.
    # Let's use a dummy index -1 (not supported) or 0?
    # If we scatter to 0, we corrupt index 0.
    # Better: scatter add. Initialize vec_out to 0.
    # Invalid vals are 0.0 (masked above).
    # Invalid indices: point them to 0.
    # Since we add 0.0 to index 0, it's safe (no change).
    # Wait, index 0 is valid for N=0 block.
    # So we are adding 0.0 to index 0 from invalid parts of other blocks?
    # Yes. 0 + 0 = 0. Safe.

    safe_indices_all = jnp.where(mask_all, flat_indices_all, 0)
    vals_all_masked = jnp.where(mask_all, vals_all, 0.0)

    vec_out = vec_out.at[safe_indices_all].add(vals_all_masked)

    # 3. Apply D(-phi)
    vec_out = vec_out * phases.conj()

    return vec_out


def jax_superblock(
    leaf_states: jnp.ndarray,  # (8, cutoff) or (8, cutoff, cutoff)
    leaf_probs: jnp.ndarray,  # (8,)
    leaf_active: jnp.ndarray,  # (8,) boolean
    leaf_pnr: jnp.ndarray,  # (8,) max PNR for each leaf
    leaf_total_pnr: jnp.ndarray,  # (8,) sum PNR for each leaf (NEW)
    leaf_modes: jnp.ndarray,  # (8,) mode count for each leaf (usually 1 or 2)
    mix_params: jnp.ndarray,  # (7, 3) theta, phi, varphi
    homodyne_x: jnp.ndarray,  # (7,) or scalar
    homodyne_window: float,
    homodyne_resolution: float,
    phi_vec: jnp.ndarray,  # (7, cutoff) or (cutoff,)
    V_matrix: jnp.ndarray,
    dx_weights: jnp.ndarray,
    cutoff: int,
    homodyne_window_is_none: bool,
    homodyne_x_is_none: bool,
    homodyne_resolution_is_none: bool,
) -> Tuple[jnp.ndarray, float, float, float, float, float, float]:
    """
    Implements a fixed-depth binary tree of blocks (Depth 3 = 8 leaves).
    Returns (final_state, final_prob, joint_prob, active_modes_count, max_pnr, total_sum_pnr).
    """

    # Broadcast homodyne args if scalars/single vectors
    # This supports legacy (scalar hom_x) and new (vector hom_x)
    n_nodes = mix_params.shape[0]  # Should be 7

    # Ensure hom_x is array (N,)
    hom_xs = jnp.atleast_1d(homodyne_x)
    if hom_xs.ndim == 1 and hom_xs.shape[0] == 1:
        hom_xs = jnp.broadcast_to(hom_xs, (n_nodes,))
    elif hom_xs.ndim == 0:
        hom_xs = jnp.broadcast_to(hom_xs, (n_nodes,))

    # Ensure phi_vec is array (N, C)
    phi_vecs = phi_vec
    if phi_vecs.ndim == 1:
        phi_vecs = jnp.broadcast_to(phi_vecs, (n_nodes, cutoff))

    # Helper to mix or pass
    def mix_node(
        stateA,
        stateB,
        probA,
        probB,
        activeA,
        activeB,
        pnrA,
        pnrB,
        sumPnrA,
        sumPnrB,
        modesA,
        modesB,
        params,
        hx,  # Per-node homodyne x
        phi_v,  # Per-node phi vector
        # source removed
    ):
        # Implicit Source Logic:
        # Both Active -> Mix (0)
        # Only A Active -> Pass Left (1)
        # Only B Active -> Pass Right (2)
        # Neither Active -> Pass Left (1) [Inactive]

        def do_mix(_):
            theta, phi, varphi = params

            # Optimization: Skip U if pure vectors & point homodyne
            are_leaves_vectors = leaf_states.ndim == 2
            can_skip_U = (
                are_leaves_vectors
                and homodyne_window_is_none
                and (not homodyne_x_is_none)
            )

            def construct_U():
                return jax_u_bs(theta, phi, cutoff)

            def dummy_U():
                # Dynamically match dtype of construct_U (which depends on theta)
                target_dtype = (theta + 0j).dtype
                return jnp.zeros((cutoff * cutoff, cutoff * cutoff), dtype=target_dtype)

            U = jax.lax.cond(can_skip_U, dummy_U, construct_U)

            res_state, p_mix, joint = jax_compose_pair(
                stateA,
                stateB,
                U,
                probA,
                probB,
                hx,  # Use local hx
                homodyne_window,
                homodyne_resolution,
                phi_v,  # Use local phi_v
                V_matrix,
                dx_weights,
                cutoff,
                homodyne_window_is_none,
                homodyne_x_is_none,
                homodyne_resolution_is_none,
                theta=theta,
                phi=phi,
            )

            # Descriptors logic
            new_modes = modesA + modesB
            new_pnr = jnp.maximum(pnrA, pnrB)
            new_total_pnr = sumPnrA + sumPnrB
            new_active = activeA | activeB

            return (
                res_state,
                p_mix,
                joint,
                new_active,
                new_pnr,
                new_total_pnr,
                new_modes,
            )

        def do_left(_):
            return stateA, 1.0, probA, activeA, pnrA, sumPnrA, modesA

        def do_right(_):
            return stateB, 1.0, probB, activeB, pnrB, sumPnrB, modesB

        # Determine effective source
        # Default 0 (Mix)
        effective_source = jnp.zeros((), dtype=jnp.int32)

        # Override based on active flags
        # If A active and B inactive -> 1
        effective_source = jax.lax.select(
            activeA & (~activeB), jnp.array(1, dtype=jnp.int32), effective_source
        )
        # If B active and A inactive -> 2
        effective_source = jax.lax.select(
            activeB & (~activeA), jnp.array(2, dtype=jnp.int32), effective_source
        )
        # If Neither active -> 1
        effective_source = jax.lax.select(
            (~activeA) & (~activeB), jnp.array(1, dtype=jnp.int32), effective_source
        )

        return jax.lax.switch(effective_source, [do_mix, do_left, do_right], None)

    # Tree Construction
    current_states = leaf_states
    current_probs = leaf_probs
    current_actives = leaf_active
    current_pnrs = leaf_pnr
    current_sum_pnrs = leaf_total_pnr
    current_modes = leaf_modes

    mix_idx = 0

    # Handle V_matrix broadcasting if needed
    # If V_matrix is (cutoff, n_points), broadcast to (n_mix, cutoff, n_points)
    # n_mix in this topology is 7 (4+2+1).
    # We can infer from mix_params or hom_xs.
    n_mix_total = 7  # Fixed for depth 3.
    # But hom_xs might be (7,).

    # Layer 0 (8 -> 4)
    n_pairs = 4
    params_0 = mix_params[mix_idx : mix_idx + n_pairs]
    hx_0 = hom_xs[mix_idx : mix_idx + n_pairs]
    phi_0 = phi_vecs[mix_idx : mix_idx + n_pairs]
    mix_idx += n_pairs

    sA = leaf_states[0::2]
    sB = leaf_states[1::2]
    pA = leaf_probs[0::2]
    pB = leaf_probs[1::2]
    actA = leaf_active[0::2]
    actB = leaf_active[1::2]
    pnrA = leaf_pnr[0::2]
    pnrB = leaf_pnr[1::2]
    sumA = leaf_total_pnr[0::2]
    sumB = leaf_total_pnr[1::2]
    mA = leaf_modes[0::2]
    mB = leaf_modes[1::2]

    (
        current_states,
        _,
        current_probs,
        current_actives,
        current_pnrs,
        current_sum_pnrs,
        current_modes,
    ) = jax.vmap(mix_node)(
        sA,
        sB,
        pA,
        pB,
        actA,
        actB,
        pnrA,
        pnrB,
        sumA,
        sumB,
        mA,
        mB,
        params_0,
        hx_0,
        phi_0,
    )

    # Layer 1 (4 -> 2)
    n_pairs = 2
    params_1 = mix_params[mix_idx : mix_idx + n_pairs]
    hx_1 = hom_xs[mix_idx : mix_idx + n_pairs]
    phi_1 = phi_vecs[mix_idx : mix_idx + n_pairs]
    mix_idx += n_pairs

    sA = current_states[0::2]
    sB = current_states[1::2]
    pA = current_probs[0::2]
    pB = current_probs[1::2]
    actA = current_actives[0::2]
    actB = current_actives[1::2]
    pnrA = current_pnrs[0::2]
    pnrB = current_pnrs[1::2]
    sumA = current_sum_pnrs[0::2]
    sumB = current_sum_pnrs[1::2]
    mA = current_modes[0::2]
    mB = current_modes[1::2]

    (
        current_states,
        _,
        current_probs,
        current_actives,
        current_pnrs,
        current_sum_pnrs,
        current_modes,
    ) = jax.vmap(mix_node)(
        sA,
        sB,
        pA,
        pB,
        actA,
        actB,
        pnrA,
        pnrB,
        sumA,
        sumB,
        mA,
        mB,
        params_1,
        hx_1,
        phi_1,
    )

    # Layer 2 (2 -> 1)
    n_pairs = 1
    params_2 = mix_params[mix_idx : mix_idx + n_pairs]
    hx_2 = hom_xs[mix_idx : mix_idx + n_pairs]
    phi_2 = phi_vecs[mix_idx : mix_idx + n_pairs]
    mix_idx += n_pairs

    sA = current_states[0::2]
    sB = current_states[1::2]
    pA = current_probs[0::2]
    pB = current_probs[1::2]
    actA = current_actives[0::2]
    actB = current_actives[1::2]
    pnrA = current_pnrs[0::2]
    pnrB = current_pnrs[1::2]
    sumA = current_sum_pnrs[0::2]
    sumB = current_sum_pnrs[1::2]
    mA = current_modes[0::2]
    mB = current_modes[1::2]

    (
        final_state,
        _,
        joint_prob,
        is_active,
        max_pnr,
        total_sum_pnr,
        active_modes,
    ) = jax.vmap(mix_node)(
        sA,
        sB,
        pA,
        pB,
        actA,
        actB,
        pnrA,
        pnrB,
        sumA,
        sumB,
        mA,
        mB,
        params_2,
        hx_2,
        phi_2,
    )

    # Final result
    # Final result
    root_state = final_state[0]
    root_prob = joint_prob[0]
    root_active = is_active[0]
    root_pnr = max_pnr[0]
    root_sum_pnr = total_sum_pnr[0]
    root_modes = active_modes[0]

    return root_state, 1.0, root_prob, root_active, root_pnr, root_sum_pnr, root_modes
