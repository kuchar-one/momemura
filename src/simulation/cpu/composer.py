# composer.py
"""
Improved Strawberry Fields composer with aggressive caching and a SuperblockTopology
builder that allows flexible ways of combining identical blocks (full binary tree,
mixed superblock+block pairings, custom patterns).

This replacement:
 - standardizes quadrature conventions to hbar = 2.0 everywhere
 - implements both pure-state fast path and exact mixed-state propagation
 - constructs numerically-stable beamsplitter unitaries via matrix exponentials
 - aggressively caches heavy objects (U_bs, rho_full, homodyne results) via CacheManager
 - supports homodyne point conditioning (pure-state-preserving) and homodyne-window (produces mixed states)
"""

from typing import Optional, Tuple, List, Any, Union
import numpy as np
import math
import threading
import os
import warnings

from src.utils.cache_manager import CacheManager
from src.simulation.cpu.ops import (
    HBAR,
    build_beamsplitter_unitary,
    kron_state_vector,
    contract_rho_with_phi,
    get_phi_matrix_cached,
)
from numpy.polynomial.legendre import leggauss

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


# Silence potential strawberryfields deprecation noise in importers if present
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# Project-level cache directory
# Project-level cache directory
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
global_composer_cache = CacheManager(
    cache_dir=CACHE_DIR, size_limit_bytes=1024 * 1024 * 1024
)

# Global in-memory cache for beam splitter unitaries to persist across Composer instances
# Key: (cutoff, theta, phi)
# Value: np.ndarray
_GLOBAL_BS_CACHE = {}
_GLOBAL_BS_CACHE_LIMIT = 1000


# -----------------------------------------------------------
# Composer class (public)
# -----------------------------------------------------------
class Composer:
    """
    Composer that composes two single-mode states via a beamsplitter and optionally conditions
    one output mode on homodyne outcomes.

    Key features:
    - cutoff: single-mode Fock truncation
    - hbar standardized to HBAR (2.0)
    - caches beamsplitter unitaries and expensive results via CacheManager
    """

    def __init__(
        self,
        cutoff: int,
        cache: Optional[CacheManager] = None,
        cache_enabled: bool = True,
        backend: str = "thewalrus",
    ):
        self.cutoff = int(cutoff)
        self.cache_enabled = bool(cache_enabled)
        self.cache = cache if cache is not None else global_composer_cache
        self.backend = backend.lower()
        if self.backend == "jax" and not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not available.")

        # small in-memory caches to avoid hitting diskcache often
        self._U_cache_local = {}
        self._engine_lock = threading.Lock()

        # Cache for leggauss nodes/weights
        self._leggauss_cache = {}

    # ------------
    # Helpers
    # ------------
    def _u_bs(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """
        Return (and cache) the beamsplitter unitary for current cutoff/theta/phi.
        Uses a global in-memory cache to persist across Composer instances.
        """
        key = (self.cutoff, theta, phi)

        # 1. Check global in-memory cache
        if key in _GLOBAL_BS_CACHE:
            return _GLOBAL_BS_CACHE[key]

        # 2. Check local cache (legacy, but fast)
        if key in self._U_cache_local:
            return self._U_cache_local[key]

        # 3. Build (this hits disk cache internally if configured, but we want to avoid that overhead too)
        # build_beamsplitter_unitary uses the passed cache manager (self.cache)
        U = build_beamsplitter_unitary(self.cutoff, theta, phi, cache=self.cache)

        # 4. Update global cache
        if len(_GLOBAL_BS_CACHE) < _GLOBAL_BS_CACHE_LIMIT:
            _GLOBAL_BS_CACHE[key] = U
        else:
            # Simple eviction
            _GLOBAL_BS_CACHE.pop(next(iter(_GLOBAL_BS_CACHE)))
            _GLOBAL_BS_CACHE[key] = U

        return U

    # ------------
    # Compose pair (flexible)
    # ------------
    def compose_pair(
        self,
        stateA: Union[np.ndarray, np.ndarray],
        stateB: Union[np.ndarray, np.ndarray],
        pA: float = 1.0,
        pB: float = 1.0,
        homodyne_x: Optional[float] = None,
        homodyne_window: Optional[float] = None,
        homodyne_resolution: Optional[float] = None,
        theta: float = math.pi / 4.0,
        phi: float = 0.0,
        n_hom_points: int = 201,
    ) -> Tuple[Union[np.ndarray, np.ndarray], float, float]:
        """
        Compose two single-mode states (each may be either:
           - a 1D complex vector (pure Fock amplitudes length cutoff), or
           - a 2D complex density matrix (cutoff x cutoff)
        Returns:
           - state_out: if pure path -> 1D state vector (length cutoff), else 2D density matrix
           - p_hom_or_Pwin:
                * If homodyne_resolution is None and homodyne_window is None: returns probability DENSITY p(x).
                * If homodyne_resolution is set OR homodyne_window is used: returns PROBABILITY (approximate or integrated).
           - joint_prob: product pA * pB * (p_hom_or_Pwin)
        Behavior notes:
           - If both inputs are pure vectors and homodyne_x is provided (point), this will take the pure-state
             fast path (use U_bs @ (f1 ⊗ f2), compute conditional vector via phi).
           - If homodyne_window is provided or any input is a density matrix, the method uses the mixed-state
             route: rho_out = U_bs @ (rhoA ⊗ rhoB) @ U_bs^\dag and then performs homodyne contraction/integration.
        """
        # Validate inputs
        is_vec_A = stateA.ndim == 1
        is_vec_B = stateB.ndim == 1
        is_dm_A = stateA.ndim == 2
        is_dm_B = stateB.ndim == 2
        if not (is_vec_A or is_dm_A) or not (is_vec_B or is_dm_B):
            raise ValueError(
                "stateA/stateB must be either 1D vector or 2D density matrix."
            )

        if self.backend == "jax":
            # JAX Backend Path
            return self._compose_pair_jax(
                stateA,
                stateB,
                pA,
                pB,
                homodyne_x,
                homodyne_window,
                homodyne_resolution,
                theta,
                phi,
                n_hom_points,
            )

        # choose beamsplitter unitary
        U = self._u_bs(theta, phi)

        # --- Pure-state path: both vectors, homodyne point or no homodyne (but note: trace -> mixed) ---
        if is_vec_A and is_vec_B and homodyne_window is None:
            # input pure global vector
            psi_in = kron_state_vector(stateA, stateB)  # shape (c^2,)
            psi_out = U @ psi_in  # full two-mode pure state vector

            if homodyne_x is None:
                # No measurement: partial trace over mode2 -> reduced density (mixed in general)
                c = self.cutoff
                psi2 = psi_out.reshape((c, c))  # psi2[n1, n2]
                # compute reduced rho for mode1: sum_n2 psi[:,n2] psi[:,n2]^*
                rho_red = np.tensordot(psi2, psi2.conj(), axes=([1], [1]))  # (n0,m0)
                rho_red = np.asarray(rho_red, dtype=complex)
                # normalize for numerical safety
                tr = np.real(np.trace(rho_red))
                if tr != 0:
                    rho_red = rho_red / tr
                joint = float(pA * pB)
                return rho_red, 1.0, joint

            # homodyne point requested -> pure-state conditional vector (fast)
            # phi_vec = quadrature_vector(self.cutoff, float(homodyne_x), hbar=HBAR)
            # Use cached vectorized version even for single point
            phi_vec = get_phi_matrix_cached(
                self.cutoff, np.array([float(homodyne_x)]), hbar=HBAR
            )[:, 0]
            psi2d = psi_out.reshape((self.cutoff, self.cutoff))  # psi[n1, n2]
            # unnormalized conditional vector for mode1:
            v = psi2d @ phi_vec  # v[n1] = sum_{n2} psi[n1,n2] phi[n2]
            p_x_density = float(np.real(np.vdot(v, v)))

            # convert density -> probability if resolution provided
            if homodyne_resolution is None:
                p_measure = p_x_density  # keep density (analytic mode)
            else:
                p_measure = float(
                    p_x_density * float(homodyne_resolution)
                )  # approximate probability

            if p_x_density > 0:
                vec_cond = v / math.sqrt(p_x_density)
            else:
                vec_cond = np.zeros_like(v)
            joint = float(pA * pB * p_measure)
            return vec_cond, p_measure, joint

        # --- Mixed-state / window path: construct densities and propagate exactly ---
        # Convert inputs to densities if needed
        def ensure_dm(s):
            if s.ndim == 2:
                return s
            else:
                # vector -> outer product
                v = s
                return np.outer(v, v.conj())

        rhoA = ensure_dm(stateA)
        rhoB = ensure_dm(stateB)

        # form full input density
        rho_in = np.kron(rhoA, rhoB)  # shape (c^2, c^2)

        # propagate via U
        rho_out = U @ rho_in @ U.conj().T

        # If no homodyne, trace out mode2 and return reduced rho
        if homodyne_x is None and homodyne_window is None:
            c = self.cutoff
            rho_t = rho_out.reshape((c, c, c, c))
            rho_red = np.zeros((c, c), dtype=complex)
            for n2 in range(c):
                rho_red += rho_t[:, n2, :, n2]
            # normalize (trace 1)
            tr = float(np.real_if_close(np.trace(rho_red)))
            if tr != 0:
                rho_red = rho_red / tr
            joint = float(pA * pB)
            return rho_red, 1.0, joint

        # If we reach here, we must perform homodyne point or window on mode2 (mixed-state approach)
        if homodyne_window is None:
            # point homodyne: contract
            # phi_vec = quadrature_vector(self.cutoff, float(homodyne_x), hbar=HBAR)
            phi_vec = get_phi_matrix_cached(
                self.cutoff, np.array([float(homodyne_x)]), hbar=HBAR
            )[:, 0]
            new_rho = contract_rho_with_phi(rho_out, phi_vec)
            p_x_density = float(np.real(np.trace(new_rho)))

            # convert density -> probability if resolution provided
            if homodyne_resolution is None:
                p_measure = p_x_density
            else:
                p_measure = float(p_x_density * float(homodyne_resolution))

            if p_x_density > 0:
                rho_cond = new_rho / p_x_density
            else:
                rho_cond = np.zeros_like(new_rho)
            joint = float(pA * pB * p_measure)
            return rho_cond, p_measure, joint

        # homodyne window: integrate numerically
        c = self.cutoff

        # Vectorized integration
        # 1. Integration nodes
        if homodyne_window is not None:
            # Use Gauss-Legendre quadrature for window integration
            # n_hom_points default 201 is for trapz; for Gauss 51 is usually sufficient/better
            # We use min(n_hom_points, 61) to avoid excessive nodes if user passed 201
            n_nodes = min(n_hom_points, 61)

            # Cache leggauss results
            if n_nodes in self._leggauss_cache:
                nodes, weights = self._leggauss_cache[n_nodes]
            else:
                nodes, weights = leggauss(n_nodes)
                self._leggauss_cache[n_nodes] = (nodes, weights)

            # Map [-1, 1] to [x - w/2, x + w/2]
            half_w = homodyne_window / 2.0
            center = homodyne_x if homodyne_x is not None else 0.0
            xs = center + half_w * nodes
            dx_weights = half_w * weights  # Jacobian is half_w

        else:
            # Point homodyne: just one point
            xs = np.array([homodyne_x], dtype=float)
            dx_weights = np.array(
                [1.0], dtype=float
            )  # Not used for point measurement except maybe scaling?
            # Actually for point measurement we don't integrate, we just take the value.

        # Get quadrature matrix for all xs at once (cached)
        # V shape: (c, N_points)
        V = get_phi_matrix_cached(c, xs, hbar=HBAR)

        # 2. Contract rho_out with all phi_vecs
        # rho_out is (c^2, c^2) corresponding to (a, u, b, v) where u,v are mode B indices
        # V is (c, N)
        # We want rho_cond_i[a, b] = sum_{u, v} V[u, i]^* rho[a, u, b, v] V[v, i]

        c = self.cutoff
        rho_reshaped = rho_out.reshape((c, c, c, c))

        # Calculate conditional states for all x points
        # shape (c, c, N)
        rho_cond_stack = np.einsum("aubv,vi,ui->abi", rho_reshaped, V, V.conj())

        # Calculate p(x) for all x points
        # p(x) = Tr(rho_cond(x))
        # shape (N,)
        p_xs = np.real(np.einsum("aai->i", rho_cond_stack))

        if homodyne_window is not None:
            # 3. Integrate p(x) to get Pwin
            # Pwin = sum(p(x_i) * w_i)
            Pwin = float(np.sum(p_xs * dx_weights))

            # 4. Compute integrated conditional state
            # rho_cond = \int dx rho_cond(x)
            # rho_cond = sum(rho_cond(x_i) * w_i)
            # rho_cond_stack is (c, c, N), dx_weights is (N,)
            rho_cond_integrated = np.sum(
                rho_cond_stack * dx_weights[None, None, :], axis=2
            )

            if Pwin > 0:
                rho_cond = rho_cond_integrated / Pwin
            else:
                rho_cond = np.zeros((c, c), dtype=complex)

            joint = float(pA * pB * Pwin)
            return rho_cond, Pwin, joint

        else:
            # Point homodyne
            # p(x) is density. To get prob, multiply by resolution if provided.
            p_val = p_xs[0]
            rho_cond_unnorm = rho_cond_stack[:, :, 0]

            if homodyne_resolution is not None:
                prob = p_val * homodyne_resolution
            else:
                prob = p_val  # Density treated as prob (legacy behavior or specific use case)

            if p_val > 1e-15:
                rho_cond = rho_cond_unnorm / p_val
            else:
                rho_cond = np.zeros((c, c), dtype=complex)
                prob = 0.0

            joint = float(pA * pB * prob)
            return rho_cond, prob, joint

    # convenience wrapper that caches on input bytes + params
    def compose_pair_cached(
        self,
        stateA: Union[np.ndarray, np.ndarray],
        stateB: Union[np.ndarray, np.ndarray],
        pA: float = 1.0,
        pB: float = 1.0,
        homodyne_x: Optional[float] = None,
        homodyne_window: Optional[float] = None,
        homodyne_resolution: Optional[float] = None,
        theta: float = math.pi / 4.0,
        phi: float = 0.0,
        n_hom_points: int = 201,
    ) -> Tuple[Union[np.ndarray, np.ndarray], float, float]:
        """
        Wrap compose_pair with persistent caching keyed by:
           (bytes(stateA), bytes(stateB), cutoff, theta, phi, homodyne_x, homodyne_window, homodyne_resolution, n_hom_points)
        Where states are serialized as bytes (vectors/density mats).
        """
        # Bypass caching due to high overhead and low hit rate in continuous optimization
        return self.compose_pair(
            stateA,
            stateB,
            pA,
            pB,
            homodyne_x,
            homodyne_window,
            homodyne_resolution,
            theta,
            phi,
            n_hom_points,
        )

    def clear_caches(self):
        """Clear the persistent cache (useful between experiments)."""
        self.cache.clear()

    # -----------------------------
    # JAX Implementation
    # -----------------------------
    def _u_bs_jax(self, theta: float, phi: float) -> "jnp.ndarray":
        """
        Return JAX array for beamsplitter unitary.
        Uses the same cache mechanism but converts to JAX array.
        """
        # Get numpy unitary from cache
        U_np = self._u_bs(theta, phi)
        # Convert to JAX array (this will move to GPU if available)
        return jnp.array(U_np)

    def _compose_pair_jax(
        self,
        stateA: Union[np.ndarray, "jnp.ndarray"],
        stateB: Union[np.ndarray, "jnp.ndarray"],
        pA: float,
        pB: float,
        homodyne_x: Optional[float],
        homodyne_window: Optional[float],
        homodyne_resolution: Optional[float],
        theta: float,
        phi: float,
        n_hom_points: int,
    ) -> Tuple[Union[np.ndarray, "jnp.ndarray"], float, float]:
        """
        JAX implementation of compose_pair using JIT-compiled kernel.
        """
        # Ensure inputs are JAX arrays
        stateA = jnp.asarray(stateA)
        stateB = jnp.asarray(stateB)

        U = self._u_bs_jax(theta, phi)

        # Prepare arguments for JIT function
        hom_x_val = float(homodyne_x) if homodyne_x is not None else 0.0
        hom_win_val = float(homodyne_window) if homodyne_window is not None else 0.0
        hom_res_val = (
            float(homodyne_resolution) if homodyne_resolution is not None else 0.0
        )

        phi_vec = jnp.zeros(1)  # Dummy
        V_matrix = jnp.zeros((1, 1))  # Dummy
        dx_weights = jnp.zeros(1)  # Dummy

        if homodyne_x is not None and homodyne_window is None:
            # Point
            phi_vec_np = get_phi_matrix_cached(
                self.cutoff, np.array([float(homodyne_x)]), hbar=HBAR
            )[:, 0]
            phi_vec = jnp.array(phi_vec_np)

        if homodyne_window is not None:
            # Window
            n_nodes = min(n_hom_points, 61)
            if n_nodes in self._leggauss_cache:
                nodes, weights = self._leggauss_cache[n_nodes]
            else:
                nodes, weights = leggauss(n_nodes)
                self._leggauss_cache[n_nodes] = (nodes, weights)

            half_w = homodyne_window / 2.0
            center = homodyne_x if homodyne_x is not None else 0.0
            xs = center + half_w * nodes
            dx_weights_np = half_w * weights

            V_np = get_phi_matrix_cached(self.cutoff, xs, hbar=HBAR)
            V_matrix = jnp.array(V_np)
            dx_weights = jnp.array(dx_weights_np)

        # Call JIT function
        from src.simulation.jax.composer import jax_compose_pair

        return jax_compose_pair(
            stateA,
            stateB,
            U,
            pA,
            pB,
            hom_x_val,
            hom_win_val,
            hom_res_val,
            phi_vec,
            V_matrix,
            dx_weights,
            self.cutoff,
            homodyne_window is None,
            homodyne_x is None,
            homodyne_resolution is None,
            theta=theta,
            phi=phi,
        )


# -----------------------------------------------------------
# SuperblockTopology — generalized evaluator supporting pure fast path
# -----------------------------------------------------------
class SuperblockTopology:
    """
    Build nested pairing plans and evaluate them using Composer.

    Plan representation:
      - ("leaf", idx)
      - ("pair", left, right)

    evaluate_topology(..., exact_mixed=False)
      - exact_mixed=False: try fast pure-state traversal; if a node requires mixed-state (homodyne_window or tracing),
                         that node and its ancestors are computed exactly with densities.
      - exact_mixed=True: always use exact density propagation.
    """

    def __init__(self, plan: Any):
        self.plan = plan

    @staticmethod
    def from_full_binary(depth: int) -> "SuperblockTopology":
        def build_range(start, length):
            if length == 1:
                return ("leaf", start)
            half = length // 2
            return ("pair", build_range(start, half), build_range(start + half, half))

        n = 2**depth
        return SuperblockTopology(build_range(0, n))

    @staticmethod
    def build_layered(num_blocks: int) -> "SuperblockTopology":
        items = [("leaf", i) for i in range(num_blocks)]
        while len(items) > 1:
            next_items = []
            i = 0
            while i < len(items):
                if i + 1 < len(items):
                    next_items.append(("pair", items[i], items[i + 1]))
                    i += 2
                else:
                    next_items.append(items[i])
                    i += 1
            items = next_items
        return SuperblockTopology(items[0])

    # count leaves
    def _count_leaves(self, node) -> int:
        if node[0] == "leaf":
            return 1
        return self._count_leaves(node[1]) + self._count_leaves(node[2])

    def evaluate_topology(
        self,
        composer: Composer,
        fock_vecs: List[np.ndarray],
        p_heralds: List[float],
        homodyne_x: float = 0.0,
        homodyne_window: Optional[float] = None,
        homodyne_resolution: Optional[float] = None,
        theta: float = math.pi / 4.0,
        phi: float = 0.0,
        exact_mixed: bool = False,
        n_hom_points: int = 201,
        pure_only: bool = False,
    ) -> Tuple[Union[np.ndarray, np.ndarray], float]:
        """
        Evaluate plan with provided composer.

        - fock_vecs: list of single-mode state vectors (pure) for each leaf (length = leaf_count)
        - p_heralds: matching list of herald probabilities
        - homodyne_x / homodyne_window: homodyne measurement parameters applied at each pairing
        - exact_mixed: if True, force exact mixed-state propagation for the whole tree
        Returns (final_state, joint_probability)
          - final_state: 1D vector if pure path possible, else density matrix (2D)
        """

        leaf_count = self._count_leaves(self.plan)
        if len(fock_vecs) != leaf_count or len(p_heralds) != leaf_count:
            raise ValueError(
                f"Expected {leaf_count} fock_vecs/p_heralds got {len(fock_vecs)}/{len(p_heralds)}"
            )

        mapping = {
            i: (np.asarray(fock_vecs[i], dtype=complex), float(p_heralds[i]))
            for i in range(leaf_count)
        }

        # recursive evaluator that attempts to return pure vector whenever possible
        # returns tuple (is_pure:bool, state (vec or dm), joint_prob:float)
        def eval_node(node) -> Tuple[bool, Union[np.ndarray, np.ndarray], float]:
            typ = node[0]
            if typ == "leaf":
                idx = node[1]
                vec, p = mapping[idx]
                return True, vec, p
            elif typ == "pair":
                left = node[1]
                right = node[2]
                left_pure, left_state, p_left = eval_node(left)
                right_pure, right_state, p_right = eval_node(right)

                # decide whether this node can remain in pure-path
                # pure path possible if:
                #  - exact_mixed == False
                #  - both children are pure vectors
                #  - homodyne_window is None (point homodyne ok)
                pure_possible = (
                    (not exact_mixed)
                    and left_pure
                    and right_pure
                    and (homodyne_window is None)
                )

                if pure_possible:
                    # compose using pure fast path (composer.compose_pair_cached handles detection)
                    vecL = left_state if left_state.ndim == 1 else None
                    vecR = right_state if right_state.ndim == 1 else None
                    if (vecL is None) or (vecR is None):
                        # fall back to mixed path
                        pure_possible = False
                    else:
                        # call composer with point homodyne or no homodyne (if homodyne_x is None)
                        state_out, p_hom_or_density, joint = (
                            composer.compose_pair_cached(
                                vecL,
                                vecR,
                                p_left,
                                p_right,
                                homodyne_x=homodyne_x,
                                homodyne_window=None,  # pure_possible only if window None
                                homodyne_resolution=homodyne_resolution,
                                theta=theta,
                                phi=phi,
                                n_hom_points=n_hom_points,
                            )
                        )
                        # composer returns vector for pure homodyne point, or density if no measurement leads to mixed
                        # detect returned type
                        if state_out.ndim == 1:
                            # pure continuation
                            return True, state_out, joint
                        else:
                            # got a density -> can't continue pure path above
                            # If pure_only is enforced, we must fail here!
                            if pure_only:
                                raise ValueError(
                                    "Pure-only mode: failed to maintain purity (mixed state produced)"
                                )
                            return False, state_out, joint

                # If pure_only is True: require both children are pure vectors and homodyne_window must be None
                if pure_only:
                    if not pure_possible:
                        # Try to diagnose why pure_possible failed
                        if not (left_pure and right_pure):
                            raise ValueError("Pure-only mode: child not pure")
                        if homodyne_window is not None:
                            raise ValueError(
                                "Pure-only mode: homodyne window requested (breaks purity)"
                            )
                        # If we are here, it means pure_possible was False but maybe because of vecL/vecR check or state_out check
                        # But if left_pure and right_pure are True, vecL/vecR should be vectors.
                        # The only other case is if state_out returned density.
                        # Re-run logic explicitly for error message if needed, or just raise.
                        raise ValueError(
                            "Pure-only mode: failed to maintain purity (mixed state produced)"
                        )

                # else: do exact mixed-state composition
                # ensure densities for children
                def ensure_dm_local(s, is_pure_flag):
                    if is_pure_flag:
                        return np.outer(s, s.conj())
                    else:
                        return s

                rho_left = ensure_dm_local(left_state, left_pure)
                rho_right = ensure_dm_local(right_state, right_pure)

                # compose densities exactly and apply homodyne/window if requested
                rho_out, p_hom_val, joint = composer.compose_pair_cached(
                    rho_left,
                    rho_right,
                    p_left,
                    p_right,
                    homodyne_x=homodyne_x,
                    homodyne_window=homodyne_window,
                    homodyne_resolution=homodyne_resolution,
                    theta=theta,
                    phi=phi,
                    n_hom_points=n_hom_points,
                )
                # rho_out is a density
                return False, rho_out, joint

            else:
                raise ValueError("Unknown node type")

        is_pure_final, state_final, joint_prob = eval_node(self.plan)
        return state_final, joint_prob
