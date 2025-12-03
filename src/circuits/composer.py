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

from src.utils.cache_manager import CacheManager, _short_hash_bytes, _bytes_key_of_array
from src.circuits.ops import (
    HBAR,
    quadrature_vector,
    build_beamsplitter_unitary,
    _bs_cache_key,
    kron_state_vector,
    contract_rho_with_phi,
)

# Silence potential strawberryfields deprecation noise in importers if present
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# Project-level cache directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
global_composer_cache = CacheManager(
    cache_dir=CACHE_DIR, size_limit_bytes=1024 * 1024 * 1024
)


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
    ):
        self.cutoff = int(cutoff)
        self.cache_enabled = bool(cache_enabled)
        self.cache = cache if cache is not None else global_composer_cache
        # small in-memory caches to avoid hitting diskcache often
        self._U_cache_local = {}
        self._engine_lock = threading.Lock()

    # ------------
    # Helpers
    # ------------
    def _u_bs(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """
        Return (and cache) the beamsplitter unitary for current cutoff/theta/phi.
        Local memory + persistent cache is used.
        """
        key = _bs_cache_key(self.cutoff, theta, phi)
        if key in self._U_cache_local:
            return self._U_cache_local[key]
        U = build_beamsplitter_unitary(self.cutoff, theta, phi, cache=self.cache)
        self._U_cache_local[key] = U
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
            phi_vec = quadrature_vector(self.cutoff, float(homodyne_x), hbar=HBAR)
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
            phi_vec = quadrature_vector(self.cutoff, float(homodyne_x), hbar=HBAR)
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
        xs = np.linspace(
            homodyne_x - homodyne_window / 2.0,
            homodyne_x + homodyne_window / 2.0,
            int(n_hom_points),
        )

        # Vectorized integration
        # 1. Compute all quadrature vectors: shape (cutoff, n_points)
        # quadrature_vector returns (cutoff,), so we stack them.
        # Optimization: quadrature_vector depends on x.
        # We can compute them in a loop or vectorize if quadrature_vector supports it (it likely doesn't).
        # But we can build the matrix V where V[:, i] = phi_vec(xs[i])

        c = self.cutoff
        V = np.zeros((c, len(xs)), dtype=complex)
        for i, x in enumerate(xs):
            V[:, i] = quadrature_vector(c, float(x), hbar=HBAR)

        # 2. Contract rho_out with all phi_vecs
        # rho_out: (c, c)
        # V: (c, N)
        # We want p(x) = <phi_x| rho |phi_x> = sum_{mn} phi_x[m]^* rho[m,n] phi_x[n]
        # Matrix form: diag(V^H @ rho @ V)
        # Efficiently: sum(V.conj() * (rho @ V), axis=0)

        rho_V = rho_out @ V  # (c, N)
        p_xs = np.real(np.sum(V.conj() * rho_V, axis=0))  # (N,)

        # 3. Integrate p(x) to get Pwin
        Pwin = float(np.trapz(p_xs, xs))

        # 4. Compute conditional state: integral of (rho_x * p(x)) / Pwin ?
        # Actually, the conditional state for a window measurement is:
        # rho_cond = \int dx M_x rho M_x^dag / Pwin
        # where M_x is the projection |phi_x><phi_x|.
        # So rho_cond = \int dx |phi_x><phi_x| * p(x)? No.
        # The POVM element is E = \int dx |phi_x><phi_x|.
        # The post-measurement state is \int dx (M_x rho M_x^dag).
        # M_x rho M_x^dag = |phi_x><phi_x| rho |phi_x><phi_x| = |phi_x> p(x) <phi_x|.
        # So rho_cond = \int dx p(x) |phi_x><phi_x|.

        # We can compute this integral:
        # Racc = sum_i (p_xs[i] * outer(V[:,i], V[:,i].conj())) * dx

        dx = xs[1] - xs[0]

        # Vectorized Racc accumulation?
        # Racc = V @ diag(p_xs) @ V.H * dx
        # V: (c, N)
        # V * sqrt(p_xs): (c, N) scaled columns
        # Let W = V * sqrt(p_xs)
        # Racc = W @ W.H * dx

        # Ensure p_xs is non-negative (numerical noise might make it slightly negative)
        p_xs_safe = np.maximum(p_xs, 0.0)
        W = V * np.sqrt(p_xs_safe)[None, :]
        Racc = W @ W.conj().T * dx

        if Pwin > 0:
            rho_cond = Racc / Pwin
        else:
            rho_cond = np.zeros((c, c), dtype=complex)

        joint = float(pA * pB * Pwin)
        return rho_cond, Pwin, joint

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
        # stable serialization
        a_bytes = _bytes_key_of_array(np.ascontiguousarray(stateA))
        b_bytes = _bytes_key_of_array(np.ascontiguousarray(stateB))
        key_bytes = (
            a_bytes
            + b_bytes
            + b"|"
            + str(self.cutoff).encode()
            + b"|"
            + repr(float(theta)).encode()
            + b"|"
            + repr(float(phi)).encode()
            + b"|"
            + (b"None" if homodyne_x is None else repr(float(homodyne_x)).encode())
            + b"|"
            + (
                b"None"
                if homodyne_window is None
                else repr(float(homodyne_window)).encode()
            )
            + b"|"
            + (
                b"None"
                if homodyne_resolution is None
                else repr(float(homodyne_resolution)).encode()
            )
            + b"|"
            + repr(int(n_hom_points)).encode()
        )
        key = "compose_pair_v3:" + _short_hash_bytes(key_bytes)

        if self.cache_enabled:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        res = self.compose_pair(
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
        if self.cache_enabled:
            self.cache.set(key, res)
        return res

    def clear_caches(self):
        """Clear the persistent cache (useful between experiments)."""
        if self.cache_enabled:
            self.cache.clear()
        self._U_cache_local.clear()


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


# -----------------------------------------------------------
# Small self-test when run as script (smoke tests)
# -----------------------------------------------------------
if __name__ == "__main__":
    # quick smoke: HOM |1,1> -> no |1,1> at output for theta=pi/4, phi=0
    c = 6
    comp = Composer(cutoff=c)
    f1 = np.zeros(c, dtype=complex)
    f1[1] = 1.0
    f2 = np.zeros(c, dtype=complex)
    f2[1] = 1.0
    out, p, joint = comp.compose_pair_cached(
        f1, f2, homodyne_x=None, homodyne_window=None, theta=math.pi / 4, phi=0.0
    )
    # out is reduced density for mode0
    idx11 = 1 * c + 1
    # check full rho via U_bs
    U = comp._u_bs(theta=math.pi / 4, phi=0.0)
    psi_in = np.kron(f1, f2)
    psi_out = U @ psi_in
    rho_full = np.outer(psi_out, psi_out.conj())
    p11 = np.real(rho_full[idx11, idx11])
    print("HOM p(|1,1>) full-state:", p11)
    print("reduced mode0 diagonal (photon probs) example:", np.real(np.diag(out))[:6])
