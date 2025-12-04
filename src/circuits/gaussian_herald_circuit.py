"""
gaussian_herald_circuit.py

Robust implementation of the canonical-form Gaussian resource + heralding circuit
(Fig.3(d) style). Builds global Gaussian mean/cov from TMSS pairs + passive
interferometers + displacements; efficiently computes conditional signal Fock
amplitudes by calling thewalrus.pure_state_amplitude for only the requested
signal basis states.

Includes a Clements rectangular-mesh builder that converts packed theta/phi/varphi
parameters into an explicit unitary matrix U (drop-in compatible with PennyLane's
Interferometer(mesh='rectangular')).
"""

# Apply SciPy patch for StrawberryFields compatibility
import src.utils.scipy_patch  # noqa: F401

from typing import Sequence, Tuple, Optional
import numpy as np
import itertools
from thewalrus.quantum import pure_state_amplitude
import matplotlib.pyplot as plt
import os
from src.utils.params import (
    two_mode_squeezer_symplectic,
    vacuum_covariance,
    xp_to_interleaved,
    expand_mode_symplectic,
    passive_unitary_to_symplectic,
    complex_alpha_to_qp,
    interleaved_to_xp,
)
from src.utils.accel import njit_wrapper as njit
from src.utils.cache_manager import CacheManager, _short_hash_bytes
from src.circuits.ops import beamsplitter_2x2

# create global circuit cache (persist on disk)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
# Global cache instance for circuits
global_circuit_cache = CacheManager(
    cache_dir=CACHE_DIR, size_limit_bytes=500 * 1024 * 1024
)

# In-memory LRU cache for unitaries
_UNITARY_CACHE = {}
_UNITARY_CACHE_LIMIT = 20000

# -----------------------------
# Beam-splitter and mesh builder
# -----------------------------


@njit
def interferometer_params_to_unitary(
    theta: np.ndarray,
    phi: np.ndarray,
    varphi: np.ndarray,
    M: int,
    mesh: str = "rectangular",
) -> np.ndarray:
    """
    Convert interferometer parameters (theta, phi, varphi) to an explicit MxM unitary
    following the rectangular (Clements) mesh ordering.

    Parameters
    ----------
    theta : length M(M-1)/2 list of transmittivity angles
    phi : length M(M-1)/2 list of beam-splitter phase angles
    varphi : length M list of final local rotation angles (phases)
    M : int : number of modes
    mesh : 'rectangular' (default) or 'triangular' (not implemented here)

    Returns
    -------
    U : (M x M) complex unitary matrix
    """
    if mesh != "rectangular":
        raise NotImplementedError("Only 'rectangular' mesh implemented in this helper.")
    # theta = list(theta) # Removed list conversion for Numba
    # phi = list(phi)
    # varphi = list(varphi)
    expected = M * (M - 1) // 2
    if len(theta) != expected or len(phi) != expected:
        raise ValueError(f"theta/phi must have length {expected} for M={M}.")
    if len(varphi) != M:
        raise ValueError("varphi must have length M (one phase per mode).")

    # Create identity
    U = np.eye(M, dtype=np.complex128)

    # Build slices left-to-right; there are M slices indexed s=0..M-1
    # In each slice s, beam splitters act on pairs depending on parity of s:
    #   if s % 2 == 0: pairs (0,1), (2,3), ...
    #   else:          pairs (1,2), (3,4), ...
    # We consume theta/phi parameters in order left-to-right, top-to-bottom per slice
    param_idx = 0
    for s in range(M):
        # start with identity layer
        L = np.eye(M, dtype=np.complex128)
        start = 0 if (s % 2 == 0) else 1
        # pair indices
        for a in range(start, M - 1, 2):
            th = theta[param_idx]
            ph = phi[param_idx]
            param_idx += 1
            B = beamsplitter_2x2(th, ph)
            # embed B into L on rows/cols [a, a+1]
            L_block = L.copy()
            # Ensure contiguous memory for the slice to avoid Numba performance warning
            target_slice = np.ascontiguousarray(L_block[a : a + 2, a : a + 2])
            L_block[a : a + 2, a : a + 2] = B @ target_slice
            L = L_block
        # left-multiply the layer to current U (operations are applied left-to-right)
        U = L @ U

    # Apply final diagonal phases varphi as R = diag(exp(i varphi_j))
    R = np.diag(np.exp(1j * np.asarray(varphi, dtype=np.float64)))
    U = R @ U
    # At this point U should be unitary (up to numerical error). Optionally enforce unitarity.
    return U


# -----------------------------
# Main circuit (class)
# -----------------------------


class GaussianHeraldCircuit:
    """
    Canonical-form Gaussian resource + heralding circuit class.

    Usage:
      circ = GaussianHeraldCircuit(n_signal, n_control, tmss_squeezing, us_params=..., uc_params=..., ...)
      circ.build()
      state, prob = circ.herald(pnr_outcome, signal_cutoff=6)
      circ.print_circuit()
      circ.plot_circuit()
    """

    def __init__(
        self,
        n_signal: int,
        n_control: int,
        tmss_squeezing: Sequence[float],
        us_params: Optional[dict] = None,
        uc_params: Optional[dict] = None,
        U_s: Optional[np.ndarray] = None,
        U_c: Optional[np.ndarray] = None,
        disp_s: Optional[Sequence[complex]] = None,
        disp_c: Optional[Sequence[complex]] = None,
        mesh: str = "rectangular",
        hbar: float = 2.0,
        cache_enabled: bool = True,
    ):
        """
        Initialize circuit builder.

        - You can supply either explicit unitaries U_s/U_c (n x n matrices) OR the
          parameter dicts us_params/uc_params of the form {'theta':..., 'phi':..., 'varphi':...}
          (like PennyLane Interferometer). If both provided, U_* takes precedence.
        - tmss_squeezing is a sequence of r parameters (schmidt_rank).
        - mode ordering is signal-first internally.
        """
        self.n_signal = int(n_signal)
        self.n_control = int(n_control)
        self.tmss_squeezing = list(tmss_squeezing)
        self.us_params = us_params
        self.uc_params = uc_params
        # Store explicit unitaries separately so we know if they were provided
        self._U_s_explicit = (
            np.asarray(U_s, dtype=np.complex128) if U_s is not None else None
        )
        self._U_c_explicit = (
            np.asarray(U_c, dtype=np.complex128) if U_c is not None else None
        )
        # Effective unitaries (will be computed in build/_resolve_unitaries)
        self.U_s = None
        self.U_c = None
        self.disp_s = (
            None if disp_s is None else np.asarray(disp_s, dtype=np.complex128)
        )
        self.disp_c = (
            None if disp_c is None else np.asarray(disp_c, dtype=np.complex128)
        )
        self.mesh = mesh
        self.hbar = float(hbar)
        self.cache_enabled = bool(cache_enabled)
        self.cache = global_circuit_cache

        self.mu = None
        self.cov = None
        self._built = False

        # add small caches
        self._cached_us_key = None
        self._cached_uc_key = None
        self._cached_S_us = None
        self._cached_S_uc = None

        # In-memory cache for signal basis (fast access, no need to persist usually, but we can if needed)
        # For now, we'll keep signal basis in memory as it's small and fast to recompute if needed,
        # but herald results should be persistent.
        self._signal_basis_cache = {}

    # -----------------------------
    # internal: resolve parameter dict -> unitary
    # -----------------------------
    @staticmethod
    def params_dict_to_unitary(
        params: dict, M: int, mesh: str = "rectangular"
    ) -> np.ndarray:
        """
        Convert a params dict {'theta':..., 'phi':..., 'varphi':...} to an explicit MxM unitary.

        This is compatible with the PennyLane Interferometer template parameter shapes.
        """
        theta = params.get("theta")
        phi = params.get("phi")
        varphi = params.get("varphi")
        if theta is None or phi is None or varphi is None:
            raise ValueError("params must contain 'theta','phi','varphi' keys.")

        # Cache lookup
        try:
            t_arr = np.ascontiguousarray(theta, dtype=np.float64)
            p_arr = np.ascontiguousarray(phi, dtype=np.float64)
            v_arr = np.ascontiguousarray(varphi, dtype=np.float64)

            # Key: (M, mesh, bytes of params)
            key = (M, mesh, t_arr.tobytes(), p_arr.tobytes(), v_arr.tobytes())

            if key in _UNITARY_CACHE:
                return _UNITARY_CACHE[key]

            U = interferometer_params_to_unitary(
                t_arr,
                p_arr,
                v_arr,
                M,
                mesh=mesh,
            )

            if len(_UNITARY_CACHE) < _UNITARY_CACHE_LIMIT:
                _UNITARY_CACHE[key] = U
            else:
                # Evict oldest (dict is ordered by insertion in Python 3.7+)
                _UNITARY_CACHE.pop(next(iter(_UNITARY_CACHE)))
                _UNITARY_CACHE[key] = U

            return U

        except Exception:
            # Fallback
            return interferometer_params_to_unitary(
                np.asarray(theta, dtype=np.float64),
                np.asarray(phi, dtype=np.float64),
                np.asarray(varphi, dtype=np.float64),
                M,
                mesh=mesh,
            )

    def _cache_key_for_unitary(self, U: np.ndarray):
        # cheap stable key: use shape + bytes hash; bytes can be large but OK for small unitaries
        if U is None:
            return None
        # use view of real+imag to make deterministic bytes
        data = np.ascontiguousarray(U.view(np.float64))
        return (U.shape, data.tobytes())

    def _resolve_unitaries(self):
        """
        Ensure self.U_s and self.U_c are set: prefer explicit U_s/U_c if provided,
        otherwise try to convert us_params/uc_params dicts using Clements mapping.
        """
        n_sig = self.n_signal
        n_ctrl = self.n_control

        # Resolve U_s
        if self._U_s_explicit is not None:
            self.U_s = self._U_s_explicit
        elif self.us_params is not None:
            self.U_s = self.params_dict_to_unitary(
                self.us_params, n_sig, mesh=self.mesh
            )
        else:
            # If neither explicit nor params, default to identity if not already set or if we want to be safe
            # Actually, if we want to support "set U_s manually after init", we should be careful.
            # But the standard usage is either pass U_s or params.
            # If nothing passed, Identity.
            self.U_s = np.eye(n_sig, dtype=np.complex128)

        # Resolve U_c
        if self._U_c_explicit is not None:
            self.U_c = self._U_c_explicit
        elif self.uc_params is not None:
            self.U_c = self.params_dict_to_unitary(
                self.uc_params, n_ctrl, mesh=self.mesh
            )
        else:
            self.U_c = np.eye(n_ctrl, dtype=np.complex128)

        # after computing self.U_s/self.U_c keep cache of their symplectic xp block
        key_s = self._cache_key_for_unitary(self.U_s)
        if key_s != self._cached_us_key:
            if self.n_signal > 0:
                S_us_xp = passive_unitary_to_symplectic(self.U_s)  # xp order
                S_us_ip = xp_to_interleaved(S_us_xp)
            else:
                S_us_ip = np.eye(0, dtype=float)
            self._cached_S_us = S_us_ip
            self._cached_us_key = key_s
        # same for control
        key_c = self._cache_key_for_unitary(self.U_c)
        if key_c != self._cached_uc_key:
            if self.n_control > 0:
                S_uc_xp = passive_unitary_to_symplectic(self.U_c)
                S_uc_ip = xp_to_interleaved(S_uc_xp)
            else:
                S_uc_ip = np.eye(0, dtype=float)
            self._cached_S_uc = S_uc_ip
            self._cached_uc_key = key_c

    # -----------------------------
    # Build global mu & cov
    # -----------------------------
    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compose symplectic transformations: TMSS pairs -> U_s & U_c passive unitaries ->
        optional displacements. Produces and stores (mu, cov) in q,p ordering.
        """
        n_sig = self.n_signal
        n_ctrl = self.n_control
        N = n_sig + n_ctrl
        schmidt_rank = len(self.tmss_squeezing)
        if schmidt_rank > min(n_sig, n_ctrl):
            raise ValueError("schmidt_rank cannot exceed min(n_signal, n_control).")

        # resolve unitaries and cached S_us/S_uc
        self._resolve_unitaries()

        cov_vac = vacuum_covariance(N, hbar=self.hbar)
        mu = np.zeros(2 * N, dtype=float)
        S_total = np.eye(2 * N, dtype=float)

        # apply TMSS squeezers
        for i, r in enumerate(self.tmss_squeezing):
            sig_idx = n_sig - schmidt_rank + i
            ctrl_idx = n_sig + i
            smallS = two_mode_squeezer_symplectic(r)
            bigS = expand_mode_symplectic(
                smallS, np.array([sig_idx, ctrl_idx], dtype=np.int64), N
            )
            S_total = bigS @ S_total
            mu = bigS @ mu

        # use cached S_us/S_uc instead of recomputing
        S_block = np.zeros((2 * N, 2 * N), dtype=float)
        S_block[: 2 * n_sig, : 2 * n_sig] = self._cached_S_us
        S_block[2 * n_sig :, 2 * n_sig :] = self._cached_S_uc

        S_total = S_block @ S_total
        mu = S_block @ mu

        # displacements
        displ_qp = np.zeros(2 * N, dtype=float)
        if self.disp_s is not None:
            if len(self.disp_s) != n_sig:
                raise ValueError("disp_s length mismatch for signal.")
            displ_qp[: 2 * n_sig] += complex_alpha_to_qp(self.disp_s)
        if self.disp_c is not None:
            if len(self.disp_c) != n_ctrl:
                raise ValueError("disp_c length mismatch for control.")
            displ_qp[2 * n_sig : 2 * (n_sig + n_ctrl)] += complex_alpha_to_qp(
                self.disp_c
            )
        mu = mu + displ_qp

        cov = S_total @ cov_vac @ S_total.T

        self.mu = mu
        self.cov = cov
        # convert and cache xp ordering for walrus once
        self.mu_xp, self.cov_xp = interleaved_to_xp(mu, cov)
        self._built = True
        # We don't clear the persistent cache here, but we might want to invalidate entries if we were using instance-specific keys.
        # However, the cache keys include mu/cov or similar, so it's fine.
        # Wait, the previous implementation cleared self._herald_cache.
        # If we use persistent cache, we should ensure keys are unique to the current state.
        # The herald cache key in the previous implementation was just (pnr_outcome, ...).
        # It implicitly depended on self.mu_xp/cov_xp being current.
        # For persistent cache, we MUST include mu/cov in the key.
        return mu, cov

    # -----------------------------
    # Heralding (efficient)
    # -----------------------------
    def herald(
        self,
        pnr_outcome: Sequence[int],
        signal_cutoff: int = 10,
        check_purity: bool = False,
    ) -> Tuple[np.ndarray, float]:
        if not self._built:
            raise RuntimeError("Call build() before herald().")
        n_sig = self.n_signal
        n_ctrl = self.n_control
        if len(pnr_outcome) != n_ctrl:
            raise ValueError("pnr_outcome length mismatch.")

        # get xp mu/cov once (already computed in build)
        mu_xp = self.mu_xp
        cov_xp = self.cov_xp

        # cache key for repeated herald calls
        # Must include mu/cov in the key for persistent caching!
        # We can hash mu_xp and cov_xp
        state_bytes = mu_xp.tobytes() + b"|" + cov_xp.tobytes()
        state_hash = _short_hash_bytes(state_bytes)

        cache_key_tuple = (
            state_hash,
            tuple(int(x) for x in pnr_outcome),
            int(signal_cutoff),
            bool(check_purity),
        )
        # Convert tuple to string key or rely on CacheManager to handle it (it handles pickleable objects)
        # But let's make a string key to be safe/clean
        cache_key = "herald:" + _short_hash_bytes(str(cache_key_tuple).encode())

        if self.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Precompute signal basis once per cutoff and n_signal using an internal cache
        basis_key = (n_sig, signal_cutoff)
        signal_basis = self._signal_basis_cache.get(basis_key)
        if signal_basis is None:
            # small optimization: generate as list of tuples (faster indexing)
            signal_basis = list(itertools.product(range(signal_cutoff), repeat=n_sig))
            self._signal_basis_cache[basis_key] = signal_basis

        unnorm = []
        append = unnorm.append
        clean_pnr = [int(x) for x in pnr_outcome]

        # Localize to local variables for speed
        psa = pure_state_amplitude
        hbar = self.hbar
        cp = check_purity

        for sig in signal_basis:
            full_pattern = list(sig) + clean_pnr
            amp = psa(mu_xp, cov_xp, full_pattern, hbar=hbar, check_purity=cp)
            append(amp)

        unnorm = np.array(unnorm, dtype=np.complex128)
        prob = float(np.sum(np.abs(unnorm) ** 2))
        if prob > 1e-15:
            norm = unnorm / np.sqrt(prob)
        else:
            norm = np.zeros_like(unnorm)
        if n_sig > 1:
            norm = norm.reshape([signal_cutoff] * n_sig)

        # store in cache and return
        res = (norm, prob)
        if self.cache_enabled:
            self.cache.set(cache_key, res)
        return res

    # Add a small accessor:
    def get_mu_cov_xp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mu_xp, cov_xp) — mu and cov in walrus ordering (xp). Requires build()."""
        if not self._built:
            raise RuntimeError("Call build() first.")
        return self.mu_xp, self.cov_xp

    # -----------------------------
    # Utilities
    # -----------------------------
    def print_circuit(self):
        """Print a compact ASCII summary of the circuit layout and parameters."""
        lines = []
        lines.append("GaussianHeraldCircuit:")
        lines.append(f"  signal modes:  {self.n_signal}")
        lines.append(f"  control modes: {self.n_control}")
        lines.append(f"  schmidt_rank:  {len(self.tmss_squeezing)}")
        lines.append(f"  mesh: {self.mesh}")
        lines.append(f"  U_s provided? {'yes' if self.U_s is not None else 'no'}")
        lines.append(f"  U_c provided? {'yes' if self.U_c is not None else 'no'}")
        print("\n".join(lines))

    def plot_circuit(self):
        """Draw a quick matplotlib wire+blocks sketch of the circuit."""
        n_sig = self.n_signal
        k = self.n_control
        total = n_sig + k
        fig, ax = plt.subplots(figsize=(6, 0.5 * total + 1))
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, total)
        ax.axis("off")
        for i in range(total):
            y = total - 1 - i
            ax.hlines(y, 0.1, 9.9, linewidth=1, color="black")
            label = f"s{i}" if i < n_sig else f"c{i - n_sig}"
            ax.text(0.0, y, label, va="center", ha="left", fontsize=10)
        # TMSS red links
        schmidt_rank = len(self.tmss_squeezing)
        for i in range(schmidt_rank):
            sig_idx = n_sig - schmidt_rank + i
            ctrl_idx = n_sig + i
            y_sig = total - 1 - sig_idx
            y_ctrl = total - 1 - ctrl_idx
            ax.plot([3.5, 3.5], [y_ctrl, y_sig], color="red", linewidth=2)
            ax.add_patch(plt.Circle((3.5, 0.5 * (y_ctrl + y_sig)), 0.12, color="red"))
            ax.text(
                3.7,
                0.5 * (y_ctrl + y_sig),
                f"r={self.tmss_squeezing[i]:.2f}",
                va="center",
                fontsize=8,
            )
        # boxes
        ax.add_patch(
            plt.Rectangle(
                (5.5, total - n_sig - 0.4), 2.5, n_sig, fill=False, linewidth=1.5
            )
        )
        ax.text(6.8, total - n_sig / 2 - 0.4, "U_s", ha="center", va="center")
        ax.add_patch(plt.Rectangle((5.5, -0.4), 2.5, k, fill=False, linewidth=1.5))
        ax.text(6.8, k / 2 - 0.4, "U_c", ha="center", va="center")
        plt.show()
