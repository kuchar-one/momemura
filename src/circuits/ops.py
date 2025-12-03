"""
ops.py

Common quantum operations and helpers for Hanamura circuits.
Includes:
- Constants (HBAR)
- Quadrature functions
- Single-mode operators (annihilation, creation)
- Beamsplitter builders (Fock basis and 2x2 matrix)
- State vector helpers
"""

import numpy as np
import math
from typing import Optional, Tuple
from scipy.special import eval_hermite, gammaln
from scipy.linalg import expm
from src.utils.accel import njit_wrapper as njit
from src.utils.cache_manager import CacheManager

# -----------------------------------------------------------
# Global numeric conventions
# -----------------------------------------------------------
HBAR = 2.0  # standardize to hbar=2 everywhere (Walrus-friendly choice)


# -----------------------------------------------------------
# Quadrature helpers (consistent with general hbar)
# -----------------------------------------------------------
def quadrature_wavefunction(n: int, x: float, hbar: float = HBAR) -> float:
    """
    Harmonic oscillator quadrature wavefunction psi_n(x) for general hbar.
    Formula:
        psi_n(x) = (1 / (pi*hbar)^(1/4)) * (1/sqrt(2^n n!)) * H_n(x/sqrt(hbar)) * exp(-x^2/(2 hbar))
    Returns a Python float.
    """
    arg = x / math.sqrt(hbar)
    pref = (math.pi * hbar) ** (-0.25) / math.sqrt((2.0**n) * math.exp(gammaln(n + 1)))
    Hn = eval_hermite(n, arg)
    return float(pref * Hn * math.exp(-0.5 * arg * arg))


def quadrature_vector(cutoff: int, x: float, hbar: float = HBAR) -> np.ndarray:
    """
    Build the quadrature vector phi_n(x) (length=cutoff) for given x and hbar.
    """
    return np.array(
        [quadrature_wavefunction(n, x, hbar=hbar) for n in range(cutoff)], dtype=float
    )


# -----------------------------------------------------------
# Single-mode ladder operators (truncated)
# -----------------------------------------------------------
def annihilation_operator(cutoff: int) -> np.ndarray:
    "Return the single-mode annihilation operator (cutoff x cutoff)."
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(1, cutoff):
        a[n - 1, n] = math.sqrt(n)
    return a


def creation_operator(cutoff: int) -> np.ndarray:
    "Return single-mode creation operator (cutoff x cutoff)."
    return annihilation_operator(cutoff).conj().T


# -----------------------------------------------------------
# Beamsplitter builder (expm generator) with caching
# -----------------------------------------------------------
def _bs_cache_key(
    cutoff: int, theta: float, phi: float
) -> Tuple[str, int, float, float]:
    return ("U_bs", int(cutoff), float(theta), float(phi))


def build_beamsplitter_unitary(
    cutoff: int, theta: float, phi: float = 0.0, cache: Optional[CacheManager] = None
) -> np.ndarray:
    r"""
    Build the full two-mode beamsplitter unitary U_bs of shape (c^2, c^2) in the truncated Fock basis,
    using the second-quantized generator and matrix exponential:

        K = theta * ( e^{i phi} a1^\dag a2 - e^{-i phi} a1 a2^\dag )
        U_bs = expm(K)

    This construction is numerically stable (matrix exponential of a sparse operator).
    - cutoff: single-mode cutoff dimension
    - theta: mixing angle (so that for phi=0 the transformation of annihilation ops is the usual rotation)
    - phi: beamsplitter internal phase (real)
    The result is cached (if cache is provided) using CacheManager.
    """
    key = _bs_cache_key(cutoff, theta, phi)

    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    # build single-mode operators
    a = annihilation_operator(cutoff)
    identity = np.eye(cutoff, dtype=complex)

    # two-mode operators in full Hilbert space
    a1 = np.kron(a, identity)
    a2 = np.kron(identity, a)
    adag1 = a1.conj().T
    adag2 = a2.conj().T

    # generator K
    K = np.exp(1j * phi) * np.dot(adag1, a2) - np.exp(-1j * phi) * np.dot(a1, adag2)
    K = float(theta) * K  # scale

    # exponentiate
    U = expm(K)

    # cache and return
    if cache is not None:
        cache.set(key, U)
    return U


@njit
def beamsplitter_2x2(theta: float, phi: float) -> np.ndarray:
    """
    2x2 complex beam-splitter matrix B(theta, phi) acting on [a_i, a_j]^T such that
    B = [[cosθ, -e^{-iφ} sinθ],
         [e^{iφ} sinθ, cosθ]].

    This choice matches a common parameterization used in interferometer templates.
    """
    t = np.cos(theta)
    r = np.sin(theta)
    return np.array(
        [[t, -np.exp(-1j * phi) * r], [np.exp(1j * phi) * r, t]], dtype=np.complex128
    )


# -----------------------------------------------------------
# Utility: kron of vector -> full two-mode state vector
# -----------------------------------------------------------
def kron_state_vector(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Return the Kronecker (tensor) product vector for two single-mode states.
    Both f1 and f2 may be 1D vectors (length = cutoff).
    """
    return np.kron(f1, f2)


# -----------------------------------------------------------
# Homodyne helpers
# -----------------------------------------------------------
def contract_rho_with_phi(rho_full: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Given rho_full shaped (c^2, c^2) and phi (length c), compute unnormalized reduced
    rho for mode1 after projecting mode2 onto |x> via quadrature vector phi.
    Implementation uses tensordot contraction:
       rho_t[n0,n1,m0,m1] -> contract n1 and m1 with phi -> new_rho[n0,m0]
    """
    c = int(round(np.sqrt(rho_full.shape[0])))
    rho_t = rho_full.reshape((c, c, c, c))
    tmp = np.tensordot(rho_t, phi, axes=([1], [0]))  # shape (n0,m0,m1)
    new_rho = np.tensordot(tmp, phi, axes=([2], [0]))  # shape (n0,m0)
    return new_rho
