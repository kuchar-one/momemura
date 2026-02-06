"""
Eigenvalue cache for truncated GKP operators.

This module precomputes and caches the ground state eigenvalue of the GKP
operator at different truncation dimensions. Used to determine the "achievable"
limit for states with different numbers of detected photons.
"""

import os
import numpy as np
from typing import Tuple, Optional

CACHE_DIR = "cache/eigenvalues"


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_truncated_eigenvalue(
    cutoff: int,
    alpha: complex = 1.0,
    beta: complex = 0.0,
) -> float:
    """
    Get the minimum eigenvalue of the GKP operator truncated to `cutoff` dimensions.

    For small cutoffs (1-3), the truncated operator has fewer Fock states than the
    GKP code structure, so its ground state eigenvalue is higher than 2/3.
    As cutoff increases, the eigenvalue approaches the true GKP limit.

    Args:
        cutoff: Truncation dimension (Fock space size).
        alpha: Target state coefficient for |0>_L.
        beta: Target state coefficient for |1>_L.

    Returns:
        The minimum eigenvalue of the truncated operator.
    """
    if cutoff < 1:
        return 1.0  # No Fock states, return maximum (worst) expectation

    # For very small cutoffs, compute directly
    if cutoff <= 10:
        return _compute_eigenvalue(cutoff, alpha, beta)

    # For larger cutoffs, use cache
    cache_key = _make_cache_key(alpha, beta)
    cache_path = os.path.join(CACHE_DIR, f"eigenvalues_{cache_key}.npy")

    _ensure_cache_dir()

    # Load or generate cache
    if os.path.isfile(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
            if cutoff in cache:
                return float(cache[cutoff])
        except Exception:
            pass

    # Compute and cache
    eigenvalue = _compute_eigenvalue(cutoff, alpha, beta)

    # Update cache
    try:
        if os.path.isfile(cache_path):
            cache = np.load(cache_path, allow_pickle=True).item()
        else:
            cache = {}
        cache[cutoff] = eigenvalue
        np.save(cache_path, cache)
    except Exception:
        pass  # Cache write failure is not critical

    return float(eigenvalue)


def _make_cache_key(alpha: complex, beta: complex) -> str:
    """Create a cache key from alpha and beta."""
    # Normalize and extract key parameters
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    if norm > 1e-9:
        alpha = alpha / norm
        beta = beta / norm

    # Round to avoid floating point key issues
    a_re = round(float(np.real(alpha)), 6)
    a_im = round(float(np.imag(alpha)), 6)
    b_re = round(float(np.real(beta)), 6)
    b_im = round(float(np.imag(beta)), 6)

    return f"a{a_re}_{a_im}_b{b_re}_{b_im}"


def _compute_eigenvalue(cutoff: int, alpha: complex, beta: complex) -> float:
    """Compute the minimum eigenvalue of the truncated GKP operator."""
    from src.utils.gkp_operator import construct_gkp_operator

    operator = construct_gkp_operator(cutoff, alpha, beta, backend="thewalrus")
    eigenvalues = np.linalg.eigvalsh(operator)
    return float(np.min(eigenvalues))


def get_suspicion_threshold(
    total_pnr: int,
    gaussian_limit: float = 2.0 / 3.0,
    alpha: complex = 1.0,
    beta: complex = 0.0,
) -> float:
    """
    Get the suspicion threshold for artifact detection.

    A solution is suspicious if exp_val < threshold.
    The threshold is min(gaussian_limit, truncated_eigenvalue(total_pnr)).

    For low pnr (0, 1, 2, 3), the truncated eigenvalue >= gaussian_limit,
    so the threshold is just gaussian_limit.
    For higher pnr, the truncated eigenvalue may be lower.

    Args:
        total_pnr: Total number of detected photons.
        gaussian_limit: The Gaussian achievable limit (default 2/3).
        alpha: Target state coefficient.
        beta: Target state coefficient.

    Returns:
        The suspicion threshold.
    """
    effective_cutoff = max(1, total_pnr)
    truncated_eigen = get_truncated_eigenvalue(effective_cutoff, alpha, beta)
    return min(gaussian_limit, truncated_eigen)


def precompute_eigenvalue_table(
    max_cutoff: int = 50,
    alpha: complex = 1.0,
    beta: complex = 0.0,
) -> np.ndarray:
    """
    Precompute eigenvalues for cutoffs 1..max_cutoff for fast lookup.

    Returns:
        Array of shape (max_cutoff+1,) where arr[n] is the eigenvalue at cutoff n.
        arr[0] = 1.0 (no Fock states).
    """
    eigenvalues = np.ones(max_cutoff + 1)
    for n in range(1, max_cutoff + 1):
        eigenvalues[n] = get_truncated_eigenvalue(n, alpha, beta)
    return eigenvalues
