"""
Tests for the eigenvalue cache module.
"""

import numpy as np
import pytest


def test_get_truncated_eigenvalue_monotonic():
    """Test that eigenvalues monotonically decrease with cutoff."""
    from src.utils.eigenvalue_cache import get_truncated_eigenvalue

    # Compute eigenvalues for cutoffs 1..15
    eigenvalues = []
    for n in range(1, 16):
        eig = get_truncated_eigenvalue(n, alpha=1.0, beta=0.0)
        eigenvalues.append(eig)

    # They should be monotonically decreasing (larger cutoff = lower eigenvalue)
    for i in range(len(eigenvalues) - 1):
        assert eigenvalues[i] >= eigenvalues[i + 1], (
            f"Eigenvalue at cutoff {i + 1} ({eigenvalues[i]}) should be >= "
            f"eigenvalue at cutoff {i + 2} ({eigenvalues[i + 1]})"
        )


def test_get_truncated_eigenvalue_bounds():
    """Test that eigenvalues are in reasonable range."""
    from src.utils.eigenvalue_cache import get_truncated_eigenvalue

    # For cutoff=1, eigenvalue should be positive (only vacuum state, highly truncated)
    eig_1 = get_truncated_eigenvalue(1, alpha=1.0, beta=0.0)
    assert eig_1 > 0, f"Eigenvalue at cutoff=1 should be positive, got {eig_1}"

    # For cutoff=30, eigenvalue should be well below 2/3
    # (more Fock states = better approximation of true ground state)
    eig_30 = get_truncated_eigenvalue(30, alpha=1.0, beta=0.0)
    assert 0.0 <= eig_30 <= 0.5, f"Eigenvalue at cutoff=30 should be low, got {eig_30}"


def test_get_suspicion_threshold_low_pnr():
    """Test suspicion threshold for low photon numbers."""
    from src.utils.eigenvalue_cache import get_suspicion_threshold

    gaussian_limit = 2.0 / 3.0

    # For pnr=0, the threshold should be gaussian_limit since
    # using effective_cutoff=1 gives eigenvalue >= 2/3
    threshold_0 = get_suspicion_threshold(0, gaussian_limit=gaussian_limit)
    assert threshold_0 == gaussian_limit, (
        f"Threshold for pnr=0 should be gaussian_limit ({gaussian_limit}), got {threshold_0}"
    )

    # For pnr=1,2,3, threshold should still be gaussian_limit
    for pnr in [1, 2, 3]:
        threshold = get_suspicion_threshold(pnr, gaussian_limit=gaussian_limit)
        # At these low cutoffs, truncated eigenvalue >= gaussian_limit
        assert threshold <= gaussian_limit + 0.01, (
            f"Threshold for pnr={pnr} unexpected: {threshold}"
        )


def test_get_suspicion_threshold_high_pnr():
    """Test that suspicion threshold drops below gaussian_limit for high pnr."""
    from src.utils.eigenvalue_cache import get_suspicion_threshold

    gaussian_limit = 2.0 / 3.0

    # For high enough pnr, threshold should be below gaussian_limit
    threshold_20 = get_suspicion_threshold(20, gaussian_limit=gaussian_limit)
    assert threshold_20 < gaussian_limit, (
        f"Threshold for pnr=20 should be below gaussian_limit ({gaussian_limit}), got {threshold_20}"
    )


def test_eigenvalue_cache_consistency():
    """Test that repeated calls return the same value (cache works)."""
    from src.utils.eigenvalue_cache import get_truncated_eigenvalue

    # First call
    eig1 = get_truncated_eigenvalue(15, alpha=1.0, beta=0.0)

    # Second call (should hit cache)
    eig2 = get_truncated_eigenvalue(15, alpha=1.0, beta=0.0)

    assert eig1 == eig2, f"Cache inconsistency: {eig1} != {eig2}"


if __name__ == "__main__":
    test_get_truncated_eigenvalue_monotonic()
    test_get_truncated_eigenvalue_bounds()
    test_get_suspicion_threshold_low_pnr()
    test_get_suspicion_threshold_high_pnr()
    test_eigenvalue_cache_consistency()
    print("All eigenvalue cache tests passed!")
