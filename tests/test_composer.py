import pytest
import numpy as np
import os
import time
import sys
from src.simulation.cpu.composer import Composer, global_composer_cache
from src.simulation.cpu.circuit import (
    GaussianHeraldCircuit,
    global_circuit_cache,
)


# Helper to clear caches before/after tests
@pytest.fixture(autouse=True)
def clear_caches():
    global_composer_cache.clear()
    global_circuit_cache.clear()
    yield
    global_composer_cache.clear()
    global_circuit_cache.clear()


def test_composer_caching():
    cutoff = 30
    composer = Composer(cutoff=cutoff)

    fock1 = np.zeros(cutoff)
    fock1[1] = 1.0  # |1>
    fock2 = np.zeros(cutoff)
    fock2[0] = 1.0  # |0>

    # First call: should compute and cache
    start = time.time()
    rho1, _, _ = composer.compose_pair_cached(fock1, fock2, theta=np.pi / 4)
    dur1 = time.time() - start

    # Check if cache has entry
    # We can't easily check internal cache keys without mocking, but we can check stats
    stats = global_composer_cache.stats()
    assert stats["curr_size"] > 0

    # Second call: should be faster (though for small cutoff diff might be negligible, logic check is better)
    start = time.time()
    rho2, _, _ = composer.compose_pair_cached(fock1, fock2, theta=np.pi / 4)
    dur2 = time.time() - start

    np.testing.assert_allclose(rho1, rho2)
    assert dur2 < dur1  # Flaky on small examples


def test_gaussian_herald_caching():
    n_signal = 1
    n_control = 1
    tmss_squeezing = [1.0]
    circ = GaussianHeraldCircuit(n_signal, n_control, tmss_squeezing)
    circ.build()

    pnr = [1]

    # First call
    res1 = circ.herald(pnr, signal_cutoff=5)

    stats = global_circuit_cache.stats()
    assert stats["curr_size"] > 0

    # Second call
    res2 = circ.herald(pnr, signal_cutoff=5)

    # Check equality
    np.testing.assert_allclose(res1[0], res2[0])
    assert res1[1] == res2[1]


def test_imports_and_paths():
    # Just verifying that we can import everything without error
    assert global_composer_cache.cache_dir.endswith("cache")
    assert global_circuit_cache.cache_dir.endswith("cache")
    assert os.path.exists(global_composer_cache.cache_dir)


if __name__ == "__main__":
    # Manual run
    try:
        clear_caches()
        test_composer_caching()
        print("test_composer_caching PASSED")
        test_gaussian_herald_caching()
        print("test_gaussian_herald_caching PASSED")
        test_imports_and_paths()
        print("test_imports_and_paths PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
