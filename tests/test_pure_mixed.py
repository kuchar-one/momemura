import numpy as np
import math
import pytest
from src.simulation.cpu.composer import Composer, SuperblockTopology


def test_hom_dip_pure_vs_mixed():
    """
    Verify Hong-Ou-Mandel dip gives same result (zero coincidence)
    for both pure (analytical) and mixed (density) paths.
    """
    cutoff = 5
    composer = Composer(cutoff=cutoff)

    # |1> state
    f1 = np.zeros(cutoff, dtype=complex)
    f1[1] = 1.0

    # Pure path: homodyne_window=None, homodyne_x=None -> returns reduced density (mixed)
    # But we want to test the pure fast path logic inside compose_pair_cached if we ask for homodyne point
    # Actually, let's test compose_pair directly.

    # Case 1: Pure input, no homodyne -> reduced density
    rho_pure_path, _, _ = composer.compose_pair(f1, f1, theta=math.pi / 4, phi=0.0)

    # Case 2: Mixed input (densities), no homodyne -> reduced density
    rho1 = np.outer(f1, f1.conj())
    rho_mixed_path, _, _ = composer.compose_pair(rho1, rho1, theta=math.pi / 4, phi=0.0)

    assert np.allclose(rho_pure_path, rho_mixed_path), (
        "Pure and mixed paths should yield same reduced density"
    )

    # Check HOM dip: probability of |1,1> in full state should be 0.
    # The reduced density on mode 0 should have p(1) = 0.5 (since |2,0> and |0,2> are output).
    # Wait, |1,1> -> (|2,0> - |0,2>)/sqrt(2).
    # Reduced rho on mode 0: 0.5 |0><0| + 0.5 |2><2|.
    # p(1) should be 0.

    p1 = np.real(rho_pure_path[1, 1])
    assert np.isclose(p1, 0.0), f"HOM dip failed: p(1)={p1} should be 0"

    p0 = np.real(rho_pure_path[0, 0])
    p2 = np.real(rho_pure_path[2, 2])
    assert np.isclose(p0, 0.5)
    assert np.isclose(p2, 0.5)


def test_pure_mode_enforcement():
    """Test that SuperblockTopology enforces pure mode correctly."""
    cutoff = 5
    composer = Composer(cutoff=cutoff)

    # Simple topology: pair of leaves
    topo = SuperblockTopology.build_layered(2)

    f1 = np.zeros(cutoff, dtype=complex)
    f1[0] = 1.0  # |0>
    fock_vecs = [f1, f1]
    p_heralds = [1.0, 1.0]

    # 1. Pure mode success (no window)
    state, prob = topo.evaluate_topology(
        composer,
        fock_vecs,
        p_heralds,
        homodyne_x=0.0,
        homodyne_window=None,
        pure_only=True,
    )
    assert state.ndim == 1, "Should return pure vector"

    # 2. Pure mode failure (window requested)
    with pytest.raises(ValueError, match="homodyne window requested"):
        topo.evaluate_topology(
            composer,
            fock_vecs,
            p_heralds,
            homodyne_x=0.0,
            homodyne_window=0.1,
            pure_only=True,
        )

    # 3. Pure mode failure (mixed output from composer due to no measurement?)
    # If we don't measure (homodyne_x=None), compose_pair returns reduced density.
    # This should trigger "failed to maintain purity".
    with pytest.raises(ValueError, match="failed to maintain purity"):
        topo.evaluate_topology(
            composer,
            fock_vecs,
            p_heralds,
            homodyne_x=None,
            homodyne_window=None,
            pure_only=True,
        )


def test_homodyne_resolution():
    """Test that homodyne_resolution scales the probability correctly."""
    cutoff = 10
    composer = Composer(cutoff=cutoff)
    f0 = np.zeros(cutoff, dtype=complex)
    f0[0] = 1.0  # |0>

    # Vacuum + Vacuum -> Vacuum. Homodyne x=0 on mode 2.
    # P(x) for vacuum is Gaussian. At x=0, p(x) = 1/sqrt(pi*hbar) * exp(0) ?
    # With hbar=2, p(x) = 1/sqrt(2pi) * exp(-x^2/2). At x=0, p(0) = 1/sqrt(2pi) ~ 0.3989

    # 1. Without resolution (density)
    _, p_dens, _ = composer.compose_pair(
        f0, f0, homodyne_x=0.0, homodyne_resolution=None
    )
    expected_dens = 1.0 / np.sqrt(2 * np.pi)
    assert np.isclose(p_dens, expected_dens, atol=1e-4)

    # 2. With resolution
    res = 0.01
    _, p_prob, _ = composer.compose_pair(
        f0, f0, homodyne_x=0.0, homodyne_resolution=res
    )
    assert np.isclose(p_prob, p_dens * res)
