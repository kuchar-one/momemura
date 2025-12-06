import pytest
import numpy as np
import math
from src.simulation.cpu.circuit import GaussianHeraldCircuit
from src.simulation.cpu.composer import Composer, SuperblockTopology
from src.simulation.jax.runner import jax_get_heralded_state

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    jax.config.update("jax_enable_x64", True)
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_gaussian_herald_circuit_parity():
    """Verify GaussianHeraldCircuit produces identical results on CPU and GPU."""
    n_signal = 1
    n_control = 2
    tmss_r = [0.5]
    # M_us = 1 -> theta/phi empty, varphi len 1
    us_params = {"theta": [], "phi": [], "varphi": [0.3]}
    # M_uc = 2 -> theta/phi len 1, varphi len 2
    uc_params = {"theta": [0.4], "phi": [0.5], "varphi": [0.6, 0.7]}
    pnr = [1, 0]
    cutoff = 10

    # CPU
    circ_cpu = GaussianHeraldCircuit(
        n_signal,
        n_control,
        tmss_r,
        us_params,
        uc_params,
        backend="thewalrus",
        cache_enabled=False,
    )
    circ_cpu.build()
    state_cpu, prob_cpu = circ_cpu.herald(pnr, signal_cutoff=cutoff)

    # GPU
    circ_gpu = GaussianHeraldCircuit(
        n_signal,
        n_control,
        tmss_r,
        us_params,
        uc_params,
        backend="jax",
        cache_enabled=False,
    )
    circ_gpu.build()
    state_gpu, prob_gpu = circ_gpu.herald(pnr, signal_cutoff=cutoff)

    # Compare
    assert np.isclose(prob_cpu, prob_gpu, atol=2e-4)

    # state_gpu might be JAX array
    state_gpu_np = np.array(state_gpu)

    # Check shape
    assert state_cpu.shape == state_gpu_np.shape

    # Check values
    # Note: Phase might differ by global phase?
    # But for pure state amplitude calculation from same Q, it should be identical.
    np.testing.assert_allclose(state_cpu, state_gpu_np, atol=1e-5)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_composer_parity_pure():
    """Verify Composer pure-state path parity."""
    cutoff = 10
    comp_cpu = Composer(cutoff=cutoff, backend="thewalrus")
    comp_gpu = Composer(cutoff=cutoff, backend="jax")

    # Random pure states
    np.random.seed(42)
    vecA = np.random.randn(cutoff) + 1j * np.random.randn(cutoff)
    vecA /= np.linalg.norm(vecA)
    vecB = np.random.randn(cutoff) + 1j * np.random.randn(cutoff)
    vecB /= np.linalg.norm(vecB)

    theta = 0.5
    phi = 0.2
    hom_x = 0.1

    # CPU
    out_cpu, p_cpu, joint_cpu = comp_cpu.compose_pair(
        vecA, vecB, homodyne_x=hom_x, theta=theta, phi=phi
    )

    # GPU
    # Pass numpy arrays, Composer should handle conversion
    out_gpu, p_gpu, joint_gpu = comp_gpu.compose_pair(
        vecA, vecB, homodyne_x=hom_x, theta=theta, phi=phi
    )

    # Compare
    assert np.isclose(p_cpu, p_gpu, atol=1e-4)
    assert np.isclose(joint_cpu, joint_gpu, atol=1e-4)

    out_gpu_np = np.array(out_gpu)
    np.testing.assert_allclose(out_cpu, out_gpu_np, atol=1e-4)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_composer_parity_mixed():
    """Verify Composer mixed-state path (homodyne window) parity."""
    cutoff = 6
    comp_cpu = Composer(cutoff=cutoff, backend="thewalrus")
    comp_gpu = Composer(cutoff=cutoff, backend="jax")

    # Random pure states (will become mixed after window)
    np.random.seed(43)
    vecA = np.random.randn(cutoff) + 1j * np.random.randn(cutoff)
    vecA /= np.linalg.norm(vecA)
    vecB = np.random.randn(cutoff) + 1j * np.random.randn(cutoff)
    vecB /= np.linalg.norm(vecB)

    theta = math.pi / 4
    phi = 0.0
    hom_x = 0.0
    hom_window = 0.5

    # CPU
    out_cpu, p_cpu, joint_cpu = comp_cpu.compose_pair(
        vecA, vecB, homodyne_x=hom_x, homodyne_window=hom_window, theta=theta, phi=phi
    )

    # GPU
    out_gpu, p_gpu, joint_gpu = comp_gpu.compose_pair(
        vecA, vecB, homodyne_x=hom_x, homodyne_window=hom_window, theta=theta, phi=phi
    )

    # Compare
    assert np.isclose(p_cpu, p_gpu, atol=1e-5)
    assert np.isclose(joint_cpu, joint_gpu, atol=1e-5)

    out_gpu_np = np.array(out_gpu)
    # Density matrices
    np.testing.assert_allclose(out_cpu, out_gpu_np, atol=2e-4)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_topology_parity():
    """Verify full topology evaluation parity."""
    cutoff = 6
    comp_cpu = Composer(cutoff=cutoff, backend="thewalrus")
    comp_gpu = Composer(cutoff=cutoff, backend="jax")

    topo = SuperblockTopology.build_layered(2)  # 2 leaves -> 1 pair

    vec = np.zeros(cutoff, dtype=complex)
    vec[1] = 1.0  # |1>

    fock_vecs = [vec, vec]
    p_heralds = [1.0, 1.0]

    # CPU
    state_cpu, prob_cpu = topo.evaluate_topology(
        comp_cpu, fock_vecs, p_heralds, homodyne_x=0.0, theta=math.pi / 4
    )

    # GPU
    state_gpu, prob_gpu = topo.evaluate_topology(
        comp_gpu, fock_vecs, p_heralds, homodyne_x=0.0, theta=math.pi / 4
    )

    assert np.isclose(prob_cpu, prob_gpu, atol=1e-5)
    np.testing.assert_allclose(state_cpu, np.array(state_gpu), atol=1e-5)
