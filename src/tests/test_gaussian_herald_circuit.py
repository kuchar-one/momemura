"""
Tests for gaussian_herald_circuit.py

These are adapted from your original tests: vacuum propagation, single-photon herald,
product TMSS heralding, and HOM interference. Run as a script or with pytest.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from circuits.gaussian_herald_circuit import (
    GaussianHeraldCircuit,
    interferometer_params_to_unitary,
)


def test_vacuum_propagation():
    circ = GaussianHeraldCircuit(n_signal=1, n_control=1, tmss_squeezing=[0.0])
    circ.build()
    state, prob = circ.herald([0], signal_cutoff=5)
    print("vacuum prob:", prob)
    assert np.isclose(prob, 1.0, atol=1e-8)
    assert np.isclose(np.abs(state[0]), 1.0, atol=1e-8)
    print("test_vacuum_propagation OK")


def test_single_photon_heralding():
    z = 0.8
    th = np.tanh(z)
    ch = np.cosh(z)
    expected_prob = (th**2) / (ch**2)

    circ = GaussianHeraldCircuit(n_signal=1, n_control=1, tmss_squeezing=[z])
    circ.build()
    state, prob = circ.herald([1], signal_cutoff=6)
    print("single-photon prob:", prob, "expected:", expected_prob)
    assert np.isclose(prob, expected_prob, atol=1e-6)
    assert np.isclose(np.abs(state[1]), 1.0, atol=1e-6)
    print("test_single_photon_heralding OK")


def test_independent_product_states():
    z = 0.5
    circ = GaussianHeraldCircuit(n_signal=2, n_control=2, tmss_squeezing=[z, z])
    circ.build()
    state, prob = circ.herald([1, 2], signal_cutoff=6)
    assert np.isclose(np.abs(state[1, 2]), 1.0, atol=1e-6)
    print("test_independent_product_states OK")


def test_hom_bunching():
    z = 0.8
    # Build 50:50 beamsplitter unitary on two signal modes using Clements param builder
    ths = np.array([np.pi / 4], dtype=np.float64)  # one beamsplitter for M=2
    phis = np.array([0.0], dtype=np.float64)
    varphi = np.array([0.0, 0.0], dtype=np.float64)
    Ubs = interferometer_params_to_unitary(ths, phis, varphi, M=2, mesh="rectangular")
    circ = GaussianHeraldCircuit(
        n_signal=2, n_control=2, tmss_squeezing=[z, z], U_s=Ubs, U_c=np.eye(2)
    )
    circ.build()
    st, pr = circ.herald([1, 1], signal_cutoff=6)
    amp_11 = np.abs(st[1, 1])
    amp_20 = np.abs(st[2, 0])
    amp_02 = np.abs(st[0, 2])
    print("HOM amps:", amp_11, amp_20, amp_02)
    assert amp_11 < 1e-5
    assert amp_20 > 1e-2
    assert amp_02 > 1e-2
    print("test_hom_bunching OK")


if __name__ == "__main__":
    test_vacuum_propagation()
    test_single_photon_heralding()
    test_independent_product_states()
    test_hom_bunching()
    print("ALL TESTS PASSED")
