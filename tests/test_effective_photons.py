"""Regression test: the effective-photon (dud-detection) guard.

A PNR detection on a control mode whose leaf interferometer decodes to
~identity (Clements theta ~ pi: cos = -1, sin ~ 0) is physically inert and
must not count toward the photons descriptor / artifact guard.
See HANAMURA_VALIDATION_FINDINGS.md (the plus_0 exploit).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from src.simulation.jax.runner import _leaf_effective_pnr


def _leaf(theta, pnr=2, n_ctrl=1):
    """2-mode leaf: phases = [theta, phi, varphi1, varphi2] (Clements N=2)."""
    return {
        "r": jnp.array([1.0, 0.6, 0.0]),
        "phases": jnp.array([theta, 0.3, 0.1, 0.2, 0, 0, 0, 0, 0], dtype=float),
        "pnr": jnp.array([pnr, 0]),
        "n_ctrl": jnp.array(n_ctrl),
        "disp": jnp.zeros(3, dtype=complex),
    }


def test_decoupled_detection_is_dud():
    """theta = pi -> BS is (-1)*identity -> control decoupled -> eff ~ 0."""
    tot, mx = _leaf_effective_pnr(_leaf(np.pi, pnr=2), eps=0.05)
    assert float(tot) < 0.05
    assert float(mx) < 0.05


def test_coupled_detection_counts_fully():
    """theta = pi/4 -> balanced BS -> strongly coupled -> eff ~ pnr."""
    tot, mx = _leaf_effective_pnr(_leaf(np.pi / 4, pnr=2), eps=0.05)
    assert float(tot) == pytest.approx(2.0, abs=0.05)
    assert float(mx) == pytest.approx(2.0, abs=0.05)


def test_no_controls_no_photons():
    tot, mx = _leaf_effective_pnr(_leaf(np.pi / 4, pnr=3, n_ctrl=0), eps=0.05)
    assert float(tot) == 0.0 and float(mx) == 0.0
