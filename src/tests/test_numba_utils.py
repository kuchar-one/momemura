
import numpy as np
import sys
import os

# Ensure src is in path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.params import passive_unitary_to_symplectic, xp_to_interleaved, interleaved_to_xp

def test_passive_unitary_to_symplectic():
    print("Testing passive_unitary_to_symplectic...")
    U = np.eye(1, dtype=np.complex128)
    S_xp = passive_unitary_to_symplectic(U)
    print("U:\n", U)
    print("S_xp:\n", S_xp)
    expected_S_xp = np.eye(2, dtype=np.float64)
    assert np.allclose(S_xp, expected_S_xp), "S_xp is not identity"
    print("PASS: S_xp is identity")

def test_xp_to_interleaved():
    print("\nTesting xp_to_interleaved...")
    expected_S_xp = np.eye(2, dtype=np.float64)
    S_ip = xp_to_interleaved(expected_S_xp)
    print("S_ip:\n", S_ip)
    assert np.allclose(S_ip, expected_S_xp), "S_ip is not identity"
    print("PASS: S_ip is identity")

def test_interleaved_to_xp():
    print("\nTesting interleaved_to_xp...")
    mu_ip = np.array([1.0, 2.0, 3.0, 4.0]) # q0, p0, q1, p1
    # xp: q0, q1, p0, p1 -> 1, 3, 2, 4
    mu_xp, _ = interleaved_to_xp(mu_ip, np.eye(4))
    print("mu_ip:", mu_ip)
    print("mu_xp shape:", mu_xp.shape)
    print("mu_xp:", mu_xp)
    expected_mu_xp = np.array([1.0, 3.0, 2.0, 4.0])
    assert np.allclose(mu_xp, expected_mu_xp), "mu_xp incorrect"
    print("PASS: mu_xp correct")

if __name__ == "__main__":
    test_passive_unitary_to_symplectic()
    test_xp_to_interleaved()
    test_interleaved_to_xp()
    print("ALL NUMBA UTILS TESTS PASSED")
