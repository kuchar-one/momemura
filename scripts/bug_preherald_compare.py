"""Compare path-3's pre-herald 3-mode state (moment-space BS+homodyne over a
4-mode [sig0,ctrl0,sig1,ctrl1] Gaussian) against a direct Fock computation of
the same operations. Isolates whether moment-space BS+homodyne on a signal mode
that is entangled with a spectator control mode (with displacement) is correct.
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from thewalrus.quantum import state_vector
from frontend.independent_verifier import _build_gaussian_moments, _fock_bs_unitary, _hermite_phi
from frontend.gaussian_decomposition import get_bs_symplectic, measure_homodyne

HBAR = 2.0
CUT = int(os.environ.get("PRE_CUT", "18"))


def assemble_4mode(muL, VL):
    """Assemble two 2-mode leaves into a 4-mode xp covariance ordered
    [sig0, ctrl0, sig1, ctrl1] exactly like compute_equivalent_gaussian."""
    N = 4
    V = np.zeros((2 * N, 2 * N)); mu = np.zeros(2 * N)
    off = 0
    for (m, c) in zip(muL, VL):
        nl = 2
        for xi in range(nl):
            for yi in range(nl):
                V[off + xi, off + yi] = c[xi, yi]
                V[off + xi, N + off + yi] = c[xi, nl + yi]
                V[N + off + xi, off + yi] = c[nl + xi, yi]
                V[N + off + xi, N + off + yi] = c[nl + xi, nl + yi]
        mu[off:off + nl] = m[:nl]
        mu[N + off:N + off + nl] = m[nl:]
        off += nl
    return mu, V


def fock_preherald(muL, VL, theta, phi, x):
    """Direct Fock: 4-mode state -> BS on (sig0=mode0, sig1=mode2) -> homodyne
    mode2 -> 3-mode tensor [sig0, ctrl0, ctrl1]."""
    mu, V = assemble_4mode(muL, VL)
    T = np.asarray(state_vector(mu, V, cutoff=CUT, hbar=HBAR, normalize=False,
                                check_purity=False))  # [s0,c0,s1,c1]
    U = _fock_bs_unitary(theta, phi, CUT)  # acts on (mode_i, mode_j) pair basis
    # apply BS to axes (0,2): reshape pair (s0,s1) -> apply U -> back
    # move axes 0,2 to front
    Tm = np.moveaxis(T, [0, 2], [0, 1])           # [s0,s1,c0,c1]
    sh = Tm.shape
    Tm = Tm.reshape(CUT * CUT, sh[2], sh[3])
    Tm = np.tensordot(U, Tm, axes=([1], [0]))      # apply U on flattened (s0,s1)
    Tm = Tm.reshape(sh)                            # [s0,s1,c0,c1]
    # homodyne mode s1 (axis 1) onto |x>
    phivec = _hermite_phi(x, CUT)
    T3 = np.tensordot(Tm, phivec, axes=([1], [0]))  # [s0,c0,c1]
    return T3 / (np.linalg.norm(T3) + 1e-300)


def symp_preherald(muL, VL, theta, phi, x):
    mu, V = assemble_4mode(muL, VL)
    S = get_bs_symplectic(theta, phi, 4, 0, 2)
    V = S @ V @ S.T; mu = S @ mu
    Vn, mun = measure_homodyne(V, mu, 2, 4, x)     # drop mode 2 (sig1)
    T3 = np.asarray(state_vector(mun, Vn, cutoff=CUT, hbar=HBAR, normalize=False,
                                 check_purity=False))  # [s0, c0, c1]
    return T3 / (np.linalg.norm(T3) + 1e-300)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for tag, (r0, r1, d0, d1) in {
        "weak  nodisp": ((0.4, 0.3), (0.5, 0.2), (0j, 0j), (0j, 0j)),
        "weak  +disp ": ((0.4, 0.3), (0.5, 0.2), (1.0 + 0.3j, 0.2j), (-0.5 + 0.4j, 0.1j)),
        "strong nodisp": ((1.5, 1.2), (1.4, 1.0), (0j, 0j), (0j, 0j)),
        "strong +disp": ((1.5, 1.2), (1.4, 1.0), (1.2 - 0.5j, 0.3j), (0.9 + 0.6j, -0.2j)),
    }.items():
        ph0 = rng.uniform(0, 2 * np.pi, 4); ph1 = rng.uniform(0, 2 * np.pi, 4)
        m0, V0 = _build_gaussian_moments(np.array(r0), ph0, np.array(d0), 2)
        m1, V1 = _build_gaussian_moments(np.array(r1), ph1, np.array(d1), 2)
        theta, phi, x = np.pi / 4, 0.0, 0.7
        Tf = fock_preherald([m0, m1], [V0, V1], theta, phi, x)
        Ts = symp_preherald([m0, m1], [V0, V1], theta, phi, x)
        F = abs(np.vdot(Tf.ravel(), Ts.ravel())) ** 2
        print(f"{tag}: F(fock_preherald, symp_preherald) = {F:.5f}")
