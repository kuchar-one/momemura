#!/usr/bin/env python3
"""Unit-test the Gaussian vacuum-projection formulas in the thewalrus hbar=2
convention (vacuum cov = I, xp-ordered (x1..xN,p1..pN)):

  P0        = 2^{nB} / sqrt(det(S_BB + I)) * exp(-mu_B^T (S_BB+I)^{-1} mu_B ... )
  Sigma'    = S_AA - S_AB (S_BB + I)^{-1} S_BA
  mu'       = mu_A - S_AB (S_BB + I)^{-1} mu_B

against brute-force Fock-space computation on random 2-mode pure Gaussian
states, using thewalrus.quantum.state_vector for the Fock amplitudes.
"""
import numpy as np
import thewalrus.symplectic as symp
from thewalrus.quantum import state_vector

rng = np.random.default_rng(7)

def random_two_mode(hbar=2):
    mu = np.zeros(4); cov = np.eye(4) * hbar / 2
    # squeeze both modes, random interferometer, displacements
    for m, r in enumerate(rng.uniform(0.2, 0.9, 2)):
        S = symp.expand(symp.squeezing(r, rng.uniform(0, 2*np.pi)), m, 2)
        cov = S @ cov @ S.T; mu = S @ mu
    th = rng.uniform(0, 2*np.pi); ph = rng.uniform(0, 2*np.pi)
    BS = symp.expand(symp.beam_splitter(th, ph), [0, 1], 2)
    cov = BS @ cov @ BS.T; mu = BS @ mu
    mu = mu + rng.uniform(-1.2, 1.2, 4)
    return mu, cov

def fock_check(mu, cov, cut=30):
    psi = state_vector(mu, cov, cutoff=cut)          # (cut, cut) two-mode
    # project mode 1 (second) onto <0|
    amp0 = psi[:, 0]                                  # unnormalized 1-mode state
    P0_fock = float(np.sum(np.abs(amp0)**2))
    psiA = amp0 / np.sqrt(P0_fock)
    return P0_fock, psiA

def gaussian_project(mu, cov, B=(1,), N=2):
    A = [i for i in range(N) if i not in B]
    iA = A + [a + N for a in A]; iB = list(B) + [b + N for b in B]
    S_AA = cov[np.ix_(iA, iA)]; S_AB = cov[np.ix_(iA, iB)]
    S_BB = cov[np.ix_(iB, iB)]; muA = mu[iA]; muB = mu[iB]
    M = S_BB + np.eye(len(iB))
    Minv = np.linalg.inv(M)
    nB = len(B)
    P0 = 2.0**nB / np.sqrt(np.linalg.det(M)) * np.exp(-muB @ Minv @ muB / 2.0)
    # NOTE the 1/2 in the exponent is a guess to be tested; try both
    P0_alt = 2.0**nB / np.sqrt(np.linalg.det(M)) * np.exp(-muB @ Minv @ muB)
    covp = S_AA - S_AB @ Minv @ S_AB.T
    mup = muA - S_AB @ Minv @ muB
    return P0, P0_alt, covp, mup

ok = True
for trial in range(5):
    mu, cov = random_two_mode()
    P0_fock, psiA_fock = fock_check(mu, cov, cut=36)
    P0, P0_alt, covp, mup = gaussian_project(mu, cov)
    psiA_gauss = state_vector(mup, covp, cutoff=36)
    f = abs(np.vdot(psiA_fock, psiA_gauss / np.linalg.norm(psiA_gauss)))**2
    d1, d2 = abs(P0 - P0_fock), abs(P0_alt - P0_fock)
    print(f"trial {trial}: P0_fock={P0_fock:.6f}  half-exp={P0:.6f} (d={d1:.1e})  "
          f"full-exp={P0_alt:.6f} (d={d2:.1e})  state fid={f:.10f}")
    ok &= (min(d1, d2) < 1e-6 * max(P0_fock, 1e-3) + 1e-10 and f > 1 - 1e-9)
print("ALL OK" if ok else "FAILURE")
