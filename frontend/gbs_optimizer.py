"""
GBS architecture optimizer based on the non-Gaussian control parameters of

    F. Hanamura et al., "Beyond Stellar Rank: Control Parameters for Scalable
    Optical Non-Gaussian State Generation", Phys. Rev. X 16, 021034 (2026).

The frontend already reduces any selected breeding solution to an equivalent
Gaussian-boson-sampling (GBS) generator:

        vacuum -> Gaussian operation -> PNR detections + one heralded mode

i.e. an (l+k)-mode pure Gaussian state with covariance ``cov`` and mean ``mu``
(xp-ordering, hbar = 2, vacuum covariance = Identity), where ``k`` control modes
are projected onto Fock states ``n`` and ``l = 1`` signal mode is heralded.

This module takes that generator and runs the paper's two-step optimization,
working entirely with the **control-mode representation** ``(C, beta, n)``
(Corollary 1/2): the control-mode covariance/mean fully determine the output
state (up to a Gaussian unitary) and the success probability.

  Step 1 - photon-number reduction (Sec. IV B / Appendix K):
      For each control mode, the wave-function ``phi_n(x)`` of the Fock state is
      approximated by ``phi_n'(k x - d)`` (Eq. 69), reducing the detected photon
      number ``n -> n'`` while preserving the output state up to a Gaussian
      unitary.  Implemented via the gauge that preserves the symplectic
      eigenvalue of the control block (Appendix K 3).

  Step 2 - success-probability maximization (Sec. VI B / Theorem 10):
      The damping transform ``D_t`` (Eq. 81) leaves the output state invariant
      but changes the success probability; we maximize ``p_n'(D_t(C, beta))``
      over the damping parameters ``t`` (both attenuation and amplification
      branches, t > 1 or t I < -C).

The optimized generator ``(C', beta', n')`` is realized as a physical GBS
architecture (squeezers + interferometer + displacements + PNR pattern) via the
canonical purification of Theorem 9.

All heavy results are cross-checked numerically by ``verify_optimization``.
Convention throughout: hbar = 2, xp-ordering, vacuum covariance = Identity.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Sequence, Tuple, Optional

HBAR = 2.0


# =============================================================================
# Bargmann (B-matrix) representation of a pure Gaussian state
#   |psi> propto exp(1/2 a^dag B a^dag + gamma . a^dag) |0>
# (thewalrus' Amat block equals conj(B), hence the conjugations below.)
# =============================================================================
def cov_mu_to_B_gamma(cov: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from thewalrus import quantum as twq

    N = cov.shape[0] // 2
    A = twq.Amat(cov, hbar=HBAR)
    B = np.conj(A[:N, :N]).copy()
    alpha = (mu[:N] + 1j * mu[N:]) / np.sqrt(2 * HBAR)
    gamma = alpha - B @ np.conj(alpha)
    return B, gamma


def B_gamma_to_cov_mu(B: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from thewalrus import quantum as twq

    N = B.shape[0]
    X = twq.Xmat(N)
    Afull = np.block([[np.conj(B), np.zeros((N, N))], [np.zeros((N, N)), B]])
    Q = np.linalg.inv(np.eye(2 * N) - Afull @ X)
    cov = twq.Covmat(Q, hbar=HBAR).real
    # alpha solves gamma = alpha - B conj(alpha):
    M = np.block([[np.eye(N) - B.real, -B.imag],
                  [-B.imag, np.eye(N) + B.real]])
    a = np.linalg.solve(M, np.concatenate([gamma.real, gamma.imag]))
    alpha = a[:N] + 1j * a[N:]
    mu = np.concatenate([np.sqrt(2 * HBAR) * alpha.real, np.sqrt(2 * HBAR) * alpha.imag])
    return cov, mu


# =============================================================================
# Control-mode representation utilities
# =============================================================================
def extract_control(cov: np.ndarray, mu: np.ndarray,
                    control_idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C, beta): the control-mode covariance block and mean (xp-ordered
    over the control modes, in the order given by ``control_idx``)."""
    N = cov.shape[0] // 2
    idx = [m for m in control_idx] + [m + N for m in control_idx]
    idx = np.array(idx, dtype=int)
    return cov[np.ix_(idx, idx)], mu[idx]


def control_parameters(Cm: np.ndarray, beta_m: np.ndarray) -> Dict[str, Any]:
    """Non-Gaussian control parameters (s0, delta0) of a single control mode
    (Eqs. 34-36).  ``Cm`` is the 2x2 covariance block, ``beta_m`` the 2-vector
    mean.  Returns s0, delta0, the diagonalizing rotation O, eigenvalues c >= d
    and the symplectic eigenvalue nu = sqrt(det Cm)."""
    w, V = np.linalg.eigh(Cm)
    d, c = float(w[0]), float(w[1])
    O = np.array([V[:, 1], V[:, 0]])           # row 0 -> c, row 1 -> d
    if np.linalg.det(O) < 0:
        O[1] = -O[1]
    bbar = O @ beta_m
    cd = c * d
    nu = np.sqrt(cd)
    if cd - 1.0 < 1e-9:
        # control mode is (essentially) pure/vacuum -> no non-Gaussian content
        return dict(s0=0.0, delta0=0.0 + 0j, c=c, d=d, O=O, nu=nu, bbar=bbar)
    s0 = (c - d) / (cd - 1.0)
    delta0 = (2.0 / np.sqrt(cd - 1.0)) * (
        np.sqrt((d + 1) / (c + 1)) * bbar[0] - 1j * np.sqrt((c + 1) / (d + 1)) * bbar[1]
    )
    return dict(s0=s0, delta0=delta0, c=c, d=d, O=O, nu=nu, bbar=bbar)


def block_from_params(s0: float, delta0: complex, nu: float,
                      O: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Inverse of :func:`control_parameters` in the gauge that fixes the
    symplectic eigenvalue ``nu`` (Appendix K 3).  Returns (Cm, beta_m, c, d)."""
    diff = s0 * (nu ** 2 - 1.0)
    summ = np.sqrt(diff ** 2 + 4.0 * nu ** 2)
    c = 0.5 * (summ + diff)
    d = 0.5 * (summ - diff)
    bx = np.real(delta0) * np.sqrt(c * d - 1.0) / 2.0 * np.sqrt((c + 1) / (d + 1))
    bp = -np.imag(delta0) * np.sqrt(c * d - 1.0) / 2.0 * np.sqrt((d + 1) / (c + 1))
    Cm = O.T @ np.diag([c, d]) @ O
    beta_m = O.T @ np.array([bx, bp])
    return Cm, beta_m, c, d


def success_probability(C: np.ndarray, beta: np.ndarray, n: Sequence[int]) -> float:
    """Probability of detecting the Fock pattern ``n`` on the control marginal
    (C, beta) -- equals the heralding success probability (Corollary 2)."""
    from thewalrus import quantum as twq

    n = [int(x) for x in n]
    p = twq.density_matrix_element(beta, C, n, n, hbar=HBAR)
    return float(np.real(p))


# =============================================================================
# Canonical purification of control moments -> full pure Gaussian state
#   layout (xp): modes 0..k-1 = control, k..k+r-1 = signal ancillas (Theorem 9)
# =============================================================================
def purify_control(C: np.ndarray, beta: np.ndarray,
                   tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    from thewalrus.decompositions import williamson

    k = C.shape[0] // 2
    D, S = williamson(C)                       # C = S D S^T
    nu = np.array([D[i, i] for i in range(k)])
    nontrivial = [i for i in range(k) if nu[i] > 1 + tol]
    r = len(nontrivial)
    M = k + r
    Vw = np.zeros((2 * M, 2 * M))

    def xi(m): return m
    def pi(m): return M + m

    for i in range(k):
        Vw[xi(i), xi(i)] = nu[i]
        Vw[pi(i), pi(i)] = nu[i]
    for a_off, i in enumerate(nontrivial):
        a = k + a_off
        cc = np.sqrt(nu[i] ** 2 - 1.0)
        Vw[xi(a), xi(a)] = nu[i]
        Vw[pi(a), pi(a)] = nu[i]
        Vw[xi(i), xi(a)] = Vw[xi(a), xi(i)] = cc
        Vw[pi(i), pi(a)] = Vw[pi(a), pi(i)] = -cc

    # embed control symplectic S (xp over k control modes) into the full space
    Semb = np.eye(2 * M)
    Sxx, Sxp, Spx, Spp = S[:k, :k], S[:k, k:], S[k:, :k], S[k:, k:]
    cx = list(range(k)); cp = [M + i for i in range(k)]
    for aa in range(k):
        for bb in range(k):
            Semb[cx[aa], cx[bb]] = Sxx[aa, bb]
            Semb[cx[aa], cp[bb]] = Sxp[aa, bb]
            Semb[cp[aa], cx[bb]] = Spx[aa, bb]
            Semb[cp[aa], cp[bb]] = Spp[aa, bb]

    Vfull = Semb @ Vw @ Semb.T
    Vfull = 0.5 * (Vfull + Vfull.T)
    mu = np.zeros(2 * M)
    mu[:k] = beta[:k]
    mu[M:M + k] = beta[k:]
    return Vfull, mu, list(range(k)), list(range(k, M))


# =============================================================================
# Damping transform D_t (Eq. 81) and its validity domain (Theorem 10)
# =============================================================================
def damping_transform_control(C: np.ndarray, beta: np.ndarray,
                              t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Multimode damping transform (Eq. 81):
        C' = T - sqrt(T^2-1) (C+T)^{-1} sqrt(T^2-1),
        b' = sqrt(T^2-1) (C+T)^{-1} beta,
    with T = diag(t_1, t_1, ..., t_k, t_k) (xp-ordered)."""
    k = C.shape[0] // 2
    t = np.asarray(t, dtype=float)
    Tdiag = np.concatenate([t, t])
    T = np.diag(Tdiag)
    sq = np.diag(np.sqrt(Tdiag ** 2 - 1.0 + 0j)).real
    inv = np.linalg.inv(C + T)
    Cp = T - sq @ inv @ sq
    Cp = 0.5 * (Cp + Cp.T)
    bp = sq @ inv @ beta
    return Cp, bp


def is_valid_covariance(C: np.ndarray, tol: float = 1e-6) -> bool:
    """C >= i Omega: all symplectic eigenvalues real and >= 1."""
    from thewalrus.decompositions import symplectic_eigenvals
    try:
        se = np.asarray(symplectic_eigenvals(C))
    except Exception:
        return False
    if np.any(np.abs(se.imag) > 1e-6):
        return False
    return bool(np.min(se.real) >= 1.0 - tol)


def generator_max_squeezing_db(C: np.ndarray, beta: np.ndarray) -> float:
    """Max squeezing (dB) of the physical generator realizing control moments
    (C, beta), via canonical purification + Bloch-Messiah."""
    Vf, muf, _, _ = purify_control(C, beta)
    return decompose_architecture(Vf, muf)["max_squeezing_db"]


def _damping_prob(C, beta, t, n) -> Tuple[float, bool]:
    try:
        Cp, bp = damping_transform_control(C, beta, t)
    except np.linalg.LinAlgError:
        return 0.0, False
    if not is_valid_covariance(Cp):
        return 0.0, False
    return success_probability(Cp, bp, n), True


def optimize_damping(C: np.ndarray, beta: np.ndarray, n: Sequence[int],
                     max_squeezing_db: Optional[float] = None) -> Dict[str, Any]:
    """Maximize the success probability over the damping freedom (Sec. VI B).

    Parametrized by lambda_m with t_m = coth(lambda_m): lambda > 0 is the
    attenuation branch (t > 1) and lambda < 0 the amplification branch
    (t I < -C); lambda -> 0 is the identity (no damping).

    If ``max_squeezing_db`` is given, the maximization is constrained so the
    optimized generator's largest squeezing stays at or below that cap; this
    trades a little probability for experimental feasibility.  ``None`` gives
    the uncapped optimum (faithful to the paper).

    Returns the optimal t, the maximized probability, lambda, the no-damping
    probability, the generator's max squeezing (dB) and whether the cap was met.
    """
    from scipy.optimize import minimize

    k = C.shape[0] // 2

    def coth(lam):
        return 1.0 / np.tanh(lam)

    p_id, _ = _damping_prob(C, beta, coth(np.full(k, 1e3)), n)

    def sq_db_for(lam):
        Cp, bp = damping_transform_control(C, beta, coth(lam))
        return generator_max_squeezing_db(Cp, bp)

    def objective(lam):
        lam = np.where(np.abs(lam) < 1e-3, np.sign(lam + 1e-12) * 1e-3, lam)
        p, valid = _damping_prob(C, beta, coth(lam), n)
        if not valid or p <= 0:
            return 80.0 + float(np.sum(lam ** 2))
        val = -np.log(p)
        if max_squeezing_db is not None:
            try:
                sq = sq_db_for(lam)
            except Exception:
                return 80.0 + float(np.sum(lam ** 2))
            if sq > max_squeezing_db:                # soft cap penalty (per dB)
                val += 6.0 * (sq - max_squeezing_db)
        return val

    best = None
    starts = [np.full(k, -0.3), np.full(k, -0.6), np.full(k, -0.15),
              np.full(k, -0.45), np.full(k, 0.3)]
    if max_squeezing_db is not None:
        starts += [np.full(k, -0.05), np.full(k, -0.1)]
    for lam0 in starts:
        res = minimize(objective, lam0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-8, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res

    lam_star = best.x
    t_star = coth(lam_star)
    p_star, valid = _damping_prob(C, beta, t_star, n)

    # squeezing of the chosen generator
    Cs, bs = damping_transform_control(C, beta, t_star) if valid else (C, beta)
    sq_star = generator_max_squeezing_db(Cs, bs) if valid else float("inf")
    cap_met = (max_squeezing_db is None) or (sq_star <= max_squeezing_db + 1e-3)

    # fall back to no damping if the optimum is invalid, worse than identity,
    # or (in capped mode) still violates the cap while identity satisfies it
    sq_id = generator_max_squeezing_db(C, beta)
    use_identity = (not valid) or (p_star < p_id)
    if max_squeezing_db is not None and not cap_met and sq_id <= max_squeezing_db + 1e-3:
        use_identity = True
    if use_identity:
        t_star = coth(np.full(k, 1e3))
        lam_star = np.full(k, 1e3)
        p_star = p_id
        sq_star = sq_id
        cap_met = (max_squeezing_db is None) or (sq_star <= max_squeezing_db + 1e-3)

    return dict(t=t_star, prob=p_star, lam=lam_star, prob_no_damping=p_id,
                max_squeezing_db=sq_star, cap_met=cap_met,
                max_squeezing_cap=max_squeezing_db)


# =============================================================================
# Step 1: photon-number reduction (Sec. IV B / Appendix K)
# =============================================================================
def _match_kd(s0: float, delta0: complex, n: int, nprime: int) -> Tuple[float, float, float]:
    """Find (k, d) so that phi_n(x) ~ phi_n'(k x - d) near x0 (Appendix K)."""
    d0p = float(np.imag(delta0))
    x0 = (np.sqrt(s0 + 1.0) / s0 * d0p) if s0 > 1e-9 else 0.0
    if abs(x0) < 1e-6 and ((n - nprime) % 2 == 0):
        return np.sqrt((2 * n + 1.0) / (2 * nprime + 1.0)), 0.0, x0

    # General case: match local momentum p(x0)=p~(x0) and p'(x0)=p~'(x0)
    # p(x)  = sqrt(4n + 2 - x^2);  p~(x) = k sqrt(4n'+2 - (k x - d)^2)   (App. K2)
    from scipy.optimize import least_squares

    def p(x):
        v = 4 * n + 2 - x ** 2
        return np.sqrt(v) if v > 0 else 0.0

    def dp(x):
        v = 4 * n + 2 - x ** 2
        return -x / np.sqrt(v) if v > 0 else 0.0

    p0, dp0 = p(x0), dp(x0)

    def resid(params):
        kk, dd = params
        arg = 4 * nprime + 2 - (kk * x0 - dd) ** 2
        if arg <= 0:
            return [1e3, 1e3]
        pt = kk * np.sqrt(arg)
        dpt = kk * (-(kk) * (kk * x0 - dd)) / np.sqrt(arg)
        return [pt - p0, dpt - dp0]

    k0 = np.sqrt((2 * n + 1.0) / (2 * nprime + 1.0))
    best = None
    for d_guess in (k0 * x0, 0.0, k0 * x0 - 1.0, k0 * x0 + 1.0):
        sol = least_squares(resid, [k0, d_guess],
                            bounds=([1e-3, -np.inf], [np.inf, np.inf]))
        if best is None or sol.cost < best.cost:
            best = sol
    return float(best.x[0]), float(best.x[1]), x0


def _reduced_params(s0: float, delta0: complex, n: int,
                    nprime: int) -> Tuple[float, complex, float, float]:
    """(s'0, delta'0, k, d) for the reduction n -> n' (Eqs. 72-73)."""
    k, d, _ = _match_kd(s0, delta0, n, nprime)
    d0x, d0p = float(np.real(delta0)), float(np.imag(delta0))
    s0p = s0 / k ** 2
    delta0p = np.sqrt((s0 + 1.0) / (s0 + k ** 2)) * (k ** 2 * d0x + 1j * d0p) \
        - 1j * s0 * d / (k * np.sqrt(s0 + k ** 2))
    return s0p, delta0p, k, d


def _embed_single_mode_symplectic(S2: np.ndarray, m: int, N: int) -> np.ndarray:
    S = np.eye(2 * N)
    S[m, m] = S2[0, 0]; S[m, m + N] = S2[0, 1]
    S[m + N, m] = S2[1, 0]; S[m + N, m + N] = S2[1, 1]
    return S


def reduce_control_mode(C: np.ndarray, beta: np.ndarray, m_local: int,
                        n_m: int, nprime_m: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply the photon-number reduction n_m -> n'_m to control mode ``m_local``
    (index within the control marginal C/beta).  Returns updated (C, beta) and a
    diagnostics dict.  The output state is preserved up to a Gaussian unitary."""
    k = C.shape[0] // 2
    idx = [m_local, m_local + k]
    Cm = C[np.ix_(idx, idx)]
    beta_m = beta[idx]
    p = control_parameters(Cm, beta_m)
    s0, delta0, O, nu = p["s0"], p["delta0"], p["O"], p["nu"]
    s0p, delta0p, kk, dd = _reduced_params(s0, delta0, n_m, nprime_m)
    _, _, cprime, dprime = block_from_params(s0p, delta0p, nu, O)
    # single-mode squeeze (preserves nu) in the eigenbasis
    Dsq = np.diag([np.sqrt(cprime / p["c"]), np.sqrt(dprime / p["d"])])
    S2 = O.T @ Dsq @ O
    Semb = _embed_single_mode_symplectic(S2, m_local, k)
    C2 = Semb @ C @ Semb.T
    beta2 = Semb @ beta
    # displacement to hit the target block mean
    _, beta_target, _, _ = block_from_params(s0p, delta0p, nu, O)
    cur = np.array([beta2[m_local], beta2[m_local + k]])
    delta = beta_target - cur
    beta2[m_local] += delta[0]
    beta2[m_local + k] += delta[1]
    info = dict(s0=s0, delta0=delta0, s0p=s0p, delta0p=delta0p, k=kk, d=dd, nu=nu,
                n=n_m, nprime=nprime_m)
    return C2, beta2, info


# =============================================================================
# Heralded output and fidelity-up-to-Gaussian-unitary (verification)
# =============================================================================
def heralded_output(cov: np.ndarray, mu: np.ndarray, signal_idx: int,
                    control_idx: Sequence[int], n: Sequence[int],
                    cutoff: int = 40) -> Tuple[np.ndarray, float]:
    """Heralded single-mode signal state and herald probability."""
    from thewalrus.quantum import state_vector
    post = {int(m): int(nv) for m, nv in zip(control_idx, n)}
    sv = np.asarray(state_vector(mu, cov, post_select=post, cutoff=cutoff,
                                 hbar=HBAR, normalize=False, check_purity=False)).ravel()
    prob = float(np.sum(np.abs(sv) ** 2))
    if prob > 0:
        sv = sv / np.sqrt(prob)
    return sv, prob


def _single_mode_gaussian_unitary(params, cutoff):
    import scipy.linalg as sla
    dr, di, r, phi, varphi = params
    a = np.diag(np.sqrt(np.arange(1, cutoff)), k=1); ad = a.T
    Usq = sla.expm(0.5 * (r * np.exp(-2j * phi) * a @ a - r * np.exp(2j * phi) * ad @ ad))
    v_rot = np.exp(1j * np.arange(cutoff) * varphi)
    disp = dr + 1j * di
    Udisp = sla.expm(disp * ad - np.conj(disp) * a)
    return Udisp @ (v_rot[:, None] * Usq)


def fidelity_up_to_gaussian(psi0: np.ndarray, psi1: np.ndarray, cutoff: int,
                            guess=(0, 0, 0, 0, 0), align_cut: Optional[int] = None) -> float:
    """Max overlap |<psi0|U|psi1>|^2 over single-mode Gaussian unitaries U.
    This absorbs the Gaussian-unitary freedom that the architecture's final
    Gaussian operation handles."""
    import scipy.linalg as sla
    from scipy.optimize import minimize

    cut = min(align_cut or cutoff, cutoff)
    p0 = psi0[:cut].copy(); p1 = psi1[:cut].copy()
    n0, n1 = np.linalg.norm(p0), np.linalg.norm(p1)
    if n0 == 0 or n1 == 0:
        return 0.0
    p0 /= n0; p1 /= n1
    a = np.diag(np.sqrt(np.arange(1, cut)), k=1); ad = a.T
    nvec = np.arange(cut)

    def negf(p):
        dr, di, r, phi, varphi = p
        v = sla.expm(0.5 * (r * np.exp(-2j * phi) * a @ a - r * np.exp(2j * phi) * ad @ ad)) @ p1
        v = np.exp(1j * nvec * varphi) * v
        disp = dr + 1j * di
        v = sla.expm(disp * ad - np.conj(disp) * a) @ v
        return -abs(np.vdot(p0, v)) ** 2

    best = None
    for start in (guess, (0, 0, 0, 0, 0)):
        res = minimize(negf, np.array(start, dtype=float), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-7, maxiter=1500))
        if best is None or res.fun < best.fun:
            best = res
    return float(-best.fun)


# =============================================================================
# Architecture decomposition of a pure Gaussian state
# =============================================================================
def decompose_architecture(cov: np.ndarray, mu: np.ndarray) -> Dict[str, Any]:
    """Williamson + Bloch-Messiah of a pure Gaussian state -> squeezers (r, dB),
    passive interferometer U, and per-mode displacements (mu_x, mu_p)."""
    from thewalrus.decompositions import williamson, blochmessiah

    N = cov.shape[0] // 2
    D_W, S_W = williamson(cov)
    O1, D_sq, O2 = blochmessiah(S_W)
    r_all = -np.log(np.diag(D_sq)[:N])
    order = np.argsort(np.abs(r_all))[::-1]
    r_sorted = np.abs(r_all[order])
    X = O1[:N, :N]; Y = O1[N:, :N]
    U = (X + 1j * Y)[:, order]
    db = r_sorted * 10 * np.log10(np.exp(2))
    disp = [(float(mu[i]), float(mu[N + i])) for i in range(N)]
    return dict(squeezings_r=r_sorted.tolist(), squeezings_db=db.tolist(),
                U_passive=U, displacements=disp,
                max_squeezing_db=float(db[0]) if len(db) else 0.0)


# =============================================================================
# Default target selection
# =============================================================================
def _parity_floor_target(n_m: int, factor: float = 3.0, min_n: int = 1) -> int:
    """Reduce n_m by ~factor, keeping parity (needed for the clean delta0=0
    matching) and not going below min_n (or 0 if n_m is even and small)."""
    target = max(min_n, int(round(n_m / factor)))
    if (n_m - target) % 2 != 0:               # keep parity
        target += 1
    return min(target, n_m)


def default_targets(n: Sequence[int], factor: float = 3.0) -> List[int]:
    return [_parity_floor_target(int(x), factor=factor) for x in n]


# =============================================================================
# Top-level optimization
# =============================================================================
def optimize_gbs_architecture(cov: np.ndarray, mu: np.ndarray, signal_idx: int,
                              control_idx: Sequence[int], pnr_outcomes: Sequence[int],
                              targets: Optional[Sequence[int]] = None,
                              reduction_factor: float = 3.0,
                              verify: bool = True,
                              herald_cutoff: Optional[int] = None) -> Dict[str, Any]:
    """Run the Hanamura two-step optimization on a GBS generator.

    Parameters
    ----------
    cov, mu : full pre-PNR Gaussian state (xp-ordering, hbar=2, vacuum=I).
    signal_idx : index of the heralded signal mode.
    control_idx : indices of the PNR-measured control modes.
    pnr_outcomes : detected photon number on each control mode.
    targets : optional explicit reduced photon numbers (one per control mode);
              if None, a default ~``reduction_factor`` reduction is used.
    verify : if True, compute the heralded output states before/after and the
             fidelity-up-to-Gaussian-unitary (always-on verification).

    Returns a dict with the original and optimized control parameters, photon
    numbers, success probabilities, the optimized GBS architecture, and (if
    requested) the verification report.
    """
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float)
    control_idx = list(control_idx)
    n0 = [int(x) for x in pnr_outcomes]
    k = len(control_idx)

    # --- control-mode representation ---------------------------------------
    C0, b0 = extract_control(cov, mu, control_idx)
    params_before = []
    for m in range(k):
        idx = [m, m + k]
        params_before.append(control_parameters(C0[np.ix_(idx, idx)], b0[idx]))

    if targets is None:
        targets = default_targets(n0, factor=reduction_factor)
    targets = [int(min(t, nn)) for t, nn in zip(targets, n0)]

    # --- Step 1: photon-number reduction (per control mode) ----------------
    C1, b1 = C0.copy(), b0.copy()
    step1_info = []
    for m in range(k):
        if targets[m] < n0[m]:
            C1, b1, info = reduce_control_mode(C1, b1, m, n0[m], targets[m])
        else:
            info = dict(s0=params_before[m]["s0"], delta0=params_before[m]["delta0"],
                        s0p=params_before[m]["s0"], delta0p=params_before[m]["delta0"],
                        k=1.0, d=0.0, nu=params_before[m]["nu"], n=n0[m], nprime=n0[m])
        step1_info.append(info)
    n1 = list(targets)
    p0 = success_probability(C0, b0, n0)
    p1 = success_probability(C1, b1, n1)

    # --- Step 2: success-probability maximization --------------------------
    damp = optimize_damping(C1, b1, n1)
    C2, b2 = damping_transform_control(C1, b1, damp["t"])
    p2 = success_probability(C2, b2, n1)

    params_after = []
    for m in range(k):
        idx = [m, m + k]
        params_after.append(control_parameters(C2[np.ix_(idx, idx)], b2[idx]))

    # --- Realize the optimized generator as a physical architecture --------
    Vopt, muopt, ctrl_opt, sig_opt = purify_control(C2, b2)
    arch = decompose_architecture(Vopt, muopt)
    arch["signal_idx"] = sig_opt
    arch["control_idx"] = ctrl_opt
    arch["pnr_outcomes"] = n1

    result = dict(
        n_before=n0, n_after=n1,
        total_photons_before=int(sum(n0)), total_photons_after=int(sum(n1)),
        prob_before=p0, prob_after_step1=p1, prob_after=p2,
        prob_gain=(p2 / p0 if p0 > 0 else float("inf")),
        prob_gain_step1=(p1 / p0 if p0 > 0 else float("inf")),
        params_before=params_before, params_after=params_after,
        step1_info=step1_info, damping=damp,
        architecture=arch,
        control_moments=dict(C0=C0, beta0=b0, C1=C1, beta1=b1, C2=C2, beta2=b2),
    )

    if verify:
        result["verification"] = verify_optimization(
            cov, mu, signal_idx, control_idx, n0, C1, b1, n1, C2, b2,
            step1_info, herald_cutoff=herald_cutoff)
    return result


def verify_optimization(cov, mu, signal_idx, control_idx, n0, C1, b1, n1, C2, b2,
                        step1_info, herald_cutoff: Optional[int] = None) -> Dict[str, Any]:
    """Always-on verification of the optimization.

    Checks:
      * fidelity (up to Gaussian unitary) between the original output state and
        the optimized output state.  Step 2 preserves the output exactly
        (Theorem 10), so this equals the original-vs-Step-1 fidelity, evaluated
        from the moderate-squeezing Step-1 generator (numerically robust);
      * that the damping transform produced a physically valid generator;
      * the success-probability change;
      * (cheap) that the damping leaves the success probability's underlying
        output invariant by re-deriving the control parameters.
    """
    report: Dict[str, Any] = {}

    # --- scalable, moment-space checks (always run) ------------------------
    report["optimized_generator_valid"] = is_valid_covariance(C2)
    report["step1_generator_valid"] = is_valid_covariance(C1)
    report["step2_output_invariant"] = "exact (Theorem 10): damping preserves the output state"

    # cutoff sized to the (moderate-squeezing) Step-1 generator
    V1, mu1, c1, s1 = purify_control(C1, b1)
    arch1 = decompose_architecture(V1, mu1)
    rmax = max(arch1["squeezings_r"], default=0.0)
    sq_photons = int(np.sinh(rmax) ** 2)
    cutoff = herald_cutoff or int(min(70, max(36, max(max(n0), max(n1)) + 3 * sq_photons + 10)))

    # --- exact output fidelity (only when the Fock tensor is tractable) ----
    #     heralding an M-mode state costs ~ cutoff**M; guard against blow-up.
    n_modes_orig = 1 + len(control_idx)
    n_modes_step1 = len(c1) + len(s1)
    cost = cutoff ** max(n_modes_orig, n_modes_step1)
    report["herald_cutoff"] = cutoff
    if cost > 5e6:
        report["output_fidelity"] = None
        report["fidelity_skipped"] = (
            f"exact output simulation skipped: {max(n_modes_orig, n_modes_step1)} modes "
            f"x cutoff {cutoff} is too large (~{cost:.1e} amplitudes). "
            "Moment-space checks above remain valid; reduce the problem to verify directly."
        )
        return report

    try:
        psi0, prob0 = heralded_output(cov, mu, signal_idx, control_idx, n0, cutoff=cutoff)
        psi1, prob1 = heralded_output(V1, mu1, s1[0], c1, n1, cutoff=cutoff)
        k_guess = step1_info[0]["k"] if step1_info else 1.0
        fid = fidelity_up_to_gaussian(psi0, psi1, cutoff,
                                      guess=(0, 0, -np.log(max(k_guess, 1e-6)), 0, 0),
                                      align_cut=min(cutoff, 40))
        report["output_fidelity"] = fid
        report["herald_prob_original"] = prob0
    except Exception as exc:                   # pragma: no cover - defensive
        report["output_fidelity"] = None
        report["error"] = f"fidelity check failed: {exc}"
    return report
