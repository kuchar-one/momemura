"""
GKP operator construction for truncated Fock spaces.

Constructs the GKP X, Y, Z, U operators and the combined "magic" operator
for arbitrary target superpositions on the Bloch sphere.
"""

import os
import numpy as np
import qutip as qt
from typing import Tuple, Union
import jax
import jax.numpy as jnp

# Constants
SQRT_PI = np.sqrt(np.pi)
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "operators"
)


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_O_operators(N: int) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """
    Generate O_x, O_y, O_z, and O_1 operators for GKP code as defined in the paper.
    """
    _ensure_cache_dir()

    def load_or_gen(name, gen_func):
        path = os.path.join(CACHE_DIR, f"{name}_{N}.npy")
        if os.path.isfile(path):
            try:
                arr = np.load(path)
                return qt.Qobj(arr)
            except Exception:
                pass
        obj = gen_func()
        np.save(path, obj.full())
        return obj

    Ox = load_or_gen("Ox_paper", lambda: (qt.momentum(N) * SQRT_PI).cosm())
    Oz = load_or_gen("Oz_paper", lambda: (qt.position(N) * SQRT_PI).cosm())

    def gen_Oy():
        return ((qt.position(N) - qt.momentum(N)) * SQRT_PI).cosm()
    Oy = load_or_gen("Oy_paper", gen_Oy)

    def gen_O1():
        x_op = qt.position(N)
        p_op = qt.momentum(N)
        term1 = (2 * p_op * SQRT_PI).cosm()
        term2 = (2 * (x_op - p_op) * SQRT_PI).cosm()
        term3 = (2 * x_op * SQRT_PI).cosm()
        return qt.qeye(N) - (term1 + term2 + term3) / 3.0
    O1 = load_or_gen("O1_paper", gen_O1)

    return Ox, Oy, Oz, O1


def high_dim_magic_generator_paper(ux: float, uy: float, uz: float) -> np.ndarray:
    """
    Generate the high-dimensional GKP magic operator (N=1000).
    """
    N = 1000
    filename = os.path.join(
        CACHE_DIR, f"high_dim_O_GKP_{ux:.6g}_{uy:.6g}_{uz:.6g}.npy"
    )

    if os.path.isfile(filename):
        try:
            return np.load(filename)
        except Exception:
            pass

    Ox, Oy, Oz, O1 = get_O_operators(N)
    high_dim = O1 + qt.qeye(N) - (ux * Ox + uy * Oy + uz * Oz)
    arr = high_dim.full()

    _ensure_cache_dir()
    np.save(filename, arr)
    return arr


def get_u_vec(
    ab: Tuple[float, complex], normalize: bool = False, tol: float = 1e-9
) -> Tuple[float, float, float]:
    """
    Convert (a,b) -> (u_x, u_y, u_z) for GKP Hamiltonian construction.
    """
    a, b = ab
    a = float(np.real(a))
    b = complex(b)

    if np.abs(a) > 1 + 1e-12:
        raise ValueError(f"a = {a} has |a|>1 (not a valid cos theta)")

    norm_diff = a * a + (np.abs(b) ** 2) - 1.0
    if normalize and np.abs(norm_diff) > tol:
        if np.abs(b) == 0:
            a = 1.0 if a >= 0 else -1.0
        else:
            scale = np.sqrt(max(0.0, 1.0 - a * a)) / np.abs(b)
            b *= scale

    uz = 2.0 * (a * a) - 1.0
    ux_uy = 2.0 * a * b

    ux = float(ux_uy.real)
    uy = float(ux_uy.imag)

    return (ux, uy, uz)


def get_u_vec_from_alpha_beta(alpha: complex, beta: complex) -> Tuple[float, float, float]:
    """Helper to convert generic alpha, beta to (ux, uy, uz)"""
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    if norm > 1e-9:
        alpha = alpha / norm
        beta = beta / norm
    else:
        raise ValueError("Zero vector passed as target state.")

    phase = np.angle(alpha)
    alpha_rot = alpha * np.exp(-1j * phase)
    beta_rot = beta * np.exp(-1j * phase)

    return get_u_vec((alpha_rot, beta_rot), normalize=True)


def construct_gkp_operator(
    N: int,
    alpha: complex,
    beta: complex,
    backend: str = "jax",
    gaussian_normalize: bool = False,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Construct the GKP operator for a given cutoff N and target superposition (alpha, beta).

    Args:
        N: Cutoff dimension.
        alpha: Coefficient for |0>_L (cos(theta/2)).
        beta: Coefficient for |1>_L (e^{i phi} sin(theta/2)).
        backend: "thewalrus" (numpy) or "jax" (jax.numpy).

    Returns:
        The truncated operator as a matrix.
    """
    ux, uy, uz = get_u_vec_from_alpha_beta(alpha, beta)

    high_dim_arr = high_dim_magic_generator_paper(ux, uy, uz)

    truncated = high_dim_arr[:N, :N]

    if gaussian_normalize:
        truncated = truncated / gaussian_limit(ux, uy, uz)

    if backend == "jax":
        return jnp.array(truncated)
    else:
        return truncated


def gaussian_limit(ux, uy, uz):
    return 5 / 3 - np.max(np.abs([ux, uy, uz]))


def construct_gkp_operator_angle(
    N: int,
    theta: float,
    phi: float,
    backend: str = "jax",
    gaussian_normalize: bool = False,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Construct the GKP operator for a given cutoff N and target superposition.

    Args:
        N: Cutoff dimension.
        theta: Angle theta (in radians).
        phi: Angle phi (in radians).
        backend: "thewalrus" (numpy) or "jax" (jax.numpy).

    Returns:
        The truncated operator as a matrix.
    """

    alpha = np.cos(theta)
    beta = np.sin(theta) * np.exp(1j * phi)
    return construct_gkp_operator(N, alpha, beta, backend, gaussian_normalize)


def get_0L(N: int, backend: str = "jax") -> Union[np.ndarray, jnp.ndarray]:
    """
    Get the GKP |0>_L state as the ground state of O_GKP(0, 0, 1).
    """
    op = construct_gkp_operator(N, 1.0, 0.0, backend="thewalrus")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        _op = jnp.array(op)
        _, evecs = jnp.linalg.eigh(_op)
        psi = evecs[:, 0]
        idx = jnp.argmax(jnp.abs(psi))
        phase = psi[idx] / jnp.abs(psi[idx])
        psi = psi / phase
    if backend == "jax":
        return jnp.array(psi)
    return np.array(psi)


def get_1L(N: int, backend: str = "jax") -> Union[np.ndarray, jnp.ndarray]:
    """
    Get the GKP |1>_L state as the ground state of O_GKP(0, 0, -1).
    """
    op = construct_gkp_operator(N, 0.0, 1.0, backend="thewalrus")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        _op = jnp.array(op)
        _, evecs = jnp.linalg.eigh(_op)
        psi = evecs[:, 0]
        idx = jnp.argmax(jnp.abs(psi))
        phase = psi[idx] / jnp.abs(psi[idx])
        psi = psi / phase
    if backend == "jax":
        return jnp.array(psi)
    return np.array(psi)
