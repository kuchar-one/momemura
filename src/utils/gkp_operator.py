import os
import numpy as np
import qutip as qt
from typing import Tuple, Union
import jax.numpy as jnp

# Constants
SQRT_PI = np.sqrt(np.pi)
CACHE_DIR = "cache/operators"


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_XYZU_paper(N: int) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    """
    Generate X, Y, Z, U operators for GKP code as defined in the paper.
    Uses QuTiP for construction.
    """
    _ensure_cache_dir()

    def load_or_gen(name, gen_func):
        path = os.path.join(CACHE_DIR, f"{name}_{N}.npy")
        if os.path.isfile(path):
            try:
                arr = np.load(path)
                return qt.Qobj(arr)
            except Exception:
                print(f"Cache corrupted for {path}, regenerating...")

        obj = gen_func()
        try:
            np.save(path, obj.full())
        except Exception:
            pass
        return obj

    X = load_or_gen("X_paper", lambda: (qt.momentum(N) * SQRT_PI).cosm())
    Z = load_or_gen("Z_paper", lambda: (qt.position(N) * SQRT_PI).cosm())

    def gen_Y():
        x_op = qt.position(N)
        p_op = qt.momentum(N)
        return ((x_op + p_op) * SQRT_PI).cosm()

    Y = load_or_gen("Y_paper_v2", gen_Y)

    def gen_U():
        x_op = qt.position(N)
        p_op = qt.momentum(N)
        term1 = (2 * p_op * SQRT_PI).cosm()
        term3 = (2 * x_op * SQRT_PI).cosm()
        return 2 * qt.qeye(N) - (term1 + term3) / 2.0

    U = load_or_gen("U_paper_v2", gen_U)

    return X, Y, Z, U


def high_dim_magic_generator_paper(cx: float, cy: float, cz: float) -> np.ndarray:
    """
    Generate the high-dimensional GKP magic operator (N=1000).
    """
    N = 1000  # Fixed dimension for high-dim generation
    filename = os.path.join(
        CACHE_DIR, f"high_dim_magic_operator_paper_{cx}_{cy}_{cz}.npy"
    )

    if os.path.isfile(filename):
        try:
            return np.load(filename)
        except Exception:
            print(f"Cache corrupted for {filename}, regenerating...")

    print(f"Generating pre-truncation form of the GKP Operator (N={N})...")
    X, Y, Z, U = get_XYZU_paper(N)
    high_dim = cx * X + cy * Y + cz * Z + U
    arr = high_dim.full()

    _ensure_cache_dir()
    np.save(filename, arr)
    return arr


def bloch_from_ab(
    ab: Tuple[float, complex], normalize: bool = False, tol: float = 1e-9
) -> Tuple[float, float, float]:
    """
    Convert (a,b) -> (c_x, c_y, c_z) for GKP Hamiltonian construction.

    The returned coefficients are used to construct the operator:
       H = c_x * X + c_y * Y + c_z * Z + U

    Note: The GKP X operator behaves effectively as -X_Pauli in the logical subspace.
    Therefore, this function flips the sign of c_x relative to the standard Bloch vector
    representation of influence on the Hamiltonian, ensuring that the ground state
    of the constructed operator matches the input state |psi> = a|0> + b|1>.
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
    elif abs(norm_diff) > 1e-6:
        import warnings

        warnings.warn(
            f"a^2 + |b|^2 = {a * a + abs(b) ** 2:.6g} != 1 (tol={tol}). "
            "Check inputs or enable normalize=True."
        )

    cz = 1.0 - 2.0 * (a * a)
    cx_cy = -2.0 * a * b
    # Based on phase conventions X ~ +X_L, Y ~ -Y_L, Z ~ +Z_L
    # To make target state the ground state of H = cx*X + cy*Y + cz*Z + U:
    # Bloch vector: x = 2*Re(a*b), y = 2*Im(a*b), z = |a|²-|b|²
    # cx_cy = -2*a*b, so cx_cy.real = -x, cx_cy.imag = -y
    # Convention: cx = -x, cy = +y, cz = -z
    cx = float(cx_cy.real)  # cx_cy.real = -x  →  cx = -x
    cy = -float(cx_cy.imag)  # cx_cy.imag = -y  →  cy = +y

    return (cx, cy, cz)


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
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    if norm > 1e-9:
        alpha = alpha / norm
        beta = beta / norm
    else:
        raise ValueError("Zero vector passed as target state.")

    phase = np.angle(alpha)
    alpha_rot = alpha * np.exp(-1j * phase)
    beta_rot = beta * np.exp(-1j * phase)

    cx, cy, cz = bloch_from_ab((alpha_rot, beta_rot), normalize=True)

    high_dim_arr = high_dim_magic_generator_paper(cx, cy, cz)

    truncated = high_dim_arr[:N, :N]

    if gaussian_normalize:
        truncated = truncated / gaussian_limit(cx, cy, cz)

    if backend == "jax":
        return jnp.array(truncated)
    else:
        return truncated


def gaussian_limit(cx, cy, cz):
    return 5 / 3 - np.max(np.abs([cx, cy, cz]))


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
