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

    # Helper to load or regenerate
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
        return -(((x_op + p_op) * SQRT_PI).cosm())

    Y = load_or_gen("Y_paper", gen_Y)

    def gen_U():
        x_op = qt.position(N)
        p_op = qt.momentum(N)
        term1 = (2 * p_op * SQRT_PI).cosm()
        term2 = (2 * (x_op + p_op) * SQRT_PI).cosm()
        term3 = (2 * x_op * SQRT_PI).cosm()
        return 2 * qt.qeye(N) - (term1 + term2 + term3) / 3.0

    U = load_or_gen("U_paper", gen_U)

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
    Convert (a,b) -> (c_x, c_y, c_z) where
      a = cos(theta),
      b = e^{i phi} sin(theta).
    """
    a, b = ab
    a = float(np.real(a))
    b = complex(b)

    if np.abs(a) > 1 + 1e-12:
        raise ValueError(f"a = {a} has |a|>1 (not a valid cos theta)")

    # optional normalization
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
    cx = float(cx_cy.real)
    cy = float(cx_cy.imag)

    return (cx, cy, cz)


def construct_gkp_operator(
    N: int, alpha: complex, beta: complex, backend: str = "thewalrus"
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
    # Convert alpha, beta to Bloch coefficients
    # Note: bloch_from_ab expects (a, b) where a is real (cos theta).
    # If alpha is complex, we might need to adjust phase, but usually we define
    # the superposition such that alpha is real or we factor out a global phase.
    # Assuming standard Bloch sphere mapping:
    # |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    # If alpha is complex, we can rotate so alpha is real?
    # Or just pass magnitude?
    # The snippet's bloch_from_ab takes (a, b).

    # We assume the user passes alpha, beta such that |alpha|^2 + |beta|^2 = 1.
    # If alpha is not real, we can multiply by exp(-i arg(alpha)) to make it real,
    # shifting the relative phase to beta.

    # Normalize inputs
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    if norm > 1e-9:
        alpha = alpha / norm
        beta = beta / norm
    else:
        raise ValueError("Zero vector passed as target state.")

    phase = np.angle(alpha)
    alpha_rot = alpha * np.exp(-1j * phase)
    beta_rot = beta * np.exp(-1j * phase)

    # Now alpha_rot is real.
    cx, cy, cz = bloch_from_ab((alpha_rot, beta_rot), normalize=True)

    # Get high-dim operator
    high_dim_arr = high_dim_magic_generator_paper(cx, cy, cz)

    # Truncate
    truncated = high_dim_arr[:N, :N]

    if backend == "jax":
        return jnp.array(truncated)
    else:
        return truncated
