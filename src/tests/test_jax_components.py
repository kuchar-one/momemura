import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.linalg import expm
import pytest

from src.circuits.jax_composer import jax_u_bs, jax_hermite_phi_matrix

HBAR = 2.0


def test_jax_hermite_phi_matrix():
    cutoff = 10
    xs = np.linspace(-5, 5, 100)

    # JAX implementation
    phi_jax = jax_hermite_phi_matrix(jnp.array(xs), cutoff, hbar=HBAR)

    # Reference implementation (scipy)
    phi_ref = np.zeros((cutoff, len(xs)))
    for n in range(cutoff):
        # phi_n(x) = (1/sqrt(2^n n! sqrt(pi hbar))) * H_n(x/sqrt(hbar)) * exp(-x^2/(2 hbar))
        norm = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi * HBAR))
        h_n = eval_hermite(n, xs / np.sqrt(HBAR))
        exp_part = np.exp(-(xs**2) / (2 * HBAR))
        phi_ref[n, :] = norm * h_n * exp_part

    np.testing.assert_allclose(phi_jax, phi_ref, atol=1e-6)


def test_jax_u_bs():
    cutoff = 5
    theta = 0.5
    phi = 0.3

    # JAX implementation
    U_jax = jax_u_bs(theta, phi, cutoff)

    # Reference implementation (manual matrix exp)
    # Generator for BS(theta, phi):
    # exp( theta * (e^{i phi} a^dag b - e^{-i phi} a b^dag) )
    # Note: TheWalrus/StrawberryFields convention might differ in sign or phase.
    # Let's check against the standard definition used in our code comments.
    # Our code comment said: G = theta * (exp(i phi) a b^dag - exp(-i phi) a^dag b)
    # But my implementation used: term = exp(i phi) a b^dag. G = theta * (term - term.conj().T)
    # term.conj().T = exp(-i phi) a^dag b
    # So implementation was: theta * (exp(i phi) a b^dag - exp(-i phi) a^dag b)
    # This matches the comment.

    # Let's verify if this matches the 2x2 matrix on the single-photon subspace.
    # Single photon subspace basis: |1,0>, |0,1>.
    # |1,0> corresponds to index 1 in mode 1, index 0 in mode 2.
    # In joint basis (cutoff^2), indices:
    # |1,0> -> 1 * cutoff + 0 = cutoff
    # |0,1> -> 0 * cutoff + 1 = 1

    # Let's check the submatrix of U_jax at these indices.
    idx_10 = cutoff
    idx_01 = 1

    # Submatrix [[<10|U|10>, <10|U|01>], [<01|U|10>, <01|U|01>]]
    # U acts on column vectors.
    # U |10> = c |10> + s e^{i phi} |01> ?
    # Let's check standard matrix:
    # [ a_out ] = [ cos   -e^{-i phi} sin ] [ a_in ]
    # [ b_out ]   [ e^{i phi} sin    cos  ] [ b_in ]
    # This is Heisenberg picture (operators).
    # Schrodinger picture (states): |psi_out> = U |psi_in>
    # a_out = U^dag a U

    # If U = exp( theta (e^{i phi} a^dag b - e^{-i phi} a b^dag) )
    # Then U^dag a U = cos theta a - e^{-i phi} sin theta b
    # This matches the first row of the matrix above.

    # So the generator theta * (e^{i phi} a^dag b - e^{-i phi} a b^dag) is correct for the standard BS.

    # Let's verify my implementation logic again.
    # term = exp(i phi) a b^dag
    # G = theta * (term - term.conj().T)
    # G = theta * (e^{i phi} a b^dag - e^{-i phi} a^dag b)
    # This is the NEGATIVE of the standard generator.
    # Standard: e^{i phi} a^dag b - ...
    # Mine: e^{i phi} a b^dag - ...

    # So I expect my implementation to give the inverse (or transpose conjugate).
    # Let's check the matrix elements.

    submat = np.array(
        [
            [U_jax[idx_10, idx_10], U_jax[idx_10, idx_01]],
            [U_jax[idx_01, idx_10], U_jax[idx_01, idx_01]],
        ]
    )

    # Expected for U_std:
    # U |10> = a^dag |00> -> (cos a^dag - e^{-i phi} sin b^dag) |00> = cos |10> - e^{-i phi} sin |01>
    # So <10|U|10> = cos, <01|U|10> = -e^{-i phi} sin

    # U |01> = b^dag |00> -> (e^{i phi} sin a^dag + cos b^dag) |00> = e^{i phi} sin |10> + cos |01>
    # So <10|U|01> = e^{i phi} sin, <01|U|01> = cos

    # Expected matrix is M* (element-wise conjugate of interferometer matrix M)
    # M = [[cos, -e^{-i phi} sin], [e^{i phi} sin, cos]]
    # M* = [[cos, -e^{i phi} sin], [e^{-i phi} sin, cos]]

    expected = np.array(
        [
            [np.cos(theta), np.exp(1j * phi) * np.sin(theta)],
            [-np.exp(-1j * phi) * np.sin(theta), np.cos(theta)],
        ]
    )

    print("Submatrix from JAX:")
    print(submat)
    print("Expected:")
    print(expected)

    np.testing.assert_allclose(submat, expected, atol=2e-4)


if __name__ == "__main__":
    test_jax_hermite_phi_matrix()
    test_jax_u_bs()
