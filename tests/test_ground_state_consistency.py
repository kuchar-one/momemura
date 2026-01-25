import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import (
    MagicMock,
    patch,
)  # unused but keeping for future use or remove if strict
# Actually removing them to fix linter


# Adjust path to find src
sys.path.append(os.getcwd())

from src.simulation.jax.runner import jax_scoring_fn_batch
from src.utils.gkp_operator import construct_gkp_operator


def test_gs_eig_affects_gradients():
    """
    Verifies that changing gs_eig changes the computed gradients,
    confirming that gs_eig is participating in the loss function (Tchebycheff).
    """
    if jax is None:
        pytest.skip("JAX not available")

    cutoff = 8
    # Create a dummy operator (Number operator for simplicity)
    # Actually use GKP operator to be realistic
    operator = construct_gkp_operator(cutoff, 2.0, 0.0, backend="jax")
    op_jax = jnp.array(operator)

    # Random genotype
    from src.genotypes.genotypes import get_genotype_decoder

    key = jax.random.PRNGKey(42)
    genotype_config = {"modes": 2, "depth": 3}
    decoder = get_genotype_decoder("A", depth=3, config=genotype_config)
    D = decoder.get_length(3)
    genotypes = jax.random.uniform(key, (2, D))

    # 1. Run with gs_eig = -10.0
    _, _, extras_1 = jax_scoring_fn_batch(
        genotypes,
        cutoff,
        op_jax,
        genotype_name="A",
        genotype_config={
            "modes": 2,
            "depth": 3,
            "alpha_expectation": 1.0,
            "alpha_probability": 1.0,
        },
        gs_eig=-10.0,
    )
    grads_1 = extras_1["gradients"]

    # 2. Run with gs_eig = -2.0
    _, _, extras_2 = jax_scoring_fn_batch(
        genotypes,
        cutoff,
        op_jax,
        genotype_name="A",
        genotype_config={
            "modes": 2,
            "depth": 3,
            "alpha_expectation": 1.0,
            "alpha_probability": 1.0,
        },
        gs_eig=-2.0,
    )
    grads_2 = extras_2["gradients"]

    # Gradients should be different because the scalarization scalar changes
    # Tchebycheff terms will be different distances from Utopia.

    diff = jnp.sum(jnp.abs(grads_1 - grads_2))
    print(f"\nGradient Difference when changing gs_eig: {diff}")

    assert diff > 1e-6, "Gradients should differ when gs_eig (Utopia point) changes!"


def test_run_mome_calculates_gs_eig():
    """
    Verifies that run_mome.py calculates gs_eig correctly from the operator.
    We will mock the HanamuraMOMEAdapter to inspect it.
    """
    from run_mome import HanamuraMOMEAdapter
    from src.simulation.cpu.composer import Composer, SuperblockTopology

    cutoff = 5
    # Create a diagonal operator with known eigenvalues: [1, 2, 3, 4, 5]
    op = np.diag(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    composer = Composer(cutoff=cutoff, backend="jax")
    topology = SuperblockTopology.build_layered(2)

    adapter = HanamuraMOMEAdapter(
        composer, topology, op, cutoff, backend="jax", genotype_name="A"
    )

    # The lowest eigenvalue is 1.0.
    # run_mome logic: self.gs_exp = float(jnp.linalg.eigvalsh(jnp.array(operator))[0])

    print(f"\nCalculated GS Exp: {adapter.gs_exp}")
    assert abs(adapter.gs_exp - 1.0) < 1e-5, f"Expected 1.0, got {adapter.gs_exp}"


if __name__ == "__main__":
    test_gs_eig_affects_gradients()
    test_run_mome_calculates_gs_eig()
    print("Tests passed!")
