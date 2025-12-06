import pytest
import sys
import os
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_mome import main, run
from src.genotypes.genotypes import get_genotype_decoder


@pytest.mark.parametrize("genotype", ["legacy", "A", "B1", "B2", "C1", "C2"])
def test_pipeline_random_smoke(genotype, tmp_path):
    """
    Smoke test for the full pipeline in random mode with different genotypes.
    """
    # Use tmp_path to avoid cluttering output
    # Mock plotting to save time/deps
    with patch("run_mome.plot_mome_results") as mock_plot:
        run(
            mode="random",
            n_iters=2,
            pop_size=4,
            seed=42,
            cutoff=6,
            backend="thewalrus",  # Test fallback/serial path or standard path
            # Wait, random mode usually uses serial fallback in standard run_mome unless changed?
            # actually run_mome logic I wrote:
            # if backend=="jax": use jax batch
            # else: use serial loop
            # Here keeping default "thewalrus" means we test the serial loop + evaluate_one logic.
            # But evaluate_one requires JAX now (raises RuntimeError if missing).
            # So we MUST use backend="jax" or ensure JAX is present and logic uses it.
            # Actually, evaluate_one uses JAX internally if available.
            # So backend="thewalrus" is fine as long as JAX is installed in env.
            no_plot=True,
            genotype=genotype,
            low_mem=True,
        )


@pytest.mark.parametrize("genotype", ["A", "B2"])
def test_pipeline_jax_backend(genotype, tmp_path):
    """
    Test JAX backend explicitly.
    """
    # Skip if JAX not installed?
    try:
        import jax
    except ImportError:
        pytest.skip("JAX not installed")

    with patch("run_mome.plot_mome_results") as mock_plot:
        run(
            mode="random",  # Use random mode but with JAX backend (uses batched scoring)
            n_iters=2,
            pop_size=4,
            seed=42,
            cutoff=6,
            backend="jax",
            no_plot=True,
            genotype=genotype,
        )
