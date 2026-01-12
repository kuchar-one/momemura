import numpy as np
import pytest
from src.utils.result_manager import (
    OptimizationResult,
    AggregatedOptimizationResult,
    SimpleRepertoire,
)


def test_aggregation_heterogeneous_genotypes(tmp_path):
    """
    Test that AggregatedOptimizationResult correctly aggregates runs with different genotype lengths.
    """
    # Create two fake runs
    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()

    # Run 1: len=10
    g1 = np.zeros((5, 10))
    f1 = np.zeros((5, 4))  # 4 objectives
    d1 = np.zeros((5, 3))
    rep1 = SimpleRepertoire(g1, f1, d1)
    res1 = OptimizationResult(
        repertoire=rep1, config={"test": 1}, centroids=np.zeros((1, 3))
    )
    res1.save(str(run1_dir))

    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()

    # Run 2: len=20 (longer)
    g2 = np.zeros((5, 20))
    f2 = np.zeros((5, 4))
    d2 = np.zeros((5, 3))
    rep2 = SimpleRepertoire(g2, f2, d2)
    res2 = OptimizationResult(
        repertoire=rep2, config={"test": 2}, centroids=np.zeros((1, 3))
    )
    res2.save(str(run2_dir))

    # Aggregate
    agg = AggregatedOptimizationResult.load_group(str(tmp_path))

    # Compute Pareto Front (triggers concatenation)
    df = agg.get_pareto_front()

    assert len(df) == 10
    assert agg._cached_valid_genotypes.shape == (10, 20)

    # Verify padding (first 5 should have zeros in last 10 cols)
    # They are already zeros, so let's make run1 ones

    # Re-do with ones to check padding strictly
    g1[:] = 1.0
    rep1 = SimpleRepertoire(g1, f1, d1)
    res1 = OptimizationResult(
        repertoire=rep1, config={"test": 1}, centroids=np.zeros((1, 3))
    )
    res1.save(str(run1_dir))

    # Reload
    agg = AggregatedOptimizationResult.load_group(str(tmp_path))
    df = agg.get_pareto_front()

    # Check first 5 rows (from run1, len=10)
    # The aggregated matrix is (N, 20).
    # Run 1 is padded to 20. So [:, 10:] should be 0.
    # Note: Loading order depends on os.listdir, might be swapped.
    # We check if we have rows with valid padding.

    concatenated = agg._cached_valid_genotypes

    # We expect 5 rows that are [1,1..,1 (10 times), 0,0..,0 (10 times)]
    # And 5 rows that are [0...0 (20 times)]

    found_padded = False
    for i in range(10):
        row = concatenated[i]
        if np.all(row[:10] == 1.0) and np.all(row[10:] == 0.0):
            found_padded = True

    assert found_padded, "Did not find correctly padded rows from shorter genotype run"
    print("Aggregation successful with heterogeneous shapes.")


if __name__ == "__main__":
    test_aggregation_heterogeneous_genotypes(pytest.ensuretemp("test_agg"))
