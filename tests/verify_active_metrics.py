import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.utils import compute_active_metrics


def test_active_metrics():
    print("Testing compute_active_metrics...")

    # Case 1: Standard Design A/B structure (Leaf Params + Active Flags)
    # 4 Leaves. 2 Active, 2 Inactive.
    # Active: [True, False, True, False]
    # PNR: [[1, 0, 0], [3, 3, 3], [0, 2, 0], [1, 1, 1]] (Shape 4, 3)
    # Expected Active PNR: [1, 0, 0] + [0, 2, 0]
    # Total Photons: 1 + 2 = 3.
    # Max PNR: 2.

    params_1 = {
        "leaf_active": [True, False, True, False],
        "leaf_params": {
            "pnr": [
                [1, 0, 0],  # Active
                [3, 3, 3],  # Inactive
                [0, 2, 0],  # Active
                [1, 1, 1],  # Inactive
            ]
        },
    }

    total, max_val = compute_active_metrics(params_1)
    print(f"Case 1 (Mixed Active): Total={total}, Max={max_val}")

    assert total == 3.0, f"Case 1 Failed: Expected 3.0, got {total}"
    assert max_val == 2.0, f"Case 1 Failed: Expected 2.0, got {max_val}"

    # Case 2: Legacy / No Active Flags (Implicit All Active)
    # PNR: [[1], [2]]
    params_2 = {"leaf_params": {"pnr": [[1], [2]]}}

    total, max_val = compute_active_metrics(params_2)
    print(f"Case 2 (Implicit Active): Total={total}, Max={max_val}")

    assert total == 3.0, f"Case 2 Failed: Expected 3.0, got {total}"
    assert max_val == 2.0, f"Case 2 Failed: Expected 2.0, got {max_val}"

    # Case 3: All Inactive
    params_3 = {"leaf_active": [False, False], "leaf_params": {"pnr": [[5], [5]]}}

    total, max_val = compute_active_metrics(params_3)
    print(f"Case 3 (All Inactive): Total={total}, Max={max_val}")

    assert total == 0.0, f"Case 3 Failed: Expected 0.0, got {total}"
    assert max_val == 0.0, f"Case 3 Failed: Expected 0.0, got {max_val}"

    # Case 4: No PNR keys (Empty/Error)
    params_4 = {}
    total, max_val = compute_active_metrics(params_4)
    print(f"Case 4 (Empty): Total={total}, Max={max_val}")
    assert total == 0.0

    print("All tests passed!")


if __name__ == "__main__":
    test_active_metrics()
