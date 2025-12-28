import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from frontend.utils import compute_active_metrics


def verify_active_metrics_logic():
    print("--- Verifying compute_active_metrics ---")

    # Case 1: n_ctrl masks correctly
    # Leaf 0: PNR [8, 2], n_ctrl=2 -> Sum 10
    # Leaf 1: PNR [8, 8], n_ctrl=0 -> Sum 0 (Previously might sum to 16)

    pnr = np.array([[8, 2], [8, 8]])

    n_ctrl = np.array([2, 0])

    active = np.array([True, True])

    params = {"leaf_params": {"pnr": pnr, "n_ctrl": n_ctrl}, "leaf_active": active}

    total, max_p = compute_active_metrics(params)
    print(f"Total: {total}, Max: {max_p}")

    expected_total = 10.0
    if abs(total - expected_total) < 1e-6:
        print("PASS: Correctly masked n_ctrl=0 leaf")
    else:
        print(f"FAIL: Expected {expected_total}, got {total}")
        return False

    # Case 2: Partial n_ctrl
    # Leaf: PNR [5, 5, 5], n_ctrl=2 -> Sum 10 (5+5), ignore last 5
    pnr2 = np.array([[5, 5, 5]])
    n_ctrl2 = np.array([2])
    active2 = np.array([True])

    params2 = {"leaf_params": {"pnr": pnr2, "n_ctrl": n_ctrl2}, "leaf_active": active2}

    total2, max_p2 = compute_active_metrics(params2)
    print(f"Total2: {total2}")

    if abs(total2 - 10.0) < 1e-6:
        print("PASS: Correctly masked partial PNR")
        return True
    else:
        print(f"FAIL: Expected 10.0, got {total2}")
        return False


if __name__ == "__main__":
    if verify_active_metrics_logic():
        sys.exit(0)
    else:
        sys.exit(1)
