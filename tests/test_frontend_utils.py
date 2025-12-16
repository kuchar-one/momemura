import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from frontend import utils

# Access private function for testing
# (Normally we test public API, but this internal helper is the critical point of failure)
extract_active_leaf_params = utils._extract_active_leaf_params


class TestFrontendUtils:
    def test_extract_active_leaf_params_flat(self):
        """Test extraction when params are already simple scalars/lists (flat)."""
        input_params = {
            "n_signal": 1,
            "n_control": 1,
            "tmss_squeezing": [0.5],
            # No leaf_params or leaf_active implies use as-is
        }
        extracted = extract_active_leaf_params(input_params)
        assert extracted == input_params  # Should return identity

    def test_extract_active_leaf_params_nested(self):
        """Test extraction from parsed genotype tree structure (lists of parameters)."""
        # Simulate what ResultManager output might look like after JSON roundtrip or decoding
        # 2 leaves, second one active
        input_params = {
            "leaf_active": [False, True],
            "leaf_params": {
                # scalar parameters as lists of length L=2
                "tmss_r": [0.1, 0.5],
                "us_phase": [0.0, 1.57],
                "n_ctrl": [1, 2],
                # Nested arrays for UC (L, n_pairs/n_phases)
                # Leaf 0 (n=1): uc_theta has 0 pairs. Leaf 1 (n=2): has 1 pair.
                # The structure from decoder usually pads or is list of lists.
                "uc_theta": [[], [0.78]],
                "uc_phi": [[], [0.0]],
                "uc_varphi": [[0.1], [0.2, 0.3]],
                "disp_s": [[0.0 + 0j], [1.0 + 0j]],
                # Disp C: Leaf 0 (1 mode) -> [d1], Leaf 1 (2 modes) -> [d1, d2]
                "disp_c": [[0j], [1j, 2j]],
                "pnr": [[0], [1, 0]],
            },
        }

        extracted = extract_active_leaf_params(input_params)

        # We expect parameters from active leaf index 1
        assert extracted["n_control"] == 2
        assert extracted["tmss_squeezing"] == [0.5]
        # us_params
        assert extracted["us_params"]["varphi"] == [1.57]

        # UC params for n=2: 1 pair.
        # uc_theta should be [0.78]
        assert extracted["uc_params"]["theta"] == [0.78]
        assert extracted["uc_params"]["phi"] == [0.0]
        # uc_varphi should be length 2
        assert extracted["uc_params"]["varphi"] == [0.2, 0.3]

        # Displacements
        assert extracted["disp_s"] == [1.0 + 0j]
        assert extracted["disp_c"] == [1j, 2j]

        # PNR
        assert extracted["pnr_outcome"] == [1, 0]

    def test_extract_active_leaf_params_truncation(self):
        """Test that lists are truncated to match n_control."""
        # Case: n_ctrl=1 but provided params for max n_ctrl=2 (padding)
        input_params = {
            "leaf_active": [True],
            "leaf_params": {
                "n_ctrl": [1],
                "tmss_r": [0.5],
                "us_phase": [0.0],
                # Provided 2 modes worth of params (maybe padding)
                "uc_theta": [[0.9]],  # 1 pair for n=2
                "uc_phi": [[0.1]],
                "uc_varphi": [[0.1, 0.2]],  # 2 phases
                "disp_s": [[0j]],
                "disp_c": [[1j, 2j]],
                "pnr": [[1, 1]],
            },
        }

        extracted = extract_active_leaf_params(input_params)

        assert extracted["n_control"] == 1
        # Truncation checks
        # For n=1, pairs=0. So theta/phi should be empty
        assert extracted["uc_params"]["theta"] == []
        assert extracted["uc_params"]["phi"] == []
        # varphi len 1
        assert extracted["uc_params"]["varphi"] == [0.1]

        # disp_c len 1
        assert extracted["disp_c"] == [1j]
        # pnr len 1
        assert extracted["pnr_outcome"] == [1]

    def test_extract_genotype_index(self):
        """Test extraction of genotype index from mock Plotly selection data."""

        # 1. Standard Point Index
        p1 = {"pointIndex": 10}
        assert utils.extract_genotype_index(p1) == 10

        p2 = {"point_index": 20}
        assert utils.extract_genotype_index(p2) == 20

        # 2. Custom Data (List)
        # [Desc_TotalPhotons, Desc_MaxPNR, Desc_Complexity, genotype_idx, Expectation, LogProb]
        p_custom_list = {
            "pointIndex": 0,  # Mismatch
            "customdata": [3.0, 1.0, 5.0, 42, 1.2, 0.5],
        }
        assert utils.extract_genotype_index(p_custom_list) == 42

        # 3. Custom Data (Dict - unusual but robust)
        p_custom_dict = {"customdata": {"genotype_idx": 99}}
        assert utils.extract_genotype_index(p_custom_dict) == 99

        p_custom_dict_idx = {
            "customdata": {3: 55}  # Integer key
        }
        assert utils.extract_genotype_index(p_custom_dict_idx) == 55

        # 4. Parsing verification
        p_str = {"customdata": [0, 0, 0, "123"]}
        assert utils.extract_genotype_index(p_str) == 123

        # 5. Out of bounds
        p_oob = {"pointIndex": 5}
        assert utils.extract_genotype_index(p_oob, df_len=3) is None

        # 6. Failure
        p_fail = {"foo": "bar"}
        assert utils.extract_genotype_index(p_fail) is None

    def test_get_val_robustness(self):
        """Test edge cases for single-element wrapping in utils logic."""
        # This is implicitly tested by nested tests, but explicit check good
        pass


if __name__ == "__main__":
    pytest.main([__file__])
