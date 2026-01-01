import pytest
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from frontend import utils


# Access private function for testing
extract_active_leaf_params = utils._extract_active_leaf_params


class TestFrontendUtils:
    def test_extract_active_leaf_params_nested(self):
        """Test extraction from parsed genotype tree structure (lists of parameters)."""
        # Leaf 0: Low energy. Leaf 1: High energy (Active).
        # General Gaussian structure:
        # r: List[List[float]] (L leaves, N modes)
        # phases: List[List[float]]
        input_params = {
            "leaf_active": [False, True],
            "leaf_params": {
                "r": [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]],  # L=2, N=3
                "phases": [[0.0] * 9, [1.57] * 9],
                "n_ctrl": [1, 2],
                "disp": [[0.0] * 6, [1.0] * 6],
                "pnr": [[0, 0], [1, 0]],
            },
        }

        extracted = extract_active_leaf_params(input_params)

        # We expect parameters from active leaf index 1 (Higher energy)
        assert extracted["n_control"] == 2
        # Verify r
        assert extracted["r"] == [0.5, 0.5, 0.5]
        # Verify phases
        assert extracted["phases"] == [1.57] * 9
        # Verify disp
        assert extracted["disp"] == [1.0] * 6
        # Verify pnr
        assert extracted["pnr_outcome"] == [1, 0]
        # Verify flag
        assert extracted.get("is_general_gaussian") is True

    def test_extract_active_leaf_params_truncation(self):
        """Test that missing/short lists are handled gracefully."""
        # Case: Single leaf, N=1 (1r, 1ph, 2disp)
        input_params = {
            "leaf_active": [True],
            "leaf_params": {
                "n_ctrl": [0],
                "r": [[0.8]],  # N=1
                "phases": [[0.1]],  # N^2=1
                "disp": [[0.5, 0.0]],  # 2N=2
                "pnr": [[]],  # N-1 = 0
            },
        }

        extracted = extract_active_leaf_params(input_params)

        assert extracted["n_control"] == 0
        assert extracted["r"] == [0.8]
        assert extracted["phases"] == [0.1]
        assert extracted["disp"] == [0.5, 0.0]
        assert extracted["pnr_outcome"] == []

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
        pass


if __name__ == "__main__":
    pytest.main([__file__])
