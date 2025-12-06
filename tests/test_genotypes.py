import pytest
import jax.numpy as jnp
import jax
from src.genotypes.genotypes import get_genotype_decoder

# Constants
DEPTH = 3
LEAVES = 8
CUTOFF = 10


@pytest.mark.parametrize(
    "design_name, expected_length_fn",
    [
        ("legacy", lambda L: 256),
        ("A", lambda L: 20 * L + 2),
        ("B1", lambda L: 17 + 4 * L),
        ("B2", lambda L: 17 + 5 * L),
        ("C1", lambda L: 25),
        ("C2", lambda L: 25 + L),
    ],
)
def test_genotype_lengths(design_name, expected_length_fn):
    # Test assumes 3 modes (1 Signal + 2 Control)
    config = {"modes": 3}
    decoder = get_genotype_decoder(design_name, depth=DEPTH, config=config)
    assert decoder.get_length(DEPTH) == expected_length_fn(LEAVES)


@pytest.mark.parametrize("design_name", ["legacy", "A", "B1", "B2", "C1", "C2"])
def test_genotype_decode_shapes(design_name):
    # Test assumes 3 modes
    config = {"modes": 3}
    decoder = get_genotype_decoder(design_name, depth=DEPTH, config=config)
    L = LEAVES
    length = decoder.get_length(DEPTH)

    # Random genotype
    key = jax.random.PRNGKey(42)
    g = jax.random.uniform(key, (length,), minval=-1.0, maxval=1.0)

    decoded = decoder.decode(g, CUTOFF)

    # Check top-level keys
    assert "homodyne_x" in decoded
    assert "homodyne_window" in decoded
    assert "mix_params" in decoded
    assert "mix_source" in decoded
    assert "leaf_active" in decoded
    assert "leaf_params" in decoded
    assert "final_gauss" in decoded

    # Check shapes
    assert decoded["leaf_active"].shape == (L,)
    assert decoded["mix_params"].shape == (L - 1, 3)
    assert decoded["mix_source"].shape == (L - 1,)

    # Check Leaf Params
    lp = decoded["leaf_params"]
    assert lp["n_ctrl"].shape == (L,)

    # Canonical: tmss_r is (L,) (1D)
    assert lp["tmss_r"].shape == (L,)

    assert lp["uc_varphi"].shape == (L, 2)
    assert lp["disp_c"].shape == (L, 2)
    assert lp["pnr"].shape == (L, 2)

    # Check Final Gaussian
    fg = decoded["final_gauss"]
    assert isinstance(fg["r"], (float, jnp.ndarray)) or fg["r"].shape == ()
    assert isinstance(fg["disp"], (complex, jnp.ndarray)) or fg["disp"].shape == ()

    if design_name == "legacy":
        # Legacy should have final_gauss zeroed
        assert fg["r"] == 0.0
        assert fg["disp"] == 0.0 + 0.0j
