import pytest
import jax
from src.genotypes.genotypes import get_genotype_decoder

# Constants
DEPTH = 3
LEAVES = 8
CUTOFF = 10
MODES = 3  # 1 Signal + 2 Controls

# Length Calculations for N=3:
# GG = 3(r) + 9(ph) + 6(disp) = 18
# PNR = 2
# Leaf(A) = 1(Act) + 1(NC) + 2(PNR) + 18 = 22
# Mix = 3
# Final = 5

# A: 1 + 8*22 + 7*3 + 5 = 203
# 0: 203 - 1 + 7 = 209
# B1: 1 + 21(Shared) + 21 + 5 = 48  (Shared BP = 1(NC)+2(PNR)+18(GG)=21)
# B2: 48 + 8 = 56
# B3: 1 + 18(Shared) + 8*4(Unique) + 21 + 5 = 77  (Unique=1+1+2=4)
# B30: 77 - 1 + 7 = 83
# B3B: 77 (Fixed Balanced) -> PN=3 kept for structure alignment = 77
# B30B: 83 (Fixed Balanced) -> PN=3 kept for structure alignment = 83
# C1: 1 + 21(Shared) + 3(SharedMix) + 5 = 30
# C2: 30 + 8 = 38
# C20: 38 - 1 + 7 = 44
# C2B: 38 (Fixed Balanced) -> PN=3 kept = 38
# C20B: 44 (Fixed Balanced) -> PN=3 kept = 44


@pytest.mark.parametrize(
    "design_name, expected_length",
    [
        ("A", 203),
        ("0", 209),
        ("B1", 48),
        ("B2", 56),
        ("B3", 77),
        ("B30", 83),
        ("B3B", 77),
        ("B30B", 83),
        ("C1", 30),
        ("C2", 38),
        ("C20", 44),
        ("C2B", 38),
        ("C20B", 44),
        ("00B", 209),
    ],
)
def test_genotype_lengths(design_name, expected_length):
    config = {"modes": MODES}
    decoder = get_genotype_decoder(design_name, depth=DEPTH, config=config)
    assert decoder.get_length(DEPTH) == expected_length


@pytest.mark.parametrize(
    "design_name", ["A", "0", "B1", "B2", "B3", "C1", "C2", "C20", "B30B", "00B"]
)
def test_genotype_decode_shapes(design_name):
    config = {"modes": MODES}
    decoder = get_genotype_decoder(design_name, depth=DEPTH, config=config)
    length = decoder.get_length(DEPTH)

    key = jax.random.PRNGKey(42)
    g = jax.random.uniform(key, (length,), minval=-1.0, maxval=1.0)

    decoded = decoder.decode(g, CUTOFF)

    # Top-level keys
    assert "homodyne_x" in decoded
    assert "mix_params" in decoded
    assert "leaf_params" in decoded
    assert "final_gauss" in decoded

    # Shapes
    L = LEAVES
    lp = decoded["leaf_params"]

    # N=3 check
    assert lp["r"].shape == (L, MODES)
    assert lp["phases"].shape == (L, MODES**2)
    assert lp["disp"].shape == (L, MODES)
    assert lp["pnr"].shape == (L, MODES - 1)
    assert lp["n_ctrl"].shape == (L,)

    # Final Gauss
    fg = decoded["final_gauss"]
    assert fg["r"].size == 1
    assert fg["disp"].size == 1
