import numpy as np
from src.genotypes.converter import create_vacuum_genotype, upgrade_genotype
from src.genotypes.genotypes import get_genotype_decoder


def test_vacuum_genotype_identity():
    """Verify vacuum genotype is all zeros and has correct length."""
    cases = ["A", "B1", "B2", "C1", "C2"]
    depth = 3
    config = {"modes": 3}
    for name in cases:
        g = create_vacuum_genotype(name, depth=depth, config=config)
        decoder = get_genotype_decoder(name, depth=depth, config=config)
        expected_len = decoder.get_length(depth)

        assert len(g) == expected_len
        assert np.allclose(g, 0.0)


def test_upgrade_c1_to_b1():
    """Verify C1 (SharedMix) -> B1 (UniqueMix) broadcasts mix params."""
    depth = 3
    config = {"modes": 3}

    # Decoders
    c1_dec = get_genotype_decoder("C1", depth=depth, config=config)
    b1_dec = get_genotype_decoder("B1", depth=depth, config=config)

    # Create random C1
    c1_g = np.random.randn(c1_dec.get_length(depth)).astype(np.float32)

    # Dynamic Indices for C1
    # [Hom(1) | Block(BP) | Mix(PN) | Final(F)]
    bp_len = c1_dec.BP
    pn_len = c1_dec.PN

    idx = 0
    hom_x = c1_g[idx]
    idx += 1

    block = c1_g[idx : idx + bp_len]
    idx += bp_len

    mix = c1_g[idx : idx + pn_len]

    # Convert to B1
    b1_g = upgrade_genotype(c1_g, "C1", "B1", depth=depth, config=config)

    # Verify B1
    assert len(b1_g) == b1_dec.get_length(depth)

    # B1: [Hom(1) | Block(BP) | Mix(Nodes*PN) | Final(F)]
    idx = 0
    assert b1_g[idx] == hom_x
    idx += 1

    assert np.allclose(b1_g[idx : idx + bp_len], block)
    idx += bp_len

    # Mix section: Nodes * PN. Should be broadcasted.
    nodes = 2**depth - 1
    mix_section = b1_g[idx : idx + nodes * pn_len].reshape(nodes, pn_len)

    for i in range(nodes):
        assert np.allclose(mix_section[i], mix)


def test_upgrade_b1_to_a():
    """Verify B1 (SharedBlock) -> A (UniqueBlock) broadcasts block params."""
    depth = 3
    config = {"modes": 3}

    b1_dec = get_genotype_decoder("B1", depth=depth, config=config)
    a_dec = get_genotype_decoder("A", depth=depth, config=config)

    # Construct B1
    b1_g = np.zeros(b1_dec.get_length(depth), dtype=np.float32)

    # Set arbitrary block params to 0.5
    # B1: [Hom(1) | Block(BP) | ...]
    idx = 1
    bp_len = b1_dec.BP
    b1_g[idx : idx + bp_len] = 0.5

    a_g = upgrade_genotype(b1_g, "B1", "A", depth=depth, config=config)
    assert len(a_g) == a_dec.get_length(depth)

    # A: [Hom(1) | Leaves(L*P_leaf) | ...]
    # P_leaf = 1(Act) + BP (roughly, BP has NC/PNR/GG)
    # Actually P_leaf_full = 1(Act) + 1(NC) + PNR + GG
    # B1 BP = 1(NC) + PNR + GG.
    # So P_leaf = 1 + BP.

    L = 2**depth
    p_leaf = a_dec.P_leaf_full

    leaves_section = a_g[1 : 1 + L * p_leaf].reshape(L, p_leaf)

    for i in range(L):
        # A Block: [Active, Param1...ParamBP]
        # B1 Block: [Param1...ParamBP]

        # Check active flag (implicit B1->A sets active=True/1.0)
        assert leaves_section[i, 0] == 1.0

        # Check params (shifted by 1)
        assert np.allclose(leaves_section[i, 1:], 0.5)


def test_upgrade_c1_to_c2():
    """Verify conversion adds active flags."""
    depth = 3
    config = {"modes": 3}
    c1_decoder = get_genotype_decoder("C1", depth=depth, config=config)
    c1_g = np.zeros(c1_decoder.get_length(depth), dtype=np.float32)

    c2_g = upgrade_genotype(c1_g, "C1", "C2", depth=depth, config=config)

    # C2 appends active flags at end (default True=1.0)
    L = 2**depth
    active_flags = c2_g[-L:]
    assert np.allclose(active_flags, 1.0)
