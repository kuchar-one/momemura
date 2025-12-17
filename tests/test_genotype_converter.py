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
    # Create random C1
    c1_decoder = get_genotype_decoder("C1", depth=depth, config=config)
    c1_g = np.random.randn(c1_decoder.get_length(depth)).astype(np.float32)

    # Extract expected values manually (roughly)
    # C1: [Hom(1) | Block(15) | Mix(4) | Final(5)]
    hom_x = c1_g[0]
    block = c1_g[1:16]
    mix = c1_g[16:19]

    # Convert to B1
    b1_g = upgrade_genotype(c1_g, "C1", "B1", depth=depth, config=config)
    b1_decoder = get_genotype_decoder("B1", depth=depth, config=config)
    assert len(b1_g) == b1_decoder.get_length(depth)

    # B1: [Hom(1) | Block(15) | Mix(Nodes*4) | Final(5)]
    # Check Hom
    assert b1_g[0] == hom_x
    # Check Block
    assert np.allclose(b1_g[1:16], block)

    # Check Mix: B1 has 7 nodes. C1 has 1 mix set.
    # B1 mix should be 7 copies of C1 mix.
    nodes = 2**depth - 1
    mix_section = b1_g[16 : 16 + nodes * 3].reshape(nodes, 3)
    for i in range(nodes):
        assert np.allclose(mix_section[i], mix)


def test_upgrade_b1_to_a():
    """Verify B1 (SharedBlock) -> A (UniqueBlock) broadcasts block params."""
    depth = 3
    config = {"modes": 3}
    # We need to manually construct B1 genotype because create_vacuum doesn't take config yet?
    b1_decoder = get_genotype_decoder("B1", depth=depth, config=config)
    b1_g = np.zeros(b1_decoder.get_length(depth), dtype=np.float32)

    # Set arbitrary block params
    b1_g[1:16] = 0.5

    a_g = upgrade_genotype(b1_g, "B1", "A", depth=depth, config=config)
    a_decoder = get_genotype_decoder("A", depth=depth, config=config)
    assert len(a_g) == a_decoder.get_length(depth)

    # A: [Hom(1) | Blocks(L*16) | Mix(Nodes*4) | Final(5)]
    # Blocks in A contain Active flag at pos 0.
    # B1 implies Active=True (val=1.0)

    L = 2**depth
    blocks_section = a_g[1 : 1 + L * 16].reshape(L, 16)

    for i in range(L):
        # A Block: [Active, Param1...Param15]
        # B1 Block: [Param1...Param15]
        assert blocks_section[i, 0] == 1.0  # Implicit active
        assert np.allclose(blocks_section[i, 1:], 0.5)


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
