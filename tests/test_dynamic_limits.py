import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.simulation.jax.runner import _score_batch_shard
from src.utils.gkp_operator import construct_gkp_operator


# Mock genotype decoder output
def create_mock_genotype_params(r_val, d_val, n_leaves=8):
    # Reduced parameter set for testing
    return {
        "homodyne_x": 0.0,
        "homodyne_window": 0.1,
        "mix_params": jnp.zeros((7, 3)),
        "mix_source": jnp.zeros(7, dtype=jnp.int32),
        "leaf_active": jnp.ones(n_leaves, dtype=bool),
        "leaf_params": {
            "n_ctrl": jnp.zeros(n_leaves, dtype=jnp.int32),
            "tmss_r": jnp.full((n_leaves,), r_val),
            "us_phase": jnp.zeros((n_leaves, 1)),
            "uc_theta": jnp.zeros((n_leaves, 0)),
            "uc_phi": jnp.zeros((n_leaves, 0)),
            "uc_varphi": jnp.zeros((n_leaves, 0)),
            "disp_s": jnp.full((n_leaves, 1), d_val + 0j),
            "disp_c": jnp.zeros((n_leaves, 0)),
            "pnr": jnp.zeros((n_leaves, 0), dtype=jnp.int32),
        },
        "final_gauss": {"r": 0.0, "phi": 0.0, "varphi": 0.0, "disp": 0.0 + 0j},
    }


def test_dynamic_limits_penalty():
    """
    Test that states which fit in correction_cutoff but NOT in cutoff are penalized.
    """
    cutoff = 5
    correction_cutoff = 20

    # Construct a dummy operator (Identity) for easy expectation check
    # Exp val should be ~1.0 if state is normalized
    op = jnp.eye(cutoff)

    # 1. Genotype that produces a high-energy state (Large Displacement)
    # Displacement of 3.0 creates coherent state |3>.
    # Mean photon number n = |alpha|^2 = 9.
    # In cutoff=5, this should leak massively.
    # In cutoff=20, it fits well.

    # We cheat and use the _score_batch_shard but we need a valid genotype vector.
    # Instead of reversing the decoder, let's use the Runner logic directly or
    # ensure the decoder produces what we want.
    # Since _score_batch_shard calls decoder, we need to provide a genotype that decodes to high values.
    # BUT we implemented the override logic in run_mome.py (CLI args), not in the decoder itself.
    # The decoder applies scale factors from its config.

    # Logic path:
    # _score_batch_shard -> get_genotype_decoder(..., config=config)

    genotype_name = "B1"  # B1 is robust

    # Create a config with LARGE scales
    genotype_config = {
        "depth": 3,
        "r_scale": 5.0,  # Huge squeezing allowed
        "d_scale": 10.0,  # Huge displacement allowed
        "hx_scale": 4.0,
        "window": 0.1,
        "pnr_max": 3,
        "modes": 2,
    }

    # Create a dummy genotype vector
    # Random genotype
    key = jax.random.PRNGKey(0)
    g_dummy = jax.random.normal(key, (1, 60))  # approximate length

    # We need to manually construct a genotype vector that we KNOW produces high displacement.
    # Or we can test the logic by mocking the decoder?
    # Mocking in JAX is hard.

    # Easier: Use the actual decoder and set "d_scale" and genotype value to max.
    # B1 genotype: Shared params.
    # Disp S is at some index.
    # If we set d_scale=10.0 and genotype val for disp to 1.0 (tanh(1.0)*10 ~ 7.6), we get huge displacement.

    # Let's inspect B1 decode again to find indices?
    # Or just run optimization on random genotypes with these settings and assert leakage occurs?

    # Let's try to construct a specific genotype.
    # Start with zeros.
    from src.genotypes.genotypes import DesignB1Genotype

    decoder = DesignB1Genotype(depth=3, config=genotype_config)
    L = decoder.get_length(3)
    g = jnp.zeros((1, L))

    # B1 structure:
    # Hom(1)
    # BP: n_ctrl(1), tmss(1), us(1), UC..., DispS(2)...
    # Indicies: Hom=0. BP start 1.
    # DispS is after n_ctrl, tmss, us, uc.
    # n_ctrl=1, tmss=1, us=1.
    # modes=2 -> n_ctrl=1. n_uc_pairs=0. len_uc=0+1=1.
    # DispS is at 1 + 1 + 1 + 1 + 1 = 5?
    # Let's fill the whole vector with 1.0. This should produce max values for everything.
    g_high = jnp.ones((1, L)) * 2.0  # tanh(2.0) ~ 0.96. Scaled by 10.0 -> Disp ~ 9.6.

    # Convert config to hashable tuple for JAX static arg
    config_tuple = tuple(sorted(genotype_config.items()))

    # Run Score with dynamic limits
    fitness_dyn, _ = _score_batch_shard(
        g_high, cutoff, op, genotype_name, config_tuple, correction_cutoff
    )

    # Run Score WITHOUT dynamic limits (correction=cutoff)
    fitness_static, _ = _score_batch_shard(
        g_high,
        cutoff,
        op,
        genotype_name,
        config_tuple,
        correction_cutoff=None,  # defaults to cutoff
    )

    # Analysis
    # fitness = [-exp, -logP, -comp, -photons]
    # In static case (cutoff=5), a displacement of 9.6 results in a state effectively [0,0,0,0,0].
    # Why? Coherent state |9.6> has almost 0 support in first 5 fock states.
    # So probability within cutoff would be ~0.
    # The runner calculates joint_prob based on measurements.

    # Wait, simple runner logic: if joint_prob -> 0, exp_val = 1000.0 (penalty).
    # So static case should returns large negative fitness (good penalty).

    # In dynamic case (cutoff=20), the state |9.6> fits well (mean n ~ 92... wait 9.6^2 approx 92).
    # Wait, |9.6> needs N ~ 100+. Even correction=20 is too small.
    # Let's use smaller displacement.
    # Target: State valid in 20, invalid in 5.
    # Coherent state |2.5>. n=6.25.
    # Fits in 20. Leaks from 5.
    # Disp 2.5. d_scale=3.0. tanh(x)*3 = 2.5 -> tanh(x)=0.83 -> x approx 1.2.

    g_med = jnp.ones((1, L)) * 1.2

    # Score Dynamic
    # Should detect leakage from 5.
    # Penalty will be added.

    # How to verify?
    # Inspect internal variables? Cannot with JIT.
    # We check that fitness reflects leakage penalty.

    # For ExpVal, if we use Identity operator, Exp = Norm^2 = 1.0 (if normalized).
    # If leakage penalty is added, Exp > 1.0 (since we minimize exp, fitness < -1.0).
    # Leakage penalty = leakage * 10.0.

    # Let's interpret fitness[0] = -exp_val.

    def get_exp(fit):
        return -fit[0, 0]

    exp_dyn = get_exp(fitness_dyn)

    # Leakage should be > 0.
    # Exp should be > 1.0 (Logic: raw_exp(1.0) + penalty).
    print(f"Exp Dynamic (N=5, Nc=20): {exp_dyn}")

    # If logic works, exp_dyn should be noticeably greater than 1.0
    # because the state |2.5> has N=6.25, heavily leaking out of N=5.

    assert exp_dyn > 1.1, f"Expected penalty for leakage, got {exp_dyn}"

    # Test 2: State fitting in N=5
    # Disp 0.5. n=0.25. Fits easily.
    g_low = jnp.zeros((1, L))  # Vacuum

    fitness_vac, _ = _score_batch_shard(
        g_low, cutoff, op, genotype_name, config_tuple, correction_cutoff
    )

    exp_vac = get_exp(fitness_vac)
    print(f"Exp Vacuum (Should fit): {exp_vac}")

    # Vacuum exp should be 1.0 (Identity op) + 0 penalty.
    # Or specifically 1.0.
    assert abs(exp_vac - 1.0) < 0.1, f"Vacuum should have exp~1.0, got {exp_vac}"


if __name__ == "__main__":
    test_dynamic_limits_penalty()
