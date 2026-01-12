import os
import sys

# Force CPU JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.simulation.jax.runner import jax_scoring_fn_batch
from frontend.utils import compute_state_with_jax


def test_frontend_backend_consistency():
    """
    Verify that the frontend simulation matches the backend scoring EXACTLY.
    Specifically checks for issues when 'correction_cutoff' (Dynamic Limits) is used.
    """

    # 1. Setup Configuration mimicking a real run with dynamic limits
    # User might use cutoff=12 but correction_cutoff=30
    cutoff = 12
    correction_cutoff = 30
    pnr_max = 3

    config = {
        "cutoff": cutoff,
        "correction_cutoff": correction_cutoff,
        "pnr_max": pnr_max,
        "depth": 3,
        "genotype": "B3",
        "target_alpha": 1.0,
        "target_beta": 0.0,
        "use_dynamic_limits": True,
    }

    # 2. Create a Mock Genotype
    # We need a genotype that produces a state where truncation matters.
    # High squeezing -> large photon number -> leakage out of cutoff=12.
    # DesignB3 genotype shape: (L, D_L) + ...
    # Easier: Just use 'Legacy' or simple genotype if decoder supports it,
    # OR construct params manually and bypass decoder if possible?
    # No, jax_scoring_fn_batch uses decoder.

    # Let's use a dummy decoder or just assume DesignB3 works if we pass random genes.
    # Or import DesignB3Genotype and get a random valid genotype.

    # Pass configured depth correctly
    # Pass configured depth correctly
    # genotype_cls = DesignB3Genotype(depth=3, config=config)
    # key = jax.random.PRNGKey(42)

    # Generate random genotype
    # Shape is fixed by class
    # We create a batch of 1
    # We need range?
    # Just use random values in typical range [0, 1] for continuous params
    # Integer params might need careful handling if random.
    # DesignB3 splits into continuous and discrete.
    # For simplicity, let's construct a "Valid" random genotype manually?
    # Or just use random uniform and hope it doesn't crash (it shouldn't).

    # DesignB3 structure:
    # continuous: (active_mask, r, phi, disp_re, disp_im, BS_theta, BS_phi, Final_params...)
    # We want a state with HIGH SQUEEZING to force truncation issues.
    # Squeezing param is typically in [0, 2].

    # Let's mock the DECODER instead to return EXACT parameters we want?
    # jax_scoring_fn_batch calls `get_genotype_decoder`.
    # We can patch `get_genotype_decoder` or just rely on the fact that random params will produce some state.

    # To Ensure mismatch, we force HIGH squeezing in the decoded params.
    # But we can't easily force it without understanding the genotype mapping perfectly.
    # Better: Use the actual decoder but modify the genotype to have high value at specific index?
    # It's hard to guess the index.

    # Alternative: Use "Legacy" decoder if it is simple?
    # DesignB3 is current.

    # Let's define a Custom "Test" Decoder and register it?
    # Too complex.

    # Let's just generate a random genotype and score it.
    # If the default range includes squeezing up to 2.0 (standard),
    # cutoff 12 vs 30 will definitely show a diff.

    def make_test_genotype():
        # Manual construction of B3 genotype
        # Length calculation
        L = 8
        N = 3
        gg_len = N + N * N + 2 * N
        unique_len = 4
        # Mix params (Nodes=7) * 3
        nodes = 7
        total_len = 1 + gg_len + L * unique_len + nodes * 3 + 5

        g = np.zeros(total_len)

        # 1. Shared GG (Index 1)
        # r = 1.5. raw = arctanh(1.5 / 2.0) = 0.97
        idx = 1
        g[idx] = 0.97  # r[0]
        # Rest r=0 (raw=0)

        # Phases: Default 0 -> raw=0 -> phase=pi.
        # We want phase=0 -> raw=-5.0 (tanh->-1)
        # indices: idx + N .. idx + N + N^2
        # phases start at idx+3
        g[idx + 3 : idx + 3 + 9] = -5.0

        # Disp: 0

        # 2. Unique Params (Index 1 + 18 = 19)
        idx = 19
        for i in range(L):
            # Active (>0)
            g[idx] = 1.0
            # n_ctrl -> map to 1.
            # val = round((raw+1)/2 * 2). want 1. raw=0 -> 0.5*2=1.
            g[idx + 1] = 0.0
            # PNR -> map to 0. raw=0 -> 0.
            g[idx + 2] = -5.0
            g[idx + 3] = -5.0
            idx += 4

        return jnp.array([g])

    genotype = make_test_genotype()
    # genotype_size = ... unused

    # 3. Calculate Backend Score (Ground Truth)
    # This uses correction_cutoff (30) internally.
    # Operator: Identity for simplicity? Or GKP?
    # If Identity, ExpVal = <Psi|I|Psi> = Norm^2 = 1.0 (if no leakage).
    # But we need Probability.
    # Probability in backend is `joint_prob`.

    # We need an dummy operator
    op_size = cutoff  # Operator matches TARGET cutoff (12), as backend truncates before scoring
    operator = jnp.eye(op_size, dtype=jnp.complex64)

    fitnesses, descriptors, extras = jax_scoring_fn_batch(
        genotype,
        cutoff=cutoff,  # The nominal cutoff
        operator=operator,
        genotype_name="B3",
        genotype_config=config,
        correction_cutoff=correction_cutoff,
        pnr_max=pnr_max,
    )

    backend_prob = float(extras["joint_probability"][0])
    backend_log_prob = -float(
        fitnesses[0, 1]
    )  # Fitness[1] is -LogProb -> LogProb = -Fit

    print(f"Backend Prob (Cutoff {correction_cutoff}): {backend_prob}")
    print(f"Backend LogProb: {backend_log_prob}")

    # 4. Frontend Simulation
    # Mimic OptimizationResult.get_circuit_params -> utils.compute_state_with_jax

    # Reconstruct params using "Config" (which has cutoff=12, correction=30)
    # But usually ResultManager passes data['genotype'] to decoder.
    # And uses config.
    from src.genotypes.genotypes import get_genotype_decoder

    # ResultManager uses 'cutoff' from config by default.
    # But does it pass correction_cutoff?
    # OptimizationResult.get_circuit_params:
    # params = decoder.decode(g, cutoff) <--- USES CUTOFF (12), NOT CORRECTION_CUTOFF

    decoder = get_genotype_decoder("B3", depth=3, config=config)
    decoded_params = decoder.decode(genotype[0], cutoff)  # Frontend uses cutoff=12

    # Logic to select cutoff (Proposed Fix)
    sim_cutoff = cutoff
    # Helper to mimic runner logic
    cc = config.get("correction_cutoff")
    if cc is not None and cc > cutoff:
        sim_cutoff = cc

    # Calculate Prob with Frontend Utils
    state, frontend_prob = compute_state_with_jax(
        decoded_params, cutoff=sim_cutoff, pnr_max=pnr_max
    )

    print(f"Frontend Prob (Cutoff {sim_cutoff}): {frontend_prob}")

    # 5. Assert Match
    # This SHOULD fail if my hypothesis is correct.
    # We check if they are close.

    mismatch = abs(backend_prob - frontend_prob)
    print(f"Mismatch: {mismatch}")

    # We want to confirm they MATCH in the final fixed version.
    # So assertions should enforce equality.
    # If they fail now, it confirms the bug.
    np.testing.assert_allclose(
        frontend_prob,
        backend_prob,
        rtol=1e-5,
        err_msg="Frontend probability does not match Backend probability!",
    )


if __name__ == "__main__":
    try:
        test_frontend_backend_consistency()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
