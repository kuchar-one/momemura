import pickle
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from src.genotypes.genotypes import get_genotype_decoder

# User config
TARGET_DIR = "output/20251217-090828_c25_p500_i2000"
CHECKPOINT = os.path.join(TARGET_DIR, "checkpoint_latest.pkl")


def inspect():
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        return

    print(f"Loading {CHECKPOINT}...")
    with open(CHECKPOINT, "rb") as f:
        data = pickle.load(f)

    repertoire = data["repertoire"]
    print(f"Repertoire loaded. Keys: {data.keys()}")

    # Extract fitnesses
    # fitnesses shape: (N_centroids, Pareto_size, 4)
    # 0: -Expectation
    # 1: -LogProb
    # 2: -Complexity
    # 3: -Photons

    fits = repertoire.fitnesses
    flat_fits = fits.reshape(-1, 4)

    # Find active solutions (fitness > -inf)
    valid_mask = flat_fits[:, 0] > -np.inf
    valid_fits = flat_fits[valid_mask]

    if len(valid_fits) == 0:
        print("No valid solutions found in repertoire!")
        return

    print(f"Found {len(valid_fits)} valid solutions.")

    # Best Expectation (Maximize fitness[0])
    best_idx_valid = np.argmax(valid_fits[:, 0])
    best_fit = valid_fits[best_idx_valid]
    best_exp = -best_fit[0]

    print(f"Best Expectation: {best_exp}")
    print(f"Best Fitness Vector: {best_fit}")

    # Get Genotype
    # We need the index in the original flattened array to find the genotype
    # Or just iterate?
    # flat_fits corresponds to flat_genotypes?
    # repertoire.genotypes shape?

    gens = repertoire.genotypes
    # (N, P, D)
    flat_gens = gens.reshape(-1, gens.shape[-1])

    # We need to map `best_idx_valid` back to `flat_fits` index
    # valid_mask is boolean array.
    # np.where(valid_mask)[0] gives indices.

    global_indices = np.where(valid_mask)[0]
    best_global_idx = global_indices[best_idx_valid]

    best_genotype = flat_gens[best_global_idx]

    # Decode
    print("Decoding best genotype...")
    # Matches run_mome config?
    # "Genotype A selected"
    # "D=155"
    # Config from checkpoint might help if available, else assume defaults or try A
    genotype_name = "A"
    depth = 3

    decoder = get_genotype_decoder(genotype_name, depth=depth)
    params = decoder.decode(best_genotype, cutoff=25)

    print("Best Params:")
    # Print summary
    print(f"  Homodyne X: {params['homodyne_x']}")
    print(f"  Homodyne Window: {params['homodyne_window']}")
    print(
        f"  Final Gauss: r={params['final_gauss']['r']:.20f}, phi={params['final_gauss']['phi']:.4f}, disp={params['final_gauss']['disp']}"
    )

    print("  Leaf Active:")
    print(params["leaf_active"])

    print("  Mix Params (Theta, Phi, Varphi):")
    print(params["mix_params"])

    print("\n  Active Leaf 5 Params:")
    # leaf_params is dict of arrays (8, ...)
    lp = params["leaf_params"]
    # Check index 5
    idx = 5
    print(f"    n_ctrl: {lp['n_ctrl'][idx]}")
    print(f"    tmss_r: {lp['tmss_r'][idx]}")
    print(f"    us_phase: {lp['us_phase'][idx]}")
    # disp_s, disp_c?
    print(f"    disp_s: {lp['disp_s'][idx]}")


if __name__ == "__main__":
    inspect()
