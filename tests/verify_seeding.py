import os
import shutil
import pickle
import json
import numpy as np
from src.utils.result_scanner import scan_results_for_seeds
import jax.numpy as jnp


# Mock classes at global scope for pickle
class MockRepertoire:
    def __init__(self, fitnesses, genotypes):
        self.fitnesses = fitnesses
        self.genotypes = genotypes
        self.descriptors = jnp.zeros((genotypes.shape[0], 3))


def test_scanner_finds_results_split_config():
    """Verify scanner finds results.pkl and config.json (split structure)."""
    output_dir = "tests/temp_output/run2"
    os.makedirs(output_dir, exist_ok=True)

    # Create dummy repertoire
    # Genotype 0: D=161
    D = 161
    pop_size = 10
    genotypes = jnp.zeros((pop_size, D))
    fitnesses = jnp.zeros((pop_size, 4)) - jnp.inf
    fitnesses = fitnesses.at[0, 0].set(-0.5)

    repertoire = MockRepertoire(fitnesses, genotypes)

    config = {"genotype": "0", "modes": 3}

    # Save Config JSON
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Save Results Dictionary (mimic result_manager.save)
    data = {
        "repertoire": repertoire,
        "history": {},
        # NO config here
    }

    # Save as results.pkl
    pkl_path = os.path.join(output_dir, "results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Created mock result at {pkl_path} + config.json")

    # Scan
    seeds = scan_results_for_seeds(search_dir="tests/temp_output", top_k=5)

    print(f"Found {len(seeds)} seeds.")
    for g, name, score in seeds:
        print(f"Seed: Name={name}, Score={score}")

    # Assert
    assert len(seeds) > 0
    # Search for specific seed
    found = False
    for g, name, score in seeds:
        if name == "0" and np.isclose(score, -0.5):
            found = True
            break

    assert found

    # Cleanup
    shutil.rmtree("tests/temp_output")
    print("Test PASSED")


if __name__ == "__main__":
    test_scanner_finds_results_split_config()
