import os
import pickle
import numpy as np
from typing import List, Tuple


def scan_results_for_seeds(
    search_dir: str = "output",
    top_k: int = 10,
    metric: str = "expectation",  # 'expectation' or 'probability'
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Scans output directory for OptimizationResults and extracts best genotypes.
    Returns list of (genotype, design_name, score).
    """
    candidates = []

    if not os.path.exists(search_dir):
        print(f"Search dir {search_dir} does not exist. No seeds found.")
        return []

    print(f"Scanning {search_dir} for seeds...")

    for root, dirs, files in os.walk(search_dir):
        if "result.pkl" in files:
            pkl_path = os.path.join(root, "result.pkl")
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                # We need to extract:
                # 1. Genotype Name (from config)
                # 2. Repertoire (fitnesses, genotypes)

                # Check structure (OptimizationResult object or dict?)
                # It's usually an object, but pickle loads it as such.
                if hasattr(data, "config"):
                    config = data.config
                elif isinstance(data, dict) and "config" in data:
                    config = data["config"]
                else:
                    continue  # Unknown format

                # Extract Genotype Name
                # run_mome uses "genotype_name" in adapter but config stores "mode"?
                # Let's check run_mome.py line 1067...
                # config = { "mode": mode, ..., "backend": backend }
                # WAIT. It does NOT explicitly store genotype name in config?!
                # Checking run_mome.py...
                # Nope, I don't see "genotype" in the saved config dict in `run` function.
                # Mistake in previous impl?
                # User config: genotype arg is separate.
                # I should add "genotype": genotype to config in line 1067 of run_mome.py for future.
                # But for existing results, maybe we can't key them?
                # Assuming "legacy" if missing? Or "genotype" might be in there if I added it?
                # I did NOT add it in previous edits.
                # BUT `OptimizationResult` stores `config`.

                # If genotype is missing, we assumes 'Legacy' or maybe heuristic?
                # Actually, `run_mome.py` default was "legacy".
                geno_name = config.get("genotype", "legacy")

                # Extract Candidates from Repertoire
                repertoire = getattr(data, "repertoire", None)
                if repertoire is None and isinstance(data, dict):
                    repertoire = data.get("repertoire")

                if repertoire is None:
                    continue

                # Repertoire has .genotypes, .fitnesses
                # QDax repertoire: fitnesses shape (N, Pareto, Objs) or (N, Objs)
                # SimpleRepertoire: (N, 1, Objs)

                fit = repertoire.fitnesses
                geno = repertoire.genotypes

                # Flatten
                flat_fit = fit.reshape(-1, fit.shape[-1])
                flat_geno = geno.reshape(-1, geno.shape[-1])

                # Filter valid
                valid_mask = flat_fit[:, 0] > -1e9  # valid exp

                valid_fit = flat_fit[valid_mask]
                valid_geno = flat_geno[valid_mask]

                # Objectives: [Expectation, -LogProb, -Comp, -Photons]
                # We want to maximize Expectation (Obj 0)
                # Or Maximize Prob (Obj 1)

                for i in range(len(valid_fit)):
                    score = -1e99
                    if metric == "expectation":
                        score = valid_fit[i, 0]
                    elif metric == "probability":
                        score = valid_fit[i, 1]

                    candidates.append(
                        {
                            "genotype": valid_geno[i],
                            "name": geno_name,
                            "score": float(score),
                            "path": pkl_path,
                        }
                    )

            except Exception as e:
                print(f"Failed to load {pkl_path}: {e}")
                continue

    # Sort and take top k
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[:top_k]

    result_tuples = []
    for c in best:
        # print(f"Found seed in {c['path']}: {c['name']} score={c['score']:.4f}")
        result_tuples.append((c["genotype"], c["name"], c["score"]))

    return result_tuples
