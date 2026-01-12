import os
import pickle
import numpy as np
from typing import List, Tuple


def compute_pareto_front(candidates: List[dict]) -> List[dict]:
    """
    Identifies non-dominated solutions (Pareto front) maximizing (expect, prob).
    Expectation is stored as raw value (Metric A).
    Probability is Metric B.
    Assumption: We want to MAXIMIZE Expectation (Metric A) and MAXIMIZE Probability (Metric B).
    Wait. In existing code:
      - Valid Fit 0: -Expectation (Fitness, Maximized) -> so Higher is Better (Lower Exp)
      - Valid Fit 1: LogProb (Fitness, Maximized) -> Higher is Better (Higher Prob)

    The user wants "Pareto front in Expectation-Probability space".
    Usually we view Expectation as "Target Value" (Minimize error / Maximize Fidelity).
    Here `metric="expectation"` usually sorted by SCORE.

    Let's extract the raw fitnesses [Fit0, Fit1].
    Maximize Fit0 (-Exp) and Maximize Fit1 (LogProb).
    """
    if not candidates:
        return []

    # Sort by first objective (Fit0) descending
    # Then iterate and keep if Fit1 is better than max seen so far
    sorted_metrics = sorted(candidates, key=lambda x: x["fit0"], reverse=True)

    pareto_front = []
    max_fit1 = -float("inf")

    for c in sorted_metrics:
        if c["fit1"] > max_fit1:
            pareto_front.append(c)
            max_fit1 = c["fit1"]

    return pareto_front


def scan_results_for_seeds(
    search_dir: str = "output",
    top_k: int = 10,
    metric: str = "expectation",  # 'expectation', 'probability', or 'pareto'
    target_genotype: str = None,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Scans output directory for OptimizationResults and extracts best genotypes.
    Returns list of (genotype, design_name, score).
    """
    candidates = []

    if not os.path.exists(search_dir):
        print(f"Search dir {search_dir} does not exist. No seeds found.")
        return []

    print(f"Scanning {search_dir} for seeds (Metric: {metric})...")

    # Limit total candidates to avoid OOM
    MAX_CANDIDATES = 10000

    for root, dirs, files in os.walk(search_dir):
        if len(candidates) >= MAX_CANDIDATES * 10:  # Soft cap on raw scan
            break

        if "results.pkl" in files:
            pkl_path = os.path.join(root, "results.pkl")
            try:
                # Load Results
                # OptimizationResult object or Dict
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                # Load Config (if separate) or extract
                config = {}
                config_path = os.path.join(root, "config.json")
                if os.path.exists(config_path):
                    import json

                    with open(config_path, "r") as f:
                        config = json.load(f)
                else:
                    if hasattr(data, "config"):
                        config = data.config
                    elif isinstance(data, dict):
                        config = data.get("config", {})

                geno_name = config.get("genotype", "legacy")

                if target_genotype and geno_name != target_genotype:
                    continue

                # Extract Repertoire
                repertoire = getattr(data, "repertoire", None)
                if repertoire is None and isinstance(data, dict):
                    repertoire = data.get("repertoire")

                if repertoire is None:
                    continue

                fit = repertoire.fitnesses
                geno = repertoire.genotypes

                # Flatten
                flat_fit = fit.reshape(-1, fit.shape[-1])
                flat_geno = geno.reshape(-1, geno.shape[-1])

                # Filter valid
                # Fit 0 is -Exp. -1e9 check is valid.
                valid_mask = flat_fit[:, 0] > -1e9
                valid_fit = flat_fit[valid_mask]
                valid_geno = flat_geno[valid_mask]

                # We assume:
                # fit0 = -Expectation (Maximize)
                # fit1 = LogProb (Maximize)

                for i in range(len(valid_fit)):
                    f0 = float(valid_fit[i, 0])
                    f1 = float(valid_fit[i, 1]) if valid_fit.shape[1] > 1 else -99.0

                    # For legacy compatibility, simple score matches 'metric' arg
                    simple_score = f0
                    if metric == "probability":
                        simple_score = f1

                    candidates.append(
                        {
                            "genotype": valid_geno[i],
                            "name": geno_name,
                            "fit0": f0,
                            "fit1": f1,
                            "score": simple_score,  # For simple sort
                            "path": pkl_path,
                        }
                    )

            except Exception:
                # Failed to load
                pass

    if not candidates:
        print("No candidates found.")
        return []

    final_selection = []

    if metric == "pareto":
        # === Advanced Pareto Logic ===
        print(f"Identifying Pareto Front from {len(candidates)} candidates...")

        # 1. Compute Pareto Front
        pareto_front = compute_pareto_front(candidates)
        n_pareto = len(pareto_front)
        print(f"Found {n_pareto} Pareto optimal points.")

        if n_pareto >= top_k:
            # Case A: Too many Pareto points. Sample uniformly.
            # We sort by one axis (e.g. fit0) to ensure uniform coverage of the curve.
            # compute_pareto_front returns them sorted by fit0 descending.
            indices = np.linspace(0, n_pareto - 1, top_k, dtype=int)
            final_selection = [pareto_front[i] for i in indices]
            print(f"Selected {top_k} uniformly from Pareto front.")
        else:
            # Case B: Not enough Pareto points. Take all.
            final_selection = list(pareto_front)
            remainder = top_k - n_pareto

            if remainder > 0:
                print(
                    f"Need {remainder} more seeds. Sampling from Top 10% (by Expectation)..."
                )
                # Sort ALL candidates by Expectation (fit0)
                # Filter out those already in Pareto?
                # (Simple check: compare object id or duplicate check? candidates dicts are unique objects)
                pareto_ids = {id(c) for c in pareto_front}
                non_pareto = [c for c in candidates if id(c) not in pareto_ids]

                non_pareto.sort(key=lambda x: x["fit0"], reverse=True)

                # Top 10%
                limit_idx = max(int(len(non_pareto) * 0.1), remainder)
                top_pool = non_pareto[:limit_idx]

                if top_pool:
                    # Random sample
                    chosen = np.random.choice(
                        top_pool, size=remainder, replace=(len(top_pool) < remainder)
                    )
                    final_selection.extend(chosen)
                else:
                    # If pool empty (e.g. all were pareto), duplications? or just stop
                    pass

    else:
        # === Standard Simple Sort ===
        candidates.sort(key=lambda x: x["score"], reverse=True)
        final_selection = candidates[:top_k]

    result_tuples = []
    for c in final_selection:
        result_tuples.append((c["genotype"], c["name"], c["score"]))

    return result_tuples
