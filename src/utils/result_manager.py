import os
import pickle
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Project imports
from src.genotypes.genotypes import get_genotype_decoder

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


class SimpleRepertoire:
    """Simple container for random search results."""

    def __init__(self, genotypes, fitnesses, descriptors):
        self.genotypes = genotypes
        self.fitnesses = fitnesses
        self.descriptors = descriptors


class OptimizationResult:
    """
    Manages the results of a MOME optimization run.
    Handles saving/loading, analysis, and visualization.
    """

    def __init__(
        self,
        repertoire: Any = None,
        history: Dict[str, List[float]] = None,
        config: Dict[str, Any] = None,
        centroids: np.ndarray = None,
        history_fronts: List[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.repertoire = repertoire
        self.history = history if history is not None else {}
        self.config = config if config is not None else {}
        self.centroids = centroids
        self.history_fronts = history_fronts if history_fronts is not None else []
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

    def save(self, output_dir: str):
        """Save the optimization result to a structured directory."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Save Config (JSON)
        with open(path / "config.json", "w") as f:
            # Filter non-serializable items if any
            safe_config = {
                k: v
                for k, v in self.config.items()
                if isinstance(v, (int, float, str, bool, list, dict, type(None)))
            }
            json.dump(safe_config, f, indent=4)

        # 2. Save Repertoire & History (Pickle)
        # We save the raw repertoire object (QDax MapElitesRepertoire)
        # It contains genotypes, fitnesses, descriptors, centroids
        data = {
            "repertoire": self.repertoire,
            "history": self.history,
            "centroids": self.centroids,
            "timestamp": self.timestamp,
            # We don't save history_fronts by default to save space, unless requested?
            # User might want to re-render. Let's save it if it's not huge.
            # It's list of numpy arrays.
            "history_fronts": self.history_fronts,
        }
        with open(path / "results.pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"Saved results to {path}")

    @classmethod
    def load(cls, path: str):
        """Load an OptimizationResult from a directory."""
        path = Path(path)

        # Load Config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Load Data
        with open(path / "results.pkl", "rb") as f:
            data = pickle.load(f)

        return cls(
            repertoire=data["repertoire"],
            history=data["history"],
            config=config,
            centroids=data.get("centroids"),
            history_fronts=data.get("history_fronts"),
        )

    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for this run.
        """
        df = self.get_pareto_front()
        n_solutions = len(df)

        # Determine Generation Count
        n_gens = 0
        if "min_expectation" in self.history:
            n_chunks = len(self.history["min_expectation"])
            chunk_size = int(self.config.get("chunk_size", 100))  # Default to 100
            n_gens = n_chunks * chunk_size

        pop_size = int(self.config.get("pop_size", 1))
        n_evals = n_gens * pop_size

        # Internal dominance check for single run
        n_nondom = n_solutions
        if n_solutions > 0:
            objs = df[["Expectation", "LogProb"]].values
            is_dominated = np.zeros(len(objs), dtype=bool)
            A = objs[:, np.newaxis, :]
            B = objs[np.newaxis, :, :]
            dominates = np.all(A <= B, axis=2) & np.any(A < B, axis=2)
            is_dominated = np.any(dominates, axis=0)
            n_nondom = (~is_dominated).sum()

        return {
            "total_solutions": n_solutions,
            "total_nondominated": int(n_nondom),
            "total_generations": n_gens,
            "total_evaluations": n_evals,
        }

    def get_pareto_front(self) -> pd.DataFrame:
        """
        Retrieve the global Pareto front across all cells.
        Returns a DataFrame with objectives and descriptors.
        """
        if self.repertoire is None:
            return pd.DataFrame()

        # Extract valid solutions
        # fitnesses: (N_cells, Pareto_len, N_objs)
        fitnesses = self.repertoire.fitnesses
        descriptors = self.repertoire.descriptors
        genotypes = self.repertoire.genotypes

        # Flatten
        flat_fit = fitnesses.reshape(-1, fitnesses.shape[-1])
        flat_desc = descriptors.reshape(-1, descriptors.shape[-1])
        flat_geno = genotypes.reshape(-1, genotypes.shape[-1])

        # Filter valid
        valid_mask = flat_fit[:, 0] != -np.inf

        valid_fit = flat_fit[valid_mask]
        valid_desc = flat_desc[valid_mask]
        valid_geno = flat_geno[valid_mask]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Expectation": -valid_fit[:, 0],  # Undo negation
                "LogProb": -valid_fit[:, 1],
                "Complexity": -valid_fit[:, 2],
                "TotalPhotons": -valid_fit[:, 3],
                "Desc_Complexity": valid_desc[:, 0],  # D1 is Complexity/ActiveModes
                "Desc_MaxPNR": valid_desc[:, 1],  # D2 is MaxPNR
                "Desc_TotalPhotons": valid_desc[:, 2],  # D3 is TotalPhotons
            }
        )

        # Add genotype index for retrieval
        df["genotype_idx"] = np.arange(len(df))
        self._cached_valid_genotypes = valid_geno  # Store for retrieval

        return df

    def get_circuit_params(self, genotype_idx: int) -> Dict[str, Any]:
        """
        Reconstruct circuit parameters for a specific solution.
        genotype_idx refers to the index in the DataFrame returned by get_pareto_front().
        """
        if not hasattr(self, "_cached_valid_genotypes"):
            self.get_pareto_front()  # Refresh cache

        genotype = self._cached_valid_genotypes[genotype_idx]
        cutoff = self.config.get("cutoff", 10)

        # Use generic decoder
        algo = self.config.get("genotype", "legacy")
        # Ensure genotype is JAX array
        if jnp is not None:
            g_arr = jnp.array(genotype)
            # Extract depth from config if available
            depth = int(self.config.get("depth", 3))
            # Pass full config to decoder to respect limits (r_scale, etc.)
            params = get_genotype_decoder(algo, depth=depth, config=self.config).decode(
                g_arr, cutoff
            )
        else:
            # If JAX missing but we need to decode JAX genotype, we are in trouble.
            # But ResultManager serves analysis, maybe loaded on CPU only?
            # For now assume JAX is present if we are decoding.
            raise ImportError("JAX required to decode genotype parameters.")

        # Convert JAX arrays to native Python types for inspection
        def convert(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(params)

    def create_animation(self, filename: str = "history.gif"):
        """
        Create an animation of the optimization progress.
        Plots the Pareto front (Expectation vs LogProb) over iterations.
        """
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("imageio not installed, skipping animation.")
            return

        if not self.history_fronts:
            print(
                "No history fronts available for animation. Falling back to metrics trace."
            )
            # Fallback logic could go here, but let's assume we have fronts if requested
            return

        print(f"Creating animation from {len(self.history_fronts)} snapshots...")

        frames = []

        # Determine global bounds for consistent plotting
        all_fits = []
        for fits, _ in self.history_fronts:
            if len(fits) > 0:
                all_fits.append(fits)

        if not all_fits:
            print("No valid fitnesses in history.")
            return

        all_fits_cat = np.concatenate(all_fits, axis=0)
        # Objectives: Expectation (min), LogProb (min)
        # Fitnesses are negated: f0 = -Exp, f1 = -LogProb
        # So Exp = -f0, LogProb = -f1

        exps = -all_fits_cat[:, 0]
        lps = -all_fits_cat[:, 1]

        # Bounds with some padding
        # Filter finite values for bounds calculation
        valid_mask_fin = np.isfinite(exps) & np.isfinite(lps)
        if not np.any(valid_mask_fin):
            print("No finite values for animation bounds. Using defaults.")
            x_min, x_max = 0.0, 10.0
            y_min, y_max = 0.0, 5.0
        else:
            x_min, x_max = np.min(lps[valid_mask_fin]), np.max(lps[valid_mask_fin])
            y_min, y_max = np.min(exps[valid_mask_fin]), np.max(exps[valid_mask_fin])

        # Handle singular bounds
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05

        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)

        # Create temp dir for frames
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (fits, descs) in enumerate(self.history_fronts):
            ax.clear()

            if len(fits) > 0:
                # Convert to objectives
                curr_exps = -fits[:, 0]
                curr_lps = -fits[:, 1]
                curr_comp = -fits[:, 2]  # Complexity

                sc = ax.scatter(
                    curr_lps,
                    curr_exps,
                    c=curr_comp,
                    cmap="viridis",
                    alpha=0.7,
                    vmin=0,
                    vmax=20,  # Fixed complexity scale for stability
                )

                if i == 0:
                    plt.colorbar(sc, ax=ax, label="Complexity")

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("Negative Log10 Probability (Minimize)")
            ax.set_ylabel("Expectation Value (Minimize)")
            ax.set_title(
                f"Pareto Front Evolution (Snapshot {i + 1}/{len(self.history_fronts)})"
            )
            ax.grid(True, alpha=0.3)

            # Save frame
            frame_path = temp_dir / f"frame_{i:04d}.png"
            plt.savefig(frame_path)
            frames.append(str(frame_path))

        plt.close(fig)

        # Build GIF
        with imageio.get_writer(filename, mode="I", duration=0.2) as writer:
            for frame_path in frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        # Cleanup
        for frame_path in frames:
            os.remove(frame_path)
        temp_dir.rmdir()

        print(f"Saved animation to {filename}")


class AggregatedOptimizationResult(OptimizationResult):
    """
    Manages aggregated results from multiple optimization runs (Experiment Group).
    Merges their repertoires to form a global dataset.
    """

    def __init__(self, runs: List[OptimizationResult]):
        if not runs:
            raise ValueError("No runs provided for aggregation.")

        # Use the most recent config as the base configuration (assuming consistent params)
        # Sort runs by timestamp if possible, but they might not have it parsed.
        # We assume the list is passed in order or we just take the last one.
        base_run = runs[0]
        super().__init__(
            repertoire=None,  # We don't hold a single repertoire object
            history={},  # Merged history is complex, we skip it for now
            config=base_run.config,
            centroids=base_run.centroids,  # Assuming same centroids
            history_fronts=[],  # Aggregated history animation? Maybe later.
        )
        self.runs = runs
        self._cached_valid_genotypes = None  # Will store concatenated genotypes

    @classmethod
    def load_group(cls, experiment_path: str):
        """
        Load all sub-runs from an experiment directory.
        Expects subdirectories containing results.pkl.
        """
        experiment_path = Path(experiment_path)
        sub_runs = []

        # Walk subdirectories
        for item in os.listdir(experiment_path):
            sub_path = experiment_path / item
            if sub_path.is_dir() and (sub_path / "results.pkl").exists():
                try:
                    run = OptimizationResult.load(str(sub_path))
                    sub_runs.append(run)
                except Exception as e:
                    print(f"Skipping corrupt run {sub_path}: {e}")

        if not sub_runs:
            raise ValueError(f"No valid runs found in {experiment_path}")

        print(f"Aggregated {len(sub_runs)} runs from {experiment_path}")
        return cls(sub_runs)

    def get_pareto_front(self) -> pd.DataFrame:
        """
        Aggregates valid solutions from ALL runs and computes a Global Pareto Front (conceptually).
        Actually, aligns with user request to "load ALL THE SOLUTIONS".
        We simply concatenate all valid solutions into one DataFrame.
        """
        all_dfs = []
        all_genotypes = []

        all_genotypes = []

        for run in self.runs:
            # We use the base class method to extract valid data from THIS run
            # But we need to be careful about _cached_valid_genotypes

            # Manually extract arrays to avoid side effects on the sub-run objects if needed
            # But calling get_pareto_front() on sub-run is safe.
            df = run.get_pareto_front()

            if df.empty:
                continue

            # We need to map the global index back to (run_index, local_index)
            # Or just concatenate genotypes and let get_circuit_params handle it by index.

            # Store genotypes for this batch
            all_genotypes.append(run._cached_valid_genotypes)

            # Adjust index? No, we will rebuild the index after concat.
            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Concatenate DataFrames
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Re-assign genotype_idx to be continuous 0..N
        final_df["genotype_idx"] = np.arange(len(final_df))

        # Concatenate Genotypes
        # Each item in all_genotypes is (N_valid, D)
        # We assume consistent D across runs (checked via config usually)
        # Pad heterogeneous genotypes if necessary
        if all_genotypes:
            max_d = max(g.shape[1] for g in all_genotypes)
            padded_genotypes = []
            for g in all_genotypes:
                n, d = g.shape
                if d < max_d:
                    padding = ((0, 0), (0, max_d - d))
                    if jnp is not None and isinstance(g, jnp.ndarray):
                        g_pad = jnp.pad(g, padding, mode="constant")
                    else:
                        g_pad = np.pad(g, padding, mode="constant")
                    padded_genotypes.append(g_pad)
                else:
                    padded_genotypes.append(g)

            # Concatenate
            if jnp is not None:
                # Convert all to JAX if needed
                padded_genotypes = [
                    jnp.array(g) if not isinstance(g, jnp.ndarray) else g
                    for g in padded_genotypes
                ]
                self._cached_valid_genotypes = jnp.concatenate(padded_genotypes, axis=0)
            else:
                self._cached_valid_genotypes = np.concatenate(padded_genotypes, axis=0)
        else:
            if jnp is not None:
                self._cached_valid_genotypes = jnp.array([])
            else:
                self._cached_valid_genotypes = np.array([])

        # Post-Processing: Compute Global Dominance
        # Objectives: Expectation (min), LogProb (min)
        # We assume minimizing both.
        # N_points x 2
        objs = final_df[["Expectation", "LogProb"]].values

        is_dominated = np.zeros(len(objs), dtype=bool)

        # Simple O(N^2) dominance check (N is usually reasonably small for Pareto fronts ~ 100-1000s)
        # If N is huge, this might be slow, but for MOME analysis it's usually fine.
        # A dominates B if A.exp <= B.exp and A.prob <= B.prob and (A.exp < B.exp or A.prob < B.prob)
        # Here we deal with floats, so use epsilon?

        # Vectorized check is faster
        # Or use simple loop for clarity/safety against NaNs

        # For now, let's mark it.
        # "GlobalDominant" = True if not dominated by any other point in the aggregated set.

        # Sort by first objective to speed up?
        # Let's do a naive pass for robustness.

        N = len(objs)
        if N > 0:
            # Broadcast comparison? N=1000 => 1M ops, fine.
            # A (N, 1, 2) <= B (1, N, 2)
            # all(<=) and any(<)
            A = objs[:, np.newaxis, :]
            B = objs[np.newaxis, :, :]

            dominates = np.all(A <= B, axis=2) & np.any(A < B, axis=2)
            # dominates[i, j] means i dominates j
            # is_dominated[j] = any(dominates[:, j])
            is_dominated = np.any(dominates, axis=0)

        final_df["GlobalDominant"] = ~is_dominated

        return final_df

    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Aggregated statistics across all sub-runs.
        """
        df = self.get_pareto_front()
        n_solutions = len(df)

        # Global Dominant is computed in get_pareto_front
        if "GlobalDominant" in df.columns:
            n_nondom = df["GlobalDominant"].sum()
        else:
            n_nondom = n_solutions

        total_gens = 0
        total_evals = 0

        for run in self.runs:
            stats = run.get_experiment_stats()
            total_gens += stats["total_generations"]
            total_evals += stats["total_evaluations"]

        return {
            "total_solutions": n_solutions,
            "total_nondominated": int(n_nondom),
            "total_generations": total_gens,
            "total_evaluations": total_evals,
        }

    def get_circuit_params(self, genotype_idx: int) -> Dict[str, Any]:
        """
        Retrieve parameters from the aggregated genotype cache.
        """
        # Logic matches base class, but uses the aggregated _cached_valid_genotypes
        return super().get_circuit_params(genotype_idx)
