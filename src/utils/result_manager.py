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
from src.circuits.jax_runner import jax_decode_genotype


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
                "Desc_TotalPhotons": valid_desc[:, 0],
                "Desc_MaxPNR": valid_desc[:, 1],
                "Desc_Complexity": valid_desc[:, 2],
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

        # Use jax_decode_genotype
        # Note: jax_decode_genotype returns JAX arrays, we convert to numpy/list for readability
        params = jax_decode_genotype(genotype, cutoff=cutoff)

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
        x_min, x_max = np.min(lps), np.max(lps)
        y_min, y_max = np.min(exps), np.max(exps)

        pad_x = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
        pad_y = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

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
