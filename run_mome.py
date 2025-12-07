# run_mome.py
"""
Runner for Hanamura MOME optimization using QDax.

Based on QDax MOME tutorial: https://qdax.readthedocs.io/en/latest/examples/mome/

Provides:
 - decode_genotype(...) : maps genotype vector -> gaussian block params
 - gaussian_block_builder(...) : builds GaussianHeraldCircuit, returns single-mode state vec & prob
 - HanamuraMOME adapter (batch scoring function)
 - run() entrypoint: uses QDax MOME if available; otherwise runs random-search baseline.

Usage:
    python run_mome.py --mode qdax   # requires jax + qdax
    python run_mome.py --mode random # quick smoke-run (default)
"""

import os
import sys
import time
import argparse
import numpy as np
from functools import partial
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import matplotlib

# === Project imports ===
from src.genotypes.genotypes import get_genotype_decoder

from src.simulation.cpu.composer import Composer, SuperblockTopology
from src.simulation.cpu.circuit import GaussianHeraldCircuit
from src.utils.cache_manager import CacheManager

# Set memory allocator to avoid fragmentation OOMs
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Check for low-mem flag early
LOW_MEM = "--low-mem" in sys.argv

# Enable JAX x64 mode unless low-mem is requested
# We use float32 by default for performance in all modes (User Policy),
# so we explicitly disable x64.
os.environ["JAX_ENABLE_X64"] = "False"

if LOW_MEM:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print("Low-memory mode enabled: preallocation disabled.")


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import jax

    # JAX Config
    # Use float32 for speed (User request)
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

# Local defaults
DEFAULT_CUTOFF = 6
GLOBAL_CACHE = CacheManager(cache_dir="./cache", size_limit_bytes=1024 * 1024 * 512)


class SimpleRepertoire:
    """Simple container for random search results."""

    def __init__(self, genotypes, fitnesses, descriptors):
        self.genotypes = genotypes
        self.fitnesses = fitnesses
        self.descriptors = descriptors


# -------------------------
# Genotype -> params decoder
# -------------------------


# Genotype Decoder is imported at top level


# -------------------------
# Gaussian block builder
# -------------------------
def gaussian_block_builder_from_params(
    params: Dict[str, Any], cutoff: int = DEFAULT_CUTOFF, backend: str = "thewalrus"
) -> Tuple[np.ndarray, float]:
    """
    Build GaussianHeraldCircuit and return *single-mode* signal vector (length cutoff, complex, normalized)
    and herald probability. This function enforces n_signal == 1 for pure pipeline.
    Raises ValueError when herald probability is too small or output isn't a pure vector.
    """
    # enforce single signal
    n_signal = 1
    n_control = int(params["n_control"])

    circ = GaussianHeraldCircuit(
        n_signal=n_signal,
        n_control=n_control,
        tmss_squeezing=list(params.get("tmss_squeezing", [])),
        us_params=params.get("us_params", None),
        uc_params=params.get("uc_params", None),
        U_s=params.get("U_s", None),
        U_c=params.get("U_c", None),
        disp_s=params.get("disp_s", None),
        disp_c=params.get("disp_c", None),
        mesh="rectangular",
        hbar=2.0,
        cache_enabled=True,
        backend=backend,
    )
    circ.build()
    state, prob = circ.herald(
        pnr_outcome=params.get("pnr_outcome", None),
        signal_cutoff=int(params.get("signal_cutoff", cutoff)),
        check_purity=True,  # ensure walrus can compute pure amplitudes if present
    )

    # Expect walrus to return normalized complex amplitudes for signal basis (pure amplitudes).
    # state may be: 1D amplitude vector (len=cutoff) OR shaped array for multi-mode signals.
    arr = np.asarray(state)

    # For pure pipeline we expect 1D amplitude vector for the single signal mode.
    if arr.ndim == 0:
        # empty or trivial — treat as invalid
        raise ValueError("Empty herald output")
    if arr.ndim > 1:
        # multi-mode amplitude returned (rare because we set n_signal=1) — error
        raise ValueError("Herald produced multi-mode amplitudes but n_signal==1")

    vec = arr.reshape(-1)
    # normalize (numeric safety)
    norm = np.linalg.norm(vec)
    if norm <= 0 or prob <= 1e-12:
        # too-small probability or zero amplitude -> invalid for pure pipeline
        raise ValueError(f"Herald returned zero or tiny probability: prob={prob:.3e}")

    vec = vec / norm

    # ensure length = cutoff
    if vec.size < cutoff:
        vec = np.concatenate([vec, np.zeros(cutoff - vec.size, dtype=complex)])
    elif vec.size > cutoff:
        vec = vec[:cutoff]

    return vec.astype(complex), float(prob)


# -------------------------
# Adapter: HanamuraMOMEAdapter
# -------------------------
class HanamuraMOMEAdapter:
    """Adapter for QDax MOME optimization."""

    def __init__(
        self,
        composer: Composer,
        topology: SuperblockTopology,
        operator: np.ndarray,
        cutoff: int = DEFAULT_CUTOFF,
        mode: str = "pure",
        homodyne_resolution: float = 0.01,
        backend: str = "thewalrus",
        genotype_name: str = "legacy",
        genotype_config: Dict[str, Any] = None,
        correction_cutoff: int = None,
    ):
        """
        Initialize the MOME Adapter.

        Args:
            composer (Composer): Composer instance for circuit evaluation.
            topology (SuperblockTopology): Topology of the superblock.
            operator (np.ndarray): Target operator for fidelity calculation.
            cutoff (int): Simulation cutoff dimension (N).
            mode (str): "pure" or "mixed" (currently only "pure" is supported).
            homodyne_resolution (float): Resolution for homodyne measurement.
            backend (str): "thewalrus" or "jax".
            genotype_name (str): Name of the genotype to use (e.g., "legacy", "A").
            genotype_config (dict): Configuration dictionary for the genotype.
            correction_cutoff (int): Higher cutoff for leakage correction.
        """
        self.composer = composer
        self.topology = topology
        self.operator = operator
        self.cutoff = int(cutoff)
        self.mode = mode
        self.homodyne_resolution = homodyne_resolution
        self.backend = backend
        self.genotype_name = genotype_name
        self.genotype_config = genotype_config or {}
        self.correction_cutoff = correction_cutoff

        # Instantiate decoder for local use
        self.decoder = get_genotype_decoder(genotype_name, config=self.genotype_config)

        if hasattr(composer, "cutoff") and composer.cutoff != self.cutoff:
            raise ValueError("composer.cutoff != operator cutoff")

        # Reuse executor to avoid overhead
        # Limit max_workers to avoid excessive contention
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def evaluate_one(self, genotype: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single genotype and return metrics."""
        # Use new decoder
        # Only support "Legacy" format dict output needed for gaussian_block_builder?
        # WAIT. gaussian_block_builder_from_params expects specific keys ("n_control", "tmss_squeezing", etc.)
        # The new BaseGenotype.decode returns "leaf_params" structure which is DIFFERENT from legacy "flat" structure used in `run_mome.py`.
        # This is a critical mismatch!

        # We need to adapt the new dict structure to what `gaussian_block_builder_from_params` expects,
        # OR update `gaussian_block_builder_from_params` to understand the new structure.
        # But `gaussian_block_builder_from_params` builds a `GaussianHeraldCircuit`.

        # In JAX pipeline (jax_runner), we construct the circuit directly via JAX primitives (jax_get_heralded_state).
        # In "random" mode (using HanamuraMOMEAdapter.evaluate_one), it uses `gaussian_block_builder_from_params` which uses `thewalrus`.

        # Do we need to support "random" mode with "thewalrus" backend for NEW genotypes?
        # The user implies full integration.
        # BUT maintaining `thewalrus` builder for all new genotypes is complex because `GaussianHeraldCircuit` might not support broadcasting etc. easily.
        # ACTUALLY, the JAX runner now supports all genotypes.
        # Ideally, `run_mome.py` should prefer JAX backend even for "random" mode if available, or we update the adapter to use JAX for evaluate_one too?

        # Current `evaluate_one` uses `gaussian_block_builder_from_params` -> `GaussianHeraldCircuit` -> standard pipeline.
        # If we switch to new genotypes, `decode_genotype` is gone.

        # Temporary Fix: Map params from new keys to legacy keys if possible?
        # Legacy Genotype class returns:
        # { "homodyne_x", "mix_params", ..., "leaf_params": { "n_ctrl", ... } }

        # `gaussian_block_builder_from_params` needs: "n_signal", "n_control", "tmss_squeezing", "us_params", etc.

        # It seems `gaussian_block_builder_from_params` was built around the SPECIFIC legacy decoding.
        # The new designs (A, B, C) produce `leaf_params` arrays.
        # `GaussianHeraldCircuit` (the Python class) expects lists of params.

        # If we want to use the Python/Walrus backend with new genotypes, we need to bridge this.
        # However, for `mode="qdax"`, it uses `jax_scoring_fn_batch` which works fine (it uses JAX runner).
        # For `mode="random"`, `evaluate_one` is called on CPU.

        # If JAX is available, `evaluate_one` is mostly used for debugging or non-JAX environments?
        # But `run_mome` uses `jax_scoring_fn_batch` IF `self.backend == "jax"`.
        # So for high performance, we use JAX.

        # If we use new genotypes, we should probably enforce/prefer JAX backend.
        # Implementing the bridge for `thewalrus` backend for all complex tied designs is tedious and maybe unnecessary if JAX is the future.

        # The decoded dict has "leaf_params" with arrays of shape (8, ...).
        # We need to construct `GaussianHeraldCircuit` for each leaf?
        # `gaussian_block_builder_from_params` builds ONE block.
        # The topology evaluation loops over leaves.

        # This shows `evaluate_one` relies on the old "single genotype encodes single block params + global" assumption?
        # WAIT. The legacy code was:
        # `params = decode_genotype(...)`
        # `vec, prob = gaussian_block_builder_from_params(params)`
        # `fock_vecs = [vec for _ in range(n_leaves)]`
        # This implies standard "random" mode ONLY supported the case where ALL leaves are identical?
        # Yes, line 364: `fock_vecs = [vec.copy() for _ in range(n_leaves)]`.
        # The legacy random search assumed a single block type replicated.

        # The NEW genotypes allow heterogeneous leaves (Design A, B).
        # So `evaluate_one` logic of "decode once -> build one block -> replicate" is WRONG for Design A/B.

        # If we want to support Design A/B in `evaluate_one`, we must:
        # 1. Decode genotype -> get params for EACH leaf.
        # 2. Build circuit for EACH leaf.

        # But `gaussian_block_builder_from_params` takes a dict for one block.
        # We can reconstruct that dict 8 times.

        # Let's write an adapter helper `_params_to_legacy_dict(new_params, leaf_idx)`.

        # If backend is JAX, `evaluate_one` is NOT used in `scoring_fn_batch` (optimized path).
        # It IS used in `scoring_fn_batch` fallback loop (lines 508+).

        # For this task, I will implement a JAX-based `evaluate_one` if JAX is installed, or try to adapt.
        # Actually, if we are in JAX mode, we should just use JAX function even for single evaluation?
        # Yes.

        # But `evaluate_one` returns a dict of metrics (expectation, complexity, etc.).
        # JAX scoring function returns fitness array.
        # We need to map back.

        # Let's rely on JAX backend primarily.

        # Refactoring `evaluate_one`:

        # Convert genotype to JAX
        g_jax = jnp.array(genotype)

        # Use decoder
        params = self.decoder.decode(g_jax, self.cutoff)

        # To reuse `gaussian_block_builder_from_params`, we need conversion.
        # But maybe we can skip that and use `jax_get_heralded_state`?
        # If JAX is available, use JAX logic to compute state.

        if jax is not None:
            # Use JAX implementation of the block builder
            from src.simulation.jax.runner import jax_get_heralded_state

            # Map JAX params to leaf_params
            leaf_params = params["leaf_params"]
            # leaves: (8, ...)

            # vmap to get all leaves
            leaf_vecs, leaf_probs, leaf_modes, leaf_max_pnrs, leaf_total_pnrs = (
                jax.vmap(partial(jax_get_heralded_state, cutoff=self.cutoff))(
                    leaf_params
                )
            )

            # Convert to numpy for topology eval?
            # Topology eval uses `self.topology.evaluate_topology` which expects numpy arrays and uses Composer (walrus/piquasso).
            # Mixing JAX output with Walrus composer is inefficient but works (convert to np).

            # Unused analysis variables
            # fock_vecs, p_heralds = ...
            # Homodyne
            h_x = float(params["homodyne_x"])
            h_win = params["homodyne_window"]  # might be float or None (if -1)
            if isinstance(h_win, (int, float, jnp.ndarray)) and h_win < 1e-3:
                h_win = None
            else:
                h_win = float(h_win)

            # Mix params
            # New genotypes have "mix_params" array (7, 3).
            # Legacy topology evaluation assumes "mix_theta" constant?
            # Line 384: `theta=params.get("mix_theta", ...)`
            # The existing topology eval `evaluate_topology` assumes homogeneous mixing?

            # Let's check `src/simulation/cpu/composer.py` `evaluate_topology`.
            # If `evaluate_topology` doesn't support heterogeneous mixing arguments, we can't fully support Design A/B accurately with the old Python `SuperblockTopology` class.
            # BUT `jax_runner.py` DOES support heterogeneous mixing in `jax_superblock`.

            # So, for accurate evaluation of new genotypes, we MUST use `jax_superblock` logic, not the old `composer.py` logic.
            # The old logic is insufficient for Design A (heterogeneous).

            # Thus, `evaluate_one` should call `jax_superblock` effectively if we want consistency.
            # `jax_scoring_fn_batch` (in `jax_runner.py`) does exactly this.

            # So `evaluate_one` should wrapper `jax_scoring_fn_batch` for a batch of 1?
            # Yes! That guarantees consistency.

            # Prepare batch 1
            g_batch = g_jax[None, :]
            op_batch = jnp.array(self.operator)

            from src.simulation.jax.runner import jax_scoring_fn_batch

            fitnesses, descriptors = jax_scoring_fn_batch(
                g_batch,
                self.cutoff,
                op_batch,
                genotype_name=self.genotype_name,
                genotype_config=self.genotype_config,
                correction_cutoff=self.correction_cutoff,
            )

            # Extract metrics
            fit = fitnesses[0]
            desc = descriptors[0]

            # fitness = [-exp, -logP, -comp, -photons]
            exp_val = -float(fit[0])
            log_prob = -float(fit[1])
            joint_prob = 10 ** (-log_prob)
            complexity = -float(fit[2])
            total_ph = -float(fit[3])

            # descriptors = [active_modes, max_pnr, total_photons]
            per_det_max = float(desc[1])

            return {
                "expectation": exp_val,
                "joint_prob": joint_prob,
                "log_prob": log_prob,
                "complexity": complexity,
                "total_measured_photons": total_ph,
                "per_detector_max": per_det_max,
                "homodyne_x": h_x,  # extra info
            }

        else:
            raise RuntimeError("JAX is required for generalized genotypes.")

        # If JAX is missing, we fail. The project relies on JAX now.

    def scoring_fn_batch(
        self,
        genotypes: np.ndarray,
        rng_key: Any,  # jax.random.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Batched scoring function.
        """
        batch_size = genotypes.shape[0]

        if self.backend == "jax" and jax is not None:
            # JAX batch execution
            from src.simulation.jax.runner import jax_scoring_fn_batch

            g_jax = jnp.array(genotypes)
            op_jax = jnp.array(self.operator)

            with jax.profiler.TraceAnnotation("jax_scoring_fn_batch"):
                fitnesses_jax, descriptors_jax = jax_scoring_fn_batch(
                    g_jax,
                    self.cutoff,
                    op_jax,
                    self.genotype_name,
                    self.genotype_config,
                    self.correction_cutoff,
                )
                fitnesses_jax.block_until_ready()
                descriptors_jax.block_until_ready()

            fitnesses = fitnesses_jax
            descriptors = descriptors_jax
            extras = [{} for _ in range(batch_size)]
            return fitnesses, descriptors, extras

        # Fallback to serial (for non-JAX backend or debug)
        fitnesses = np.zeros((batch_size, 4))
        descriptors = np.zeros((batch_size, 3))
        extras = []

        def eval_single(i):
            try:
                metrics = self.evaluate_one(genotypes[i, :])
                return i, metrics, None
            except Exception as e:
                return i, None, str(e)

        # Use persistent executor
        futures = [self.executor.submit(eval_single, i) for i in range(batch_size)]
        results = [f.result() for f in futures]

        for i, metrics, error in results:
            if error:
                # invalid genotype
                fitnesses[i, :] = -np.inf
                descriptors[i, :] = np.array([-9999.0, -9999.0, -9999.0], dtype=float)
                extras.append({"error": error})
                continue

            # Objectives (QDax maximizes all)
            # 1. Expectation (maximize) -> f_expect
            # 2. Log Prob (minimize -logP -> maximize logP) -> f_prob
            # 3. Complexity (minimize) -> -complexity
            # 4. Total Photons (minimize) -> -photons

            f_expect = metrics["expectation"]
            f_prob = metrics["log_prob"]  # usage suggests this is -log10(P), wait.
            # In evaluate_one I set log_prob = -math.log10(prob). This is positive for tiny prob.
            # Minimizing "log_prob" (which is usually decreasing with P).
            # If we want to maximize probability, we minimize -log(P).
            # The metrics["log_prob"] returned by existing evaluate_one is usually -log10(P).
            # In my JAX update I set log_prob = metrics["log_prob"]...
            # In original: f_prob = metrics["log_prob"].
            # fitness[1] = -f_prob.
            # If f_prob = -log10(P) (positive), then fitness = log10(P) (negative).
            # Maximizing fitness (negative number) -> Maximizing P. Correct.

            f_prob = metrics["log_prob"]
            f_complex = float(metrics["complexity"])
            f_photons = float(metrics["total_measured_photons"])

            fitnesses[i, :] = np.array([f_expect, -f_prob, -f_complex, -f_photons])

            # Descriptors (Match JAX runner order: [Complex, Max, Total])
            d_total = float(metrics["total_measured_photons"])
            d_max = float(metrics["per_detector_max"])
            d_complex = float(metrics["complexity"])

            descriptors[i, :] = np.array([d_complex, d_max, d_total])
            extras.append(metrics)

        return fitnesses, descriptors, extras

        def eval_single(i):
            try:
                metrics = self.evaluate_one(genotypes[i, :])
                return i, metrics, None
            except Exception as e:
                return i, None, str(e)

        # Use persistent executor
        futures = [self.executor.submit(eval_single, i) for i in range(batch_size)]
        results = [f.result() for f in futures]

        for i, metrics, error in results:
            if error:
                # invalid genotype — mark with -inf fitness so the repertoire cell stays empty
                # print(f"DEBUG: Genotype invalid: {error}") # Uncomment for debugging
                fitnesses[i, :] = -np.inf
                # use a descriptor sentinel outside grid bounds so it cannot populate centroids
                descriptors[i, :] = np.array([-9999.0, -9999.0, -9999.0], dtype=float)
                extras.append({"error": error})
                continue

            # Objectives
            # 1. Expectation (maximize) -> QDax maximizes fitness, so use f_expect directly
            f_expect = metrics["expectation"]
            # 2. Log Prob (minimize -logP) -> QDax maximizes, so use -(-logP) = logP? No.
            # We want to minimize -log(P). So we want to maximize log(P).
            # metrics["log_prob"] is -log10(P). We want to minimize it.
            # So fitness = -metrics["log_prob"].
            f_prob = metrics["log_prob"]
            # 3. Complexity (minimize) -> fitness = -complexity
            f_complex = float(metrics["complexity"])
            # 4. Total Photons (minimize) -> fitness = -photons
            f_photons = float(metrics["total_measured_photons"])

            fitnesses[i, :] = np.array(
                [f_expect, -f_prob, -f_complex, -f_photons]
            )  # QDax maximizes fitness

            # Descriptors (for map)
            # Match JAX Runner: [Complexity, Max_PNR, Total_Photons]
            # 1. Complexity (Active Modes)
            d_complex = float(metrics["complexity"])
            # 2. Max Photons (Max PNR)
            d_max = float(metrics["per_detector_max"])
            # 3. Total Photons
            d_total = float(metrics["total_measured_photons"])

            descriptors[i, :] = np.array([d_complex, d_max, d_total])
            extras.append(metrics)

        # Convert extras to dict of arrays (for QDax compatibility if needed, though QDax ignores extras usually)
        # We return the list of extras for random mode usage.
        return fitnesses, descriptors, extras


# -------------------------
# Custom Metrics (JAX)
# -------------------------
def custom_metrics(repertoire):
    """Custom metrics for progress tracking."""
    if jnp is None:
        return {}

    # Coverage
    # repertoire.fitnesses shape: (N, Pareto, Objs)
    # A cell is filled if ANY pareto solution is valid (not -inf)
    # We check the first objective of the first pareto front item, or better, check if any in pareto dim is valid.
    # Usually checking [:, 0, 0] is enough if we fill sequentially, but let's be robust.
    # valid_mask = jnp.any(repertoire.fitnesses[..., 0] != -jnp.inf, axis=1)
    # Actually, let's just check the first slot of the Pareto front for simplicity and speed, assuming contiguous filling.
    valid_mask = repertoire.fitnesses[:, 0, 0] != -jnp.inf
    coverage = jnp.sum(valid_mask) / repertoire.fitnesses.shape[0]

    # Best expectation (min expectation = max fitness[0])
    # fitness[0] is -expectation
    # We want max over all cells and all pareto fronts.
    # Flatten pareto dim
    flat_fitnesses = repertoire.fitnesses.reshape(-1, repertoire.fitnesses.shape[-1])
    flat_valid = flat_fitnesses[:, 0] != -jnp.inf
    # We want the minimum expectation value.
    # fitness[0] = -expectation.
    # max(fitness[0]) = max(-expectation) = -min(expectation).
    # So min(expectation) = -max(fitness[0]).
    min_expectation = -jnp.max(jnp.where(flat_valid, flat_fitnesses[:, 0], -jnp.inf))

    # Best log prob (min log prob = max fitness[1])
    # fitness[1] is -log_prob
    min_log_prob = -jnp.max(jnp.where(flat_valid, flat_fitnesses[:, 1], -jnp.inf))

    # Max probability
    # log_prob = -log10(P) -> P = 10^(-log_prob)
    # We want max P. P is maximized when log_prob is minimized.
    # So max_prob = 10^(-min_log_prob)
    max_probability = 10.0 ** (-min_log_prob)

    return {
        "coverage": coverage,
        "min_expectation": min_expectation,
        "min_log_prob": min_log_prob,
        "max_probability": max_probability,
    }


# -------------------------
# Visualization
# -------------------------
def plot_mome_results(repertoire, metrics, filename="mome_results.png"):
    """Plot MOME results: Pareto fronts, heatmaps, coverage."""
    # Extract valid solutions
    # fitnesses shape is (N_cells, Pareto_len, N_objs) = (270, 5, 4)
    # We flatten the first two dimensions to plot all solutions
    flat_fitnesses = repertoire.fitnesses.reshape(-1, repertoire.fitnesses.shape[-1])
    flat_descriptors = repertoire.descriptors.reshape(
        -1, repertoire.descriptors.shape[-1]
    )

    valid_mask = flat_fitnesses[:, 0] != -np.inf

    fitnesses = flat_fitnesses[valid_mask]
    descriptors = flat_descriptors[valid_mask]

    # Convert fitnesses back to objectives (undo negation)
    # f0 = -expv -> expv = -f0
    # f1 = -log_prob -> log_prob = -f1
    objectives = -fitnesses

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Global Pareto Front (Expectation vs Log Prob)
    # Note: We want to minimize Expectation and minimize Log Prob (maximize Prob)
    ax = axes[0, 0]
    sc = ax.scatter(
        objectives[:, 1],
        objectives[:, 0],
        c=objectives[:, 2],
        cmap="viridis",
        alpha=0.7,
    )
    ax.set_xlabel("Negative Log10 Probability (Minimize)")
    ax.set_ylabel("Expectation Value (Minimize)")
    ax.set_title("Global Pareto Front: Expectation vs Probability")
    plt.colorbar(sc, ax=ax, label="Complexity")

    # 2. Heatmap: Total Photons vs Complexity -> Best Expectation
    ax = axes[0, 1]
    # Bin data for heatmap
    x_bins = np.arange(0, 26)  # Total photons 0..25
    y_bins = np.arange(0, 10)  # Complexity 0..9
    heatmap_data = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)

    for i in range(len(objectives)):
        d_complex = int(descriptors[i, 0])
        d_total = int(descriptors[i, 2])
        val = objectives[i, 0]  # Expectation

        if 0 <= d_total < len(x_bins) - 1 and 0 <= d_complex < len(y_bins) - 1:
            if (
                np.isnan(heatmap_data[d_complex, d_total])
                or val < heatmap_data[d_complex, d_total]
            ):
                heatmap_data[d_complex, d_total] = val

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="viridis_r",
        annot=False,  # Disable annotations to avoid massive text rendering overhead
        fmt=".2f",
        xticklabels=x_bins[:-1],
        yticklabels=y_bins[:-1],
    )
    ax.set_xlabel("Total Photons")
    ax.set_ylabel("Complexity")
    ax.set_title("Best Expectation Value per Cell")

    # 3. Coverage over time
    ax = axes[1, 0]
    if "coverage" in metrics:
        ax.plot(metrics["coverage"])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Repertoire Coverage")

    # 4. Best Expectation over time
    ax = axes[1, 1]
    if "min_expectation" in metrics:
        ax.plot(metrics["min_expectation"])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Expectation Value")
        ax.set_title("Optimization Progress")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")


# -------------------------
# Runner
# -------------------------
def run(
    mode: str,
    n_iters: int,
    pop_size: int,
    seed: int,
    cutoff: int,
    backend: str = "thewalrus",
    no_plot: bool = False,
    target_alpha: complex = 2.0,
    target_beta: complex = 0.0,
    low_mem: bool = False,
    genotype: str = "legacy",
    genotype_config: Dict[str, Any] = None,
    seed_scan: bool = False,
    correction_cutoff: int = None,
) -> Any:
    """Main runner supporting both QDax MOME and random search baseline."""
    np.random.seed(seed)

    # Setup
    from src.utils.gkp_operator import construct_gkp_operator

    composer = Composer(cutoff=cutoff, backend=backend)
    topology = SuperblockTopology.build_layered(2)

    # Construct GKP operator
    operator = construct_gkp_operator(
        cutoff, target_alpha, target_beta, backend=backend
    )

    # --- Setup MOME Adapter ---
    adapter = HanamuraMOMEAdapter(
        composer,
        topology,
        operator,
        cutoff,
        backend=backend,
        genotype_name=genotype,
        genotype_config=genotype_config,
        correction_cutoff=correction_cutoff,
    )

    # Calculate Genotype Dimension D
    # We use valid depth=3 as default for genotype length logic
    # Use config depth if available?
    # Genotype length might depend on depth, which is usually part of init.
    # Where does depth come from? CLI arg now!
    depth_val = 3
    if genotype_config and "depth" in genotype_config:
        depth_val = int(genotype_config["depth"])

    decoder = get_genotype_decoder(genotype, depth=depth_val, config=genotype_config)
    D = decoder.get_length(depth_val)
    print(f"Genotype '{genotype}' selected. Dimension D={D}")

    if mode == "qdax":
        try:
            from qdax.core.mome import MOME
            from qdax.core.emitters.mutation_operators import (
                polynomial_mutation,
                polynomial_crossover,
            )
            from qdax.core.emitters.standard_emitters import MixingEmitter
            # import jax # Already imported globally
            # import jax.numpy as jnp # Already imported globally
        except Exception as e:
            print(f"Error importing jax/qdax: {e}")
            print("Falling back to random mode.")
            mode = "random"

    # Common variables for result management
    repertoire = None
    emitter_state = None
    final_metrics = {}
    centroids = None

    if mode == "random":
        # Random search baseline
        best = None
        best_f = -1e99

        # Accumulate all results for "repertoire"
        all_genotypes = []
        all_fitnesses = []
        all_descriptors = []

        # Metrics history
        history_max_exp = []
        history_min_lp = []

        # Check devices
        if jax is not None:
            print(f"JAX Devices: {jax.devices()}")

        for it in range(n_iters):
            # Step annotation for profiling
            if jax is not None:
                step_ctx = jax.profiler.StepTraceAnnotation("random_step", step_num=it)
                step_ctx.__enter__()

            pop = np.random.randn(pop_size, D)
            fitnesses, descs, extras = adapter.scoring_fn_batch(pop, None)

            if jax is not None:
                step_ctx.__exit__(None, None, None)

            # Accumulate
            all_genotypes.append(pop)
            all_fitnesses.append(fitnesses)
            all_descriptors.append(descs)

            # Track best
            current_best_f = -1e99
            for i in range(pop_size):
                if fitnesses[i, 0] > best_f:
                    best_f = float(fitnesses[i, 0])
                    best = (
                        pop[i].copy(),
                        fitnesses[i].copy(),
                        descs[i].copy(),
                        extras[i],
                    )
                if fitnesses[i, 0] > current_best_f:
                    current_best_f = float(fitnesses[i, 0])

            # Metrics
            # fitness[0] is now f_expect (maximized)
            # We track max expectation directly.
            history_max_exp.append(current_best_f)
            # Similarly for log prob
            # This is approximate since we track best f0, not best f1.
            # But let's just take the mean or something for history?
            # Or just the best f0's corresponding f1?
            # For simplicity, let's track the best f0's metrics.

            if (it + 1) % 5 == 0:
                print(f"iter {it + 1}/{n_iters}, best f0 (exp) so far: {best_f:.6g}")

        print("RANDOM SEARCH DONE.")

        # Construct SimpleRepertoire
        # SimpleRepertoire is defined globally now

        # Concatenate all history
        # Shape: (Total_Pop, 1, Objs) to mimic QDax (N, Pareto, Objs)?
        # QDax repertoire has shape (N_cells, Pareto_len, Objs).
        # Here we just have a flat list of individuals.
        # Let's reshape to (N, 1, Objs)

        flat_geno = np.concatenate(all_genotypes, axis=0)
        flat_fit = np.concatenate(all_fitnesses, axis=0)
        flat_desc = np.concatenate(all_descriptors, axis=0)

        # Add Pareto dim
        flat_fit = flat_fit[:, np.newaxis, :]
        flat_desc = flat_desc[:, np.newaxis, :]
        flat_geno = flat_geno[:, np.newaxis, :]

        repertoire = SimpleRepertoire(flat_geno, flat_fit, flat_desc)

        # Metrics
        final_metrics = {
            "max_expectation": np.array(history_max_exp),
            "min_expectation": -np.array(
                history_max_exp
            ),  # Match custom_metrics naming convention
            # "min_log_prob": ... # Skip for random
        }

        # Fall through to Result Management

    elif mode == "qdax":
        if jax is not None:
            print(f"JAX Devices: {jax.devices()}")

        # Define scoring function using jax.pure_callback
        # Define scoring function using jax.pure_callback
        # Define scoring function using JAX runner
        from src.simulation.jax.runner import jax_scoring_fn_batch

        def scoring_fn(genotypes, key):
            """
            JAX-native scoring function.
            """
            # We don't use key in scoring (deterministic given genotype)
            # But we accept it for compatibility with QDax interface.

            # Call batched JAX scorer
            # Note: We removed block_until_ready() as it fails during JIT/scan tracing.
            # Profiling will still capture kernels.
            # We must pass the operator (GKP) to the scoring function now.
            # operator is available in the closure (from run() scope)
            with jax.profiler.TraceAnnotation("jax_scoring_fn_batch_qdax"):
                # Ensure operator is a JAX array
                op_jax = jnp.array(operator)
                fitnesses, descriptors = jax_scoring_fn_batch(
                    genotypes, cutoff, op_jax, genotype_name=genotype
                )

            # QDax expects extra scores (gradients/etc) but we don't have them.
            # We return empty dict for extras.
            return fitnesses, descriptors, {}

        metrics_function = custom_metrics

        # Initialize RNG
        if jax is None:
            print("JAX not found, cannot run QDax mode.")
            return
        key = jax.random.PRNGKey(seed)

        # Create initial population
        key, subkey = jax.random.split(key)

        # Initialization Range
        # If dynamic limits are on, [-2, 2] maps to [-20, 20] which is massive energy (10^16 photons).
        # This collapses all individuals to the edge of the grid, resulting in 0% coverage.
        # We use a tighter range [-0.1, 0.1] which maps to [-2, 2] (approx) to start in reasonable space.
        is_dynamic = genotype_config and genotype_config.get("dynamic_limits", False)
        init_range = 0.1 if is_dynamic else 2.0

        genotypes = jax.random.uniform(
            subkey,
            (pop_size, D),
            minval=-init_range,
            maxval=init_range,
            dtype=jnp.float32,
        )

        # Emitter setup
        crossover_function = partial(polynomial_crossover, proportion_var_to_change=0.5)
        mutation_function = partial(
            polynomial_mutation,
            eta=1.0,
            minval=-5.0,
            maxval=5.0,
            proportion_to_mutate=0.6,
        )

        # Adjust batch size for low memory
        emitter_batch_size = pop_size if LOW_MEM else pop_size * 2

        mixing_emitter = MixingEmitter(
            mutation_fn=mutation_function,
            variation_fn=crossover_function,
            variation_percentage=1.0,
            batch_size=emitter_batch_size,
        )

        # --- SEEDING STRATEGY ---
        # 1. Vacuum Seed (Identity) -> Index 0
        from src.genotypes.converter import create_vacuum_genotype, upgrade_genotype
        from src.utils.result_scanner import scan_results_for_seeds

        print("Applying Seeding Strategy...")
        if "modes" not in genotype_config:
            genotype_config["modes"] = 2

        # Inject Vacuum
        vacuum = create_vacuum_genotype(
            genotype, depth=depth_val, config=genotype_config
        )
        # Verify length
        if len(vacuum) == D:
            genotypes = genotypes.at[0].set(vacuum)
            print("  - Injected Vacuum State at index 0")
        else:
            print(f"Warning: Vacuum len {len(vacuum)} != D {D}. Skipping vacuum.")

        # 2. Result Scanning
        if seed_scan:
            # Find best candidates (mix of Exp and Prob?)
            # Let's get top 20 by expectation
            seeds = scan_results_for_seeds("output", top_k=20, metric="expectation")

            injected_count = 0
            # Start injecting at index 1
            idx = 1
            for g_src, name_src, score in seeds:
                if idx >= pop_size:
                    break

                try:
                    # Upgrade to current genotype
                    g_new = upgrade_genotype(
                        g_src,
                        name_src,
                        genotype,
                        depth=depth_val,
                        config=genotype_config,
                    )
                    if len(g_new) == D:
                        genotypes = genotypes.at[idx].set(g_new)
                        idx += 1
                        injected_count += 1
                except Exception:
                    # Conversion failed (e.g. Legacy)
                    # print(f"Skipping seed from {name_src}: {e}")
                    pass

            if injected_count > 0:
                print(f"  - Injected {injected_count} seeds from previous runs.")
            else:
                print("  - No valid seeds found or converted.")

        # Grid-based Centroids
        # D1: Active Modes (Complexity): 0..2^depth (Inclusive, so +1 bin)
        max_active = 2**depth_val
        n_bins_d1 = max_active + 1
        d1 = jnp.linspace(0, max_active, n_bins_d1)

        # D2: Max PNR
        # Coarse Gridding for high PNR
        pnr_max_val = 3
        if genotype_config and "pnr_max" in genotype_config:
            pnr_max_val = int(genotype_config["pnr_max"])

        # Target ~10 bins for PNR dimension if range is large
        n_bins_d2 = min(pnr_max_val + 1, 10)
        d2 = jnp.linspace(0, pnr_max_val, n_bins_d2)

        # D3: Total Photons
        # Coarse Gridding for high photon counts
        # Max theoretical photons = max_active * pnr_max_val
        max_photons = max_active * pnr_max_val

        # Target ~25 bins for Total dimension
        n_bins_d3 = min(max_photons + 1, 25)
        d3 = jnp.linspace(0, max_photons, n_bins_d3)

        print(
            f"Grid Definition: D1(0-{max_active}), D2(0-{pnr_max_val}), D3(0-{max_photons})"
        )
        grid = jnp.meshgrid(d1, d2, d3, indexing="ij")
        centroids = jnp.stack([g.flatten() for g in grid], axis=-1)

        print(f"Generated {centroids.shape[0]} centroids.")

        # MOME instance
        mome = MOME(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_function,
            pareto_front_max_length=5,  # As requested
        )

        # Init algorithm
        key, subkey = jax.random.split(key)

        # Warmup JAX compilation
        print("Warming up JAX compilation...")
        # Run a single dummy call to scoring_fn with a representative batch
        # We use the initial genotypes for warmup
        scoring_fn(genotypes, subkey)
        print("Warmup complete.")

        repertoire, emitter_state, init_metrics = mome.init(
            genotypes, centroids, subkey
        )

        # Run loop with progress reporting
        # We run in chunks to print metrics
        # Adjust chunk size for low memory
        # User requested to limit CPU calls -> larger chunks.
        # jax.lax.scan compiles the body once, so larger chunk_size doesn't increase graph size,
        # only the execution time per chunk.

        # Aggressive chunking: target ~50-100 steps per chunk
        if LOW_MEM:
            target_chunk = 50
        else:
            target_chunk = 100

        chunk_size = min(n_iters, target_chunk)
        n_chunks = n_iters // chunk_size
        remainder = n_iters % chunk_size

        # If there's a remainder, we'll handle it (or just ignore for now and run n_chunks)
        # Actually, scan requires fixed length.
        # If remainder > 0, we might need a final smaller chunk.
        # For simplicity, let's just run n_chunks and warn if n_iters is not divisible?
        # Or better, adjust chunk_size to be a divisor if possible?
        # No, just run n_chunks and then a final chunk.

        chunks = [chunk_size] * n_chunks
        if remainder > 0:
            chunks.append(remainder)
            n_chunks += 1

        all_metrics = {k: [] for k in init_metrics.keys()}

        # -------------------------
        # Progress Bar Setup
        # -------------------------
        op_jax = jnp.array(operator)
        gs_eig = float(jnp.linalg.eigvalsh(op_jax)[0])
        gaussian_limit = 2.0 / 3.0

        print(
            f"Starting optimization: {n_iters} iterations in {n_chunks} chunks (sizes={chunks}).\n"
            f"Target: GS Eig = {gs_eig:.6f}, Gaussian Limit = {gaussian_limit:.6f}"
        )
        print(f"[{' ' * 20}] Iter 0/{n_iters} | Waiting...", end="\r", flush=True)

        start_time = time.time()
        history_fronts = []

        # Target ~50-100 frames for animation to keep overhead low
        snapshot_interval = max(1, n_iters // 50)

        completed = 0

        chunk_start_time = time.time()

        for chunk_len in chunks:
            # Re-compile scan for each unique chunk length?
            # JAX compiles for specific static arguments (length is static).
            # So [100, 100, 100, 50] triggers 2 compilations (100 and 50).
            # This is acceptable overhead for the last chunk.

            (repertoire, emitter_state, key), metrics = jax.lax.scan(
                mome.scan_update,
                (repertoire, emitter_state, key),
                (),
                length=chunk_len,
            )

            # Accumulate metrics (taking the last value from the chunk)
            for k, v in metrics.items():
                # v is array of shape (chunk_len, ...)
                all_metrics[k].extend(v)

            # ETR Calculation
            elapsed = time.time() - start_time
            completed += chunk_len

            # Capture Pareto Front Snapshot
            if completed % snapshot_interval == 0 or completed == n_iters:
                fits = np.array(repertoire.fitnesses)
                descs = np.array(repertoire.descriptors)

                flat_fits = fits.reshape(-1, fits.shape[-1])
                flat_descs = descs.reshape(-1, descs.shape[-1])
                valid_mask = flat_fits[:, 0] > -np.inf

                valid_fits = flat_fits[valid_mask]
                valid_descs = flat_descs[valid_mask]

                history_fronts.append((valid_fits, valid_descs))

            # Print progress with ETR
            cov = float(metrics["coverage"][-1]) * 100
            best_exp = float(metrics["min_expectation"][-1])
            best_lp = float(metrics["min_log_prob"][-1])
            best_prob = float(metrics["max_probability"][-1])

            # Best fitness is -best_exp (since we minimize exp)
            best_fit = -best_exp

            # ETR Calculation
            elapsed = time.time() - start_time
            # completed is already updated
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = n_iters - completed
            etr_seconds = remaining / rate if rate > 0 else 0
            etr_str = time.strftime("%H:%M:%S", time.gmtime(etr_seconds))

            # Progress Bar
            bar_len = 20
            progress = completed / n_iters
            filled = int(bar_len * progress)
            bar = "=" * filled + "-" * (bar_len - filled)

            print(
                f"[{bar}] Iter {completed}/{n_iters} | "
                f"Chunk: {time.time() - chunk_start_time:.1f}s | "
                f"ETA: {etr_str} | "
                f"Cov: {cov:.1f}% | "
                f"Exp: {best_exp:.4f} (vs GS: {best_exp - gs_eig:+.4f}, vs G: {best_exp - gaussian_limit:+.4f})",
                end="\r",
                flush=True,
            )

            chunk_start_time = time.time()

        print("QDax MOME finished.")

        # Convert JAX metrics to numpy for plotting
        final_metrics = {k: np.array(v) for k, v in all_metrics.items()}

    # --- Result Management ---
    from src.utils.result_manager import OptimizationResult

    # Create structured output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    params_str = f"c{cutoff}_p{pop_size}_i{n_iters}"
    output_dir = f"output/{timestamp}_{params_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Config dict
    config = {
        "mode": mode,
        "n_iters": n_iters,
        "pop_size": pop_size,
        "seed": seed,
        "cutoff": cutoff,
        "backend": backend,
        "target_alpha": str(target_alpha),
        "target_beta": str(target_beta),
        "genotype": genotype,  # Explicitly store seed config
    }

    # Modes handling for results
    modes_val = 2
    if genotype_config and "modes" in genotype_config:
        modes_val = int(genotype_config["modes"])
    config["modes"] = modes_val

    # Create Result object
    # Pass history_fronts if available (only in qdax mode)
    h_fronts = locals().get("history_fronts", None)

    result = OptimizationResult(
        repertoire=repertoire,
        history=final_metrics,
        config=config,
        centroids=centroids,
        history_fronts=h_fronts,
    )

    # Save Results
    result.save(output_dir)

    # Create Animation
    if not no_plot:
        try:
            result.create_animation(os.path.join(output_dir, "history.gif"))
        except Exception as e:
            print(f"Animation creation failed: {e}")

    # Visualization (Static)
    if not no_plot:
        try:
            plot_mome_results(
                repertoire,
                final_metrics,
                filename=os.path.join(output_dir, "final_plot.png"),
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    else:
        print("Skipping plotting as requested.")

    return repertoire, emitter_state, final_metrics


# -------------------------
# CLI
# -------------------------
def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="random", choices=["random", "qdax"]
    )
    parser.add_argument(
        "--backend", type=str, default="thewalrus", choices=["thewalrus", "jax"]
    )
    parser.add_argument("--target-alpha", type=float, default=1.0)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--low-mem", action="store_true", help="Enable low-memory mode")
    parser.add_argument(
        "--genotype",
        type=str,
        default="legacy",
        help="Genotype design (legacy, A, B1, B2, C1, C2)",
    )
    parser.add_argument("--profile", action="store_true", help="Enable JAX profiling")
    parser.add_argument(
        "--target-beta",
        type=complex,
        default=0.0,
        help="Target superposition beta (complex)",
    )

    # Parameter Limits
    parser.add_argument("--depth", type=int, default=3, help="Circuit depth")
    parser.add_argument(
        "--r-scale", type=float, default=2.0, help="Max squeezing (tanh scale)"
    )
    parser.add_argument(
        "--d-scale", type=float, default=3.0, help="Max displacement scale"
    )
    parser.add_argument("--hx-scale", type=float, default=4.0, help="Homodyne X scale")
    parser.add_argument(
        "--window", type=float, default=0.1, help="Homodyne window width"
    )
    parser.add_argument("--pnr-max", type=int, default=3, help="Max PNR outcome")
    parser.add_argument(
        "--modes",
        type=int,
        default=2,
        help="Number of optical modes (1 Signal + N-1 Control)",
    )
    parser.add_argument(
        "--seed-scan",
        action="store_true",
        help="Scan output/ dir for high-fitness seeds",
    )
    # Dynamic Limits
    parser.add_argument(
        "--dynamic-limits",
        action="store_true",
        help="Enable dynamic parameter limits (discovery mode)",
    )
    parser.add_argument(
        "--correction-cutoff",
        type=int,
        default=None,
        help="Higher cutoff for leakage check (default: cutoff + 15)",
    )

    args = parser.parse_args()

    # Dynamic Limits Override
    r_scale_val = args.r_scale
    d_scale_val = args.d_scale
    corr_cutoff_val = args.correction_cutoff
    pnr_max_val = args.pnr_max

    if args.dynamic_limits:
        print("!!! Dynamic Limits Enabled !!!")
        print("  - Overriding r_scale/d_scale to 20.0 (Unbounded Discovery)")
        r_scale_val = 20.0
        d_scale_val = 20.0
        # User requested override: pnr_max -> cutoff - 1
        pnr_override = args.cutoff - 1
        if pnr_override > 0:
            print(f"  - Overriding pnr_max to {pnr_override} (cutoff-1)")
            pnr_max_val = pnr_override
        else:
            print(
                f"  - Warning: Cutoff {args.cutoff} too small for pnr override, using default {pnr_max_val}"
            )

        if corr_cutoff_val is None:
            corr_cutoff_val = args.cutoff + 15
        print(f"  - Correction Cutoff: {corr_cutoff_val}")

    # Build config dict
    genotype_config = {
        "depth": args.depth,
        "r_scale": r_scale_val,
        "d_scale": d_scale_val,
        "hx_scale": args.hx_scale,
        "window": args.window,
        "pnr_max": pnr_max_val,
        "modes": args.modes,
    }

    # Profiling context
    if args.profile and args.backend == "jax" and jax is not None:
        print("Profiling enabled. Traces will be saved to ./profiles")
        jax.profiler.start_trace("./profiles")
        try:
            run(
                mode=args.mode,
                n_iters=args.iters,
                pop_size=args.pop,
                seed=args.seed,
                cutoff=args.cutoff,
                no_plot=args.no_plot,
                backend=args.backend,
                target_alpha=args.target_alpha,
                target_beta=args.target_beta,
                low_mem=args.low_mem,
                genotype=args.genotype,
                genotype_config=genotype_config,
                seed_scan=args.seed_scan,
                correction_cutoff=corr_cutoff_val,
            )
        finally:
            jax.profiler.stop_trace()
            print("Profiling trace saved to ./profiles")
    else:
        run(
            mode=args.mode,
            n_iters=args.iters,
            pop_size=args.pop,
            seed=args.seed,
            cutoff=args.cutoff,
            no_plot=args.no_plot,
            backend=args.backend,
            target_alpha=args.target_alpha,
            target_beta=args.target_beta,
            low_mem=args.low_mem,
            genotype=args.genotype,
            genotype_config=genotype_config,
            seed_scan=args.seed_scan,
            correction_cutoff=corr_cutoff_val,
        )


if __name__ == "__main__":
    # Assuming DEFAULT_CUTOFF is defined elsewhere or a reasonable default like 10
    try:
        DEFAULT_CUTOFF
    except NameError:
        DEFAULT_CUTOFF = 10

    main()
