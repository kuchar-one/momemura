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
import pickle
import shutil
from pathlib import Path

# === Project imports ===
from src.genotypes.genotypes import get_genotype_decoder
from src.utils.result_manager import OptimizationResult, SimpleRepertoire

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
os.environ["JAX_ENABLE_X64"] = "True"

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

    # Memory optimization: don't preallocate all GPU memory
    # This helps with multi-GPU workloads and reduces fragmentation
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    # Limit memory growth to avoid OOM on large batches
    # os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

# Local defaults
DEFAULT_CUTOFF = 6
GLOBAL_CACHE = CacheManager(cache_dir="./cache", size_limit_bytes=1024 * 1024 * 512)


# moved to top


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
        genotype_name: str = "A",
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
        self.pnr_max = int(self.genotype_config.get("pnr_max", 3))

        # Instantiate decoder for local use
        self.decoder = get_genotype_decoder(genotype_name, config=self.genotype_config)

        if hasattr(composer, "cutoff") and composer.cutoff != self.cutoff:
            raise ValueError("composer.cutoff != operator cutoff")

        # Reuse executor to avoid overhead
        # Limit max_workers to avoid excessive contention
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Pre-calculate Ground State Expectation
        if jax is not None:
            self.gs_exp = float(jnp.linalg.eigvalsh(jnp.array(operator))[0])
        else:
            self.gs_exp = -4.0  # Fallback

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def evaluate_one(self, genotype: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single genotype and return metrics."""

        # Convert genotype to JAX
        g_jax = jnp.array(genotype)

        # Use decoder
        params = self.decoder.decode(g_jax, self.cutoff)

        if jax is not None:
            # Use JAX implementation of the block builder
            from src.simulation.jax.runner import jax_get_heralded_state

            # Map JAX params to leaf_params
            leaf_params = params["leaf_params"]
            # leaves: (8, ...)

            # vmap to get all leaves
            leaf_vecs, leaf_probs, leaf_modes, leaf_max_pnrs, leaf_total_pnrs = (
                jax.vmap(
                    partial(
                        jax_get_heralded_state, cutoff=self.cutoff, pnr_max=self.pnr_max
                    )
                )(leaf_params)
            )

            # Homodyne
            h_x = float(params["homodyne_x"])
            h_win = params["homodyne_window"]  # might be float or None (if -1)
            if isinstance(h_win, (int, float, jnp.ndarray)) and h_win < 1e-3:
                h_win = None
            else:
                h_win = float(h_win)

            # Prepare batch 1
            g_batch = g_jax[None, :]
            op_batch = jnp.array(self.operator)

            from src.simulation.jax.runner import jax_scoring_fn_batch

            fitnesses, descriptors, extras = jax_scoring_fn_batch(
                g_batch,
                self.cutoff,
                op_batch,
                genotype_name=self.genotype_name,
                genotype_config=self.genotype_config,
                correction_cutoff=self.correction_cutoff,
                pnr_max=self.pnr_max,
                gs_eig=self.gs_exp,
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
                fitnesses_jax, descriptors_jax, extras_jax = jax_scoring_fn_batch(
                    g_jax,
                    self.cutoff,
                    op_jax,
                    self.genotype_name,
                    self.genotype_config,
                    self.correction_cutoff,
                    pnr_max=self.pnr_max,
                    gs_eig=self.gs_exp,
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
            f_prob = metrics["log_prob"]

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
                fitnesses[i, :] = -np.inf
                # use a descriptor sentinel outside grid bounds so it cannot populate centroids
                descriptors[i, :] = np.array([-9999.0, -9999.0, -9999.0], dtype=float)
                extras.append({"error": error})
                continue

            # Objectives
            # 1. Expectation (maximize) -> QDax maximizes fitness, so use f_expect directly
            f_expect = metrics["expectation"]
            # 2. Log Prob (minimize logP) -> QDax maximizes, so use -logP
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

    if np.all(np.isnan(heatmap_data)):
        ax.text(0.5, 0.5, "No Data (All Cells empty)", ha="center", va="center")
    else:
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
    genotype: str = "A",
    genotype_config: Dict[str, Any] = None,
    seed_scan: bool = False,
    correction_cutoff: int = None,
    max_chunk_size: int = 100,
    resume_path: str = None,
    debug: bool = False,
    emitter_type: str = "hybrid",
    hybrid_ratio: float = 0.2,
    emitter_temp: float = 5.0,  # Base temp for Biased/Hybrid
    output_root: str = "output",
    global_seed_scan: bool = False,
) -> Any:
    """Main runner supporting both QDax MOME and random search baseline."""

    # Register SIGTERM handler to support Watchdog gracefull kill
    import signal

    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt("Received SIGTERM")

    signal.signal(signal.SIGTERM, handle_sigterm)

    np.random.seed(seed)

    # Resume Check
    if resume_path:
        # We need to verify resume path before heavy initialization
        chk_path = Path(resume_path) / "checkpoint_latest.pkl"
        if not chk_path.exists():
            print(f"ERROR: Resume path {chk_path} does not exist.")
            # Depending on policy, we might fail or start fresh.
            # Stricter is better to avoid accidental overwrites.
            sys.exit(1)
        print(f"Resuming from checkpoint: {chk_path}")

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
            from src.optimization.emitters import (
                BiasedMixingEmitter,
                HybridEmitter,
                MOMEOMGMEGAEmitter,
            )

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
    chunk_size = max_chunk_size

    # --- Output Directory Setup ---
    # Create structured output directory early for checkpoints and results
    # Grouping: output/experiments/{genotype}_c{cutoff}_a{alpha}_b{beta}/{timestamp}_...

    # Format alphas
    a_str = f"{target_alpha:.2f}".replace(".", "p")
    try:
        b_val = float(np.abs(target_beta))
    except Exception:
        b_val = 0.0
    b_str = f"{b_val:.2f}".replace(".", "p")

    group_id = f"{genotype}_c{cutoff}_a{a_str}_b{b_str}"

    if resume_path:
        output_dir = resume_path
        print(f"Using existing output directory: {output_dir}")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        params_str = f"p{pop_size}_i{n_iters}"

        # Base experiments folder
        base_exp_dir = os.path.join(output_root, "experiments", group_id)
        os.makedirs(base_exp_dir, exist_ok=True)

        output_dir = os.path.join(base_exp_dir, f"{timestamp}_{params_str}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # --- SHARED INITIALIZATION (JAX-based modes: qdax, single) ---
    if mode in ["qdax", "single"]:
        if jax is None:
            print(f"JAX not found, cannot run {mode} mode.")
            return

        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        # Initialization Range
        is_dynamic = genotype_config and genotype_config.get("dynamic_limits", False)
        init_range = 0.5 if is_dynamic else 2.0

        genotypes = jax.random.uniform(
            subkey,
            (pop_size, D),
            minval=-init_range,
            maxval=init_range,
            dtype=jnp.float32,
        )

        # Seeding Strategy (Shared)
        from src.genotypes.converter import create_vacuum_genotype, upgrade_genotype
        from src.utils.result_scanner import scan_results_for_seeds

        print("Applying Seeding Strategy...")
        if "modes" not in genotype_config:
            genotype_config["modes"] = 2

        # 1. Vacuum Seed
        vacuum = create_vacuum_genotype(
            genotype, depth=depth_val, config=genotype_config
        )
        if len(vacuum) == D:
            genotypes = genotypes.at[0].set(vacuum)
            print("  - Injected Vacuum State at index 0")

        # 2. Result Scanning
        if seed_scan:
            if global_seed_scan:
                seed_source_dir = os.path.join(output_root, "experiments")
                print(
                    f"  - GLOBAL SCAN enabled. Scanning matches in: {seed_source_dir}"
                )
            else:
                seed_source_dir = os.path.join(output_root, "experiments", group_id)
                print(f"  - Scanning seeds in group: {seed_source_dir}")

            seeds = scan_results_for_seeds(
                seed_source_dir,
                top_k=pop_size,
                metric="pareto",
                target_genotype=genotype,
            )

            idx = 1
            injected_count = 0
            for g_src, name_src, score in seeds:
                if idx >= pop_size:
                    break
                try:
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
                    pass

            if injected_count > 0:
                print(f"  - Injected {injected_count} seeds.")
            else:
                pass

        # --- GRID DEFINITION (Shared for QDax and Result Saving) ---
        # D1: Active Modes (Complexity): 0..2^depth (Inclusive, so +1 bin)
        max_active = 2**depth_val
        n_bins_d1 = max_active + 1
        d1 = jnp.linspace(0, max_active, n_bins_d1)

        # D2: Max PNR
        # Coarse Gridding for high PNR
        pnr_max_val = 3
        if genotype_config and "pnr_max" in genotype_config:
            pnr_max_val = int(genotype_config["pnr_max"])

        # Target ~5 bins for PNR dimension (Coarse: Low/Med/High/Peak)
        n_bins_d2 = min(pnr_max_val + 1, 5)
        d2 = jnp.linspace(0, pnr_max_val, n_bins_d2)

        # D3: Total Photons
        # Coarse Gridding for high photon counts
        # Max theoretical photons = max_active * pnr_max_val * n_control
        # Each leaf has n_control detectors, each detecting up to pnr_max photons
        n_modes_val = 2  # Default
        if genotype_config and "modes" in genotype_config:
            n_modes_val = int(genotype_config["modes"])
        n_control = max(1, n_modes_val - 1)
        max_photons = max_active * pnr_max_val * n_control

        # Target ~10 bins for Total dimension (Coarse buckets)
        n_bins_d3 = min(max_photons + 1, 10)
        d3 = jnp.linspace(0, max_photons, n_bins_d3)

        print(
            f"Grid Definition: D1(0-{max_active}), D2(0-{pnr_max_val}), D3(0-{max_photons})"
        )
        grid = jnp.meshgrid(d1, d2, d3, indexing="ij")
        centroids = jnp.stack([g.flatten() for g in grid], axis=-1)

        # Count achievable bins for accurate coverage calculation
        # Constraints: total_photons in [max_pnr, active * n_control * pnr_max]
        n_achievable_bins = 0
        d3_width = float(d3[1] - d3[0]) if len(d3) > 1 else 1.0
        for active_val in np.array(d1):
            active_int = int(round(active_val))
            for max_pnr_bin_val in np.array(d2):
                for total_val in np.array(d3):
                    if active_int == 0:
                        # Only (0, ~0, ~0) is valid
                        if max_pnr_bin_val < d3_width and total_val < d3_width:
                            n_achievable_bins += 1
                    else:
                        min_total = max_pnr_bin_val
                        max_total = active_int * n_control * pnr_max_val
                        # Check if bin center is in valid range (with half-bin tolerance)
                        if (
                            total_val >= min_total - d3_width / 2
                            and total_val <= max_total + d3_width / 2
                        ):
                            n_achievable_bins += 1

        print(f"Generated {centroids.shape[0]} centroids.")
        print(
            f"Achievable bins: {n_achievable_bins}/{centroids.shape[0]} ({100 * n_achievable_bins / centroids.shape[0]:.1f}%)"
        )

        # Store for metrics (closure will capture this)
        _n_achievable = n_achievable_bins
        total_centroids = centroids.shape[0]

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

            history_max_exp.append(current_best_f)

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

        from src.simulation.jax.runner import jax_scoring_fn_batch

        def scoring_fn(genotypes, key):
            """
            JAX-native scoring function.
            """
            with jax.profiler.TraceAnnotation("jax_scoring_fn_batch_qdax"):
                # Ensure operator is a JAX array
                op_jax = jnp.array(operator)

                # Extract pnr_max from config (ensure int)
                pnr_max_val = int(genotype_config.get("pnr_max", 3))

                fitnesses, descriptors, extras = jax_scoring_fn_batch(
                    genotypes,
                    cutoff,
                    op_jax,
                    genotype_name=genotype,
                    genotype_config=genotype_config,
                    correction_cutoff=correction_cutoff,
                    pnr_max=pnr_max_val,
                    gs_eig=float(
                        jnp.linalg.eigvalsh(op_jax)[0]
                    ),  # Calculated here for QDax
                )

            # QDax expects extra scores (gradients/etc).
            return fitnesses, descriptors, extras

        metrics_function = custom_metrics

        # Emitter setup
        crossover_function = partial(polynomial_crossover, proportion_var_to_change=0.5)
        mutation_function = partial(
            polynomial_mutation,
            eta=10.0,  # Higher = smaller perturbations (finer tuning), was 0.2 (random)
            minval=-5.0,
            maxval=5.0,
            proportion_to_mutate=0.3,  # Mutate 30% of genes (was 0.9)
        )

        # Grid-based Centroids MOVED TO SHARED INITIALIZATION
        # (Lines 919-985 migrated to lines 790+)

        # --- EMITTER SETUP (Moved here to access n_achievable_bins) ---
        # Adjust batch size for low memory
        emitter_batch_size = pop_size if LOW_MEM else pop_size * 2

        # Emitter Selection
        print(f"Using Emitter Type: {emitter_type}")

        if emitter_type == "standard":
            print("  - Strategy: Uniform Selection")
            mixing_emitter = MixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=emitter_batch_size,
            )

        elif emitter_type == "biased":
            print(f"  - Strategy: Biased (Rank-based) Selection (Temp={emitter_temp})")
            mixing_emitter = BiasedMixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=emitter_batch_size,
                temperature=emitter_temp,
                # Dynamic pressure config could be exposed too, using defaults for now
                total_bins=n_achievable_bins
                if genotype_config.get("dynamic_limits")
                else None,
            )

        elif emitter_type == "hybrid":
            print(f"  - Strategy: Hybrid (Uniform + Elite) | Ratio: {hybrid_ratio}")
            # 1. Exploration (Standard)
            exploration_emitter = MixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=emitter_batch_size,  # HybridEmitter will resize this
            )

            # 2. Intensification (Biased/Elite)
            # Use Low Temp for strict exploitation (~0.05 from previous logic)
            # Or use dynamic? Let's use a fixed aggressive temp for the elite stream to ensure it works as "Elite"
            elite_temp = 0.05
            print(f"    -> Elite Stream Temp: {elite_temp}")

            intensification_emitter = BiasedMixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=emitter_batch_size,  # HybridEmitter will resize this
                temperature=elite_temp,
                # No dynamic pressure needed for elite stream, it's always aggressive
            )

            # 3. Hybrid Wrapper
            mixing_emitter = HybridEmitter(
                exploration_emitter=exploration_emitter,
                intensification_emitter=intensification_emitter,
                intensification_ratio=hybrid_ratio,
                batch_size=emitter_batch_size,
            )

        elif emitter_type == "gradient":
            print("  - Strategy: Gradient Propagation (OMG-MEGA style)")
            # MOME-compatible OMG-MEGA Emitter
            mixing_emitter = MOMEOMGMEGAEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.0,
                batch_size=emitter_batch_size,
                sigma_g=0.5,
            )

        elif emitter_type == "hybrid-gradient":
            print("  - Strategy: Hybrid Gradient (Standard + OMG-MEGA)")
            # 1. Exploration (Standard)
            exploration_emitter = MixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=emitter_batch_size,
            )

            # 2. Intensification (Gradient)
            intensification_emitter = MOMEOMGMEGAEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.0,
                batch_size=emitter_batch_size,
                sigma_g=0.5,
            )

            # 3. Hybrid Wrapper
            # This uses the GENERIC HybridEmitter class from emitters.py
            # composed with the standard and gradient emitters.
            mixing_emitter = HybridEmitter(
                exploration_emitter=exploration_emitter,
                intensification_emitter=intensification_emitter,
                intensification_ratio=hybrid_ratio,
                batch_size=emitter_batch_size,
            )

        elif emitter_type == "mega-hybrid":
            print("  - Strategy: Mega Hybrid (Gradient + Biased + Standard)")
            # Architecture:
            # - Intensification Stream (20%): Gradient (OMG-MEGA)
            # - Exploration Stream (80%): Hybrid (Biased + Standard)
            #   - Intensification (50% of 80%): Biased
            #   - Exploration (50% of 80%): Standard

            # 1. Gradient Emitter (Top-Level Intensification)
            grad_emitter = MOMEOMGMEGAEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.0,
                batch_size=emitter_batch_size,  # Will be resized by parent
                sigma_g=0.5,
            )

            # 2. Inner Hybrid (Biased + Standard)
            # We need to pre-calculate its batch size to ensure internal split is correct
            n_gradient = int(emitter_batch_size * hybrid_ratio)
            n_inner = emitter_batch_size - n_gradient

            # 2a. Standard
            std_emitter = MixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=n_inner,  # Will be resized
            )
            # 2b. Biased
            biased_emitter = BiasedMixingEmitter(
                mutation_fn=mutation_function,
                variation_fn=crossover_function,
                variation_percentage=0.5,
                batch_size=n_inner,
                temperature=emitter_temp,
                total_bins=total_centroids,
            )

            # 2c. Inner Hybrid Construction
            # 80% Standard (Exploration), 20% Biased (Intensification)
            inner_hybrid = HybridEmitter(
                exploration_emitter=std_emitter,
                intensification_emitter=biased_emitter,
                intensification_ratio=0.2,
                batch_size=n_inner,
            )

            # 3. Outer Hybrid Wrapper
            mixing_emitter = HybridEmitter(
                exploration_emitter=inner_hybrid,
                intensification_emitter=grad_emitter,
                intensification_ratio=hybrid_ratio,
                batch_size=emitter_batch_size,
            )

        else:
            raise ValueError(f"Unknown emitter_type: {emitter_type}")

        # Create local metrics function with correct achievable bins
        def local_metrics(repertoire):
            """Custom metrics with accurate achievable coverage."""
            valid_mask = repertoire.fitnesses[:, 0, 0] != -jnp.inf
            # Use achievable bins as denominator, not total grid
            coverage = jnp.sum(valid_mask) / _n_achievable

            flat_fitnesses = repertoire.fitnesses.reshape(
                -1, repertoire.fitnesses.shape[-1]
            )
            flat_valid = flat_fitnesses[:, 0] != -jnp.inf
            min_expectation = -jnp.max(
                jnp.where(flat_valid, flat_fitnesses[:, 0], -jnp.inf)
            )
            min_log_prob = -jnp.max(
                jnp.where(flat_valid, flat_fitnesses[:, 1], -jnp.inf)
            )
            max_prob = jnp.power(10.0, -min_log_prob)

            return {
                "coverage": coverage,
                "min_expectation": min_expectation,
                "min_log_prob": min_log_prob,
                "max_probability": max_prob,
            }

        # MOME instance
        mome = MOME(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=local_metrics,  # Use local metrics with correct coverage
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

        # Use user-specified chunk size (or lower for low memory)
        target_chunk = min(max_chunk_size, 50 if LOW_MEM else max_chunk_size)

        chunk_size = min(n_iters, target_chunk)

        # Resume Logic (MOME State Loading)
        if resume_path:
            chk_path = Path(resume_path) / "checkpoint_latest.pkl"
            print(f"Loading MOME state from {chk_path}...")
            try:
                with open(chk_path, "rb") as f:
                    checkpoint = pickle.load(f)

                repertoire = checkpoint["repertoire"]
                emitter_state = checkpoint["emitter_state"]
                loaded_key = checkpoint["key"]
                completed_iters = checkpoint["completed_iters"]

                # Check for config mismatch (warn only)
                if checkpoint.get("pop_size", pop_size) != pop_size:
                    print(
                        "WARNING: Resume pop_size mismatch. Proceeding might fail if arrays don't match."
                    )

                print(f"Resumed from Iter {completed_iters}.")

                # Skip init
                # We need a new key? loaded_key is the one saved after update.
                key = loaded_key

                # Adjust remaining iterations
                # n_iters is total target.
                current_n_iters = n_iters
                if completed_iters >= n_iters:
                    print(f"Run already completed ({completed_iters}/{n_iters}).")
                    # Should we extend?
                    # For now just exit or allow extension if n_iters passed is higher?
                    pass
                else:
                    # We continue loop from completed_iters
                    # The chunk logic needs to know how many left.
                    pass

            except Exception as e:
                print(f"Resume Failed: {e}")
                sys.exit(1)
        else:
            completed_iters = 0
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

        # Calculate remaining chunks
        remaining_iters = n_iters - completed_iters
        if remaining_iters <= 0:
            print("Target iterations reached.")
            remaining_iters = 0
            n_chunks = 0
            chunks = []
        else:
            # We chunk the *remaining* iters
            chunk_size = min(remaining_iters, target_chunk)
            n_chunks = remaining_iters // chunk_size
            remainder = remaining_iters % chunk_size

            chunks = [chunk_size] * n_chunks
            if remainder > 0:
                chunks.append(remainder)
                n_chunks += 1

        all_metrics = {k: [] for k in init_metrics.keys()} if not resume_path else {}

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

        completed = completed_iters

        chunk_start_time = time.time()

        # Resume History?
        if resume_path and "history" in checkpoint:
            if "history" in checkpoint:
                # Checkpoint history might be numpy or list
                all_metrics = checkpoint["history"]
                print(
                    f"Restored metrics history ({len(list(all_metrics.values())[0])} items)."
                )
            else:
                pass

        chunk_start_time = time.time()

        # Signal Handling for Graceful Shutdown
        def signal_handler(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, signal_handler)

        try:
            for chunk_idx, chunk_len in enumerate(chunks):
                # Run `chunk_len` steps in Python
                chunk_metrics = {k: [] for k in init_metrics.keys()}

                # Unwrap MOME.update to avoid jit(pmap)
                # mome.update is a bound method. __wrapped__ is the unbound function.
                # We need to bind it or pass self.
                if hasattr(mome.update, "__wrapped__"):
                    update_fn = partial(mome.update.__wrapped__, mome)
                else:
                    update_fn = mome.update

                for _ in range(chunk_len):
                    # Update Step
                    # QDax MOME.update signature: (repertoire, emitter_state, key) -> (repertoire, emitter_state, metrics)
                    # It does NOT return the new key. We must manage the key splitting manually (like scan_update does).

                    key, subkey = jax.random.split(key)

                    repertoire, emitter_state, step_metrics = update_fn(
                        repertoire, emitter_state, subkey
                    )

                    # Accumulate metrics
                    # step_metrics is a dict of scalars (or 0-d arrays)
                    for k, v in step_metrics.items():
                        chunk_metrics[k].append(v)

                # Process Chunk Metrics
                # Convert list of scalars to array for `all_metrics` extend
                for k, v_list in chunk_metrics.items():
                    all_metrics[k].extend(v_list)
                    if jax is not None:
                        # Pull to host
                        v_list_np = [jax.device_get(x) for x in v_list]
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].extend(v_list_np)
                    else:
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].extend(v_list)

                # --- Checkpointing at end of chunk ---
                completed += chunk_len

                # Save Checkpoint
                try:
                    # We use a temporary file to ensure atomic write
                    chk_data = {
                        "repertoire": repertoire,
                        "emitter_state": emitter_state,
                        "key": key,
                        "completed_iters": completed,
                        "pop_size": pop_size,
                        "history": all_metrics,
                    }

                    # Ensure directory exists (paranoia check)
                    if not os.path.exists(output_dir):
                        print(
                            f"WARNING: Output directory {output_dir} missing. Recreating."
                        )
                        os.makedirs(output_dir, exist_ok=True)

                    # Temp file
                    tmp_chk = Path(output_dir) / "checkpoint_tmp.pkl"
                    final_chk = Path(output_dir) / "checkpoint_latest.pkl"

                    with open(tmp_chk, "wb") as f:
                        pickle.dump(chk_data, f)

                    # Atomic rename
                    shutil.move(tmp_chk, final_chk)

                except Exception as e:
                    print(f"Checkpoint prep failed: {e}")

                # --- Periodic Archive Sweep (Option D) ---
                # Validate archive every 250 generations to remove numerical artifacts
                sweep_interval = 250
                if (
                    correction_cutoff is not None
                    and completed % sweep_interval == 0
                    and completed > 0
                ):
                    try:
                        from src.utils.archive_validator import (
                            validate_and_clean_archive,
                        )

                        print(f"\n=== Periodic Archive Sweep (gen {completed}) ===")
                        repertoire, num_removed = validate_and_clean_archive(
                            repertoire=repertoire,
                            base_cutoff=cutoff,
                            correction_cutoff=correction_cutoff,
                            genotype_name=genotype,
                            genotype_config=genotype_config,
                            pnr_max=3,
                            fidelity_threshold=0.9,
                        )
                        if num_removed > 0:
                            print(f"Removed {num_removed} artifacts.\n")
                        else:
                            print("Archive is clean.\n")
                    except Exception as e:
                        print(f"Periodic sweep failed (non-fatal): {e}")

                # ETR Calculation

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
                # Use last val from chunk_metrics (dictionary of lists)
                cov = float(chunk_metrics["coverage"][-1]) * 100
                best_exp = float(chunk_metrics["min_expectation"][-1])
                best_lp = float(chunk_metrics["min_log_prob"][-1])
                best_prob = float(chunk_metrics["max_probability"][-1])

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

        except KeyboardInterrupt:
            print("\n\nOptimization interrupted (SIGINT/SIGTERM). Saving progress...")

        print("QDax MOME finished.")

        # Convert JAX metrics to numpy for plotting
        final_metrics = {k: np.array(v) for k, v in all_metrics.items()}

    elif mode == "single":
        # --- Single Objective Optimization (Gradient Descent) ---
        print("Starting Single Objective Optimization (Gradient Descent)...")
        try:
            import optax
        except ImportError:
            print("Error: optax is required for single-objective mode.")
            return

        if jax is None:
            print("Error: JAX is required for single-objective mode.")
            return

        # Prepare Scoring Function (Gradients)
        from src.simulation.jax.runner import jax_scoring_fn_batch

        # Wrap scoring to return 'expectations' (fitness[0]) directly for gradient
        # Actually our jax_scoring_fn_batch returns fitnesses = [exp, -logP, -comp, -photons]
        # But we need gradients of the expectation value itself to minimize it.
        # Wait, jax_scoring_fn_batch ALREADY returns 'gradients' in extras!
        # extras["gradients"] = grad(loss_fn) where loss = penalized expectation.
        # So we can just use the provided gradients directly.

        # Optimizer Setup
        # We use a multi-start gradient descent approach:
        # 1. Start with 'pop_size' random candidates
        # 2. Iterate optimization steps
        # 3. Use the computed gradients to update

        learning_rate = 0.01  # Reduced from 0.05 for stability
        # Use Chain: Clip -> Adam
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
        )
        opt_state = optimizer.init(genotypes)

        # Initialize Results Containers
        history_best_exp = []
        global_best_exp = float("inf")
        global_best_log_prob = -float("inf")
        global_best_genotype = None

        print(f"Optimizer: Adam(lr={learning_rate})")
        print(f"JAX Devices: {jax.devices()}")

        # Pre-calc GS Eig
        gs_eig = float(jnp.linalg.eigvalsh(jnp.array(operator))[0])
        print(f"Ground State Expectation: {gs_eig:.6f}")

        # Optimization Loop
        # We process in batches (population is the batch)

        # We need a step function that takes (genotypes, opt_state) -> (new_genotypes, new_opt_state, metrics)
        # But our scoring function is complex and returns gradients.
        # We can implement a manual step using the gradients returned by scoring_fn.

        @jax.jit
        def update_step(genotypes, grads, opt_state):
            # 1. NaN Guard: Replace NaN gradients with 0.0
            grads = jax.tree_util.tree_map(
                lambda x: jnp.where(jnp.isnan(x), 0.0, x), grads
            )

            # 2. Optimizer Update (includes clipping)
            updates, new_opt_state = optimizer.update(grads, opt_state, genotypes)
            new_genotypes = optax.apply_updates(genotypes, updates)

            # 3. Parameter Clamping: Prevent unphysical energy blowup
            new_genotypes = jnp.clip(new_genotypes, -5.0, 5.0)

            return new_genotypes, new_opt_state

        start_time = time.time()

        all_genotypes_history = []
        all_fitnesses_history = []
        all_descriptors_history = []

        try:
            for i in range(n_iters):
                # 1. Score & Get Gradients
                fitnesses, descriptors, extras = jax_scoring_fn_batch(
                    genotypes,
                    cutoff,
                    jnp.array(operator),
                    genotype_name=genotype,
                    genotype_config=genotype_config,
                    correction_cutoff=correction_cutoff,
                    pnr_max=int(genotype_config.get("pnr_max", 3)),
                    gs_eig=gs_eig,
                )

                # fitnesses[:, 0] is -expectation (maximized).
                # We want to MINIMIZE expectation.
                # The `extras["gradients"]` provided by runner.py are gradients of `loss_val`
                # where `loss_val = exp_val + leakage`.
                # So these are exactly the gradients we need for minimization.

                grads = extras["gradients"]

                # 3. Metrics (Check BEFORE update to align with current fitnesses)
                # fitness[0] = -expectation
                current_expectations = -fitnesses[:, 0]

                # Find batch best
                batch_min_idx = jnp.argmin(current_expectations)
                batch_min_exp = float(current_expectations[batch_min_idx])

                # Update Global Best
                if batch_min_exp < global_best_exp:
                    global_best_exp = batch_min_exp
                    # Store the JAX array for the best genotype
                    global_best_genotype = genotypes[batch_min_idx]

                history_best_exp.append(float(global_best_exp))

                # Track Best Prob
                current_log_probs = fitnesses[:, 1]
                batch_max_idx = jnp.argmax(current_log_probs)
                batch_max_log_prob = float(current_log_probs[batch_max_idx])

                if batch_max_log_prob > global_best_log_prob:
                    global_best_log_prob = batch_max_log_prob
                    global_best_prob_genotype = genotypes[batch_max_idx]

                # Formatting (Show Best Exp and Global Best Prob)
                if (i + 1) % 10 == 0 or i == 0:
                    best_prob_val = 10**global_best_log_prob
                    print(
                        f"Iter {i + 1}/{n_iters} | Best Exp: {global_best_exp:.6f} | Best Prob: {best_prob_val:.6f}"
                    )

                    # Capture bad genotype
                    if best_prob_val > 1.01:
                        print(f"!!! PROB > 1 DETECTED: {best_prob_val:.6f} !!!")
                        print("Saving bad genotype to 'bad_genotype.npz'...")

                        # Use global variable if set, otherwise current batch best
                        # global_best_prob_genotype is set above if we found a new best
                        # If we didn't update this step, we still want the GLOBAL best one (which caused the >1).

                        # Note: global_best_prob_genotype might be None if start condition, but prob > 1 implies we updated it at least once.

                        target_geno = (
                            global_best_prob_genotype
                            if global_best_prob_genotype is not None
                            else genotypes[batch_max_idx]
                        )

                        np.savez(
                            "bad_genotype.npz",
                            genotype=target_geno,
                            prob=best_prob_val,
                        )

                # Store history periodically or only final?
                # Storing all history allows animation of convergence
                if not no_plot:
                    all_genotypes_history.append(genotypes)
                    all_fitnesses_history.append(fitnesses)
                    all_descriptors_history.append(descriptors)

                # 2. Update (Now apply step to move to next)
                genotypes, opt_state = update_step(genotypes, grads, opt_state)
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Saving current progress...")

        print("Single Objective Optimization Finished.")

        # Final evaluation
        fitnesses, descriptors, extras = jax_scoring_fn_batch(
            genotypes,
            cutoff,
            jnp.array(operator),
            genotype_name=genotype,
            genotype_config=genotype_config,
            correction_cutoff=correction_cutoff,
            pnr_max=int(genotype_config.get("pnr_max", 3)),
        )

        # Check final batch
        current_expectations = -fitnesses[:, 0]
        batch_min_idx = jnp.argmin(current_expectations)
        batch_min_exp = float(current_expectations[batch_min_idx])
        if batch_min_exp < global_best_exp:
            global_best_exp = batch_min_exp
            global_best_genotype = genotypes[batch_min_idx]

        # Construct Repertoire
        g_final = np.array(genotypes)
        f_final = np.array(fitnesses)
        d_final = np.array(descriptors)

        # Inject Global Best if available
        if global_best_genotype is not None:
            # Explicitly re-evaluate global best to ensure correct matching data (deterministic)
            gb_in = global_best_genotype[None, :]
            gb_f, gb_d, _ = jax_scoring_fn_batch(
                gb_in,
                cutoff,
                jnp.array(operator),
                genotype_name=genotype,
                genotype_config=genotype_config,
                correction_cutoff=correction_cutoff,
                pnr_max=int(genotype_config.get("pnr_max", 3)),
            )

            # Append to results
            g_final = np.concatenate([g_final, np.array(gb_in)], axis=0)
            f_final = np.concatenate([f_final, np.array(gb_f)], axis=0)
            d_final = np.concatenate([d_final, np.array(gb_d)], axis=0)
            print(
                f"Injected Global Best (Exp: {global_best_exp:.6f}) into final results."
            )

        # Reshape for Repertoire (Pop, 1, Objs)
        g_final = g_final[:, np.newaxis, :]
        f_final = f_final[:, np.newaxis, :]
        d_final = d_final[:, np.newaxis, :]

        repertoire = SimpleRepertoire(g_final, f_final, d_final)

        # Metrics
        final_metrics = {"min_expectation": np.array(history_best_exp)}

    # --- Result Management ---

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
        "chunk_size": chunk_size,
    }

    if correction_cutoff is not None:
        config["correction_cutoff"] = correction_cutoff

    # Modes handling for results
    modes_val = 3
    if genotype_config and "modes" in genotype_config:
        modes_val = int(genotype_config["modes"])
    config["modes"] = modes_val

    # Merge genotype_config into main config for persistence
    if genotype_config:
        config.update(genotype_config)

    # Explicitly store genotype name
    config["genotype"] = genotype

    # Create Result object
    # Pass history_fronts if available (only in qdax mode)
    h_fronts = locals().get("history_fronts", None)

    # --- Final Archive Validation (Option F) ---
    # Validate all archive solutions at dual cutoffs to remove numerical artifacts
    if correction_cutoff is not None and cutoff is not None:
        try:
            from src.utils.archive_validator import final_archive_validation

            print("\n=== Final Archive Validation ===")
            repertoire, num_removed = final_archive_validation(
                repertoire=repertoire,
                base_cutoff=cutoff,
                correction_cutoff=correction_cutoff,
                genotype_name=genotype,
                genotype_config=genotype_config,
                pnr_max=3,  # Fixed default for artifact validation
                fidelity_threshold=0.9,
                max_iterations=5,
            )
            print(f"Archive validation complete. Removed {num_removed} artifacts.\n")
        except Exception as e:
            print(f"Archive validation failed (non-fatal): {e}")

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
        "--mode", type=str, default="random", choices=["random", "qdax", "single"]
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Maximum iterations per chunk for progress reporting (default: 100)",
    )
    # Emitter Config
    parser.add_argument(
        "--emitter",
        type=str,
        default="hybrid",
        choices=[
            "standard",
            "biased",
            "hybrid",
            "gradient",
            "hybrid-gradient",
            "mega-hybrid",
        ],
        help="Emitter strategy to use.",
    )
    parser.add_argument(
        "--hybrid-ratio",
        type=float,
        default=0.2,
        help="Ratio of elite intensification (hybrid mode only).",
    )
    parser.add_argument(
        "--emitter-temp",
        type=float,
        default=5.0,
        help="Base temperature for biased emitter.",
    )

    parser.add_argument(
        "--alpha-expectation",
        type=float,
        default=1.0,
        help="Weight for expectation value in single-objective loss (default: 1.0).",
    )
    parser.add_argument(
        "--alpha-probability",
        type=float,
        default=0.0,
        help="Weight for log-probability in single-objective loss (default: 0.0).",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to output directory to resume from.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging."
    )
    parser.add_argument(
        "--global-seed-scan",
        action="store_true",
        help="Scan ALL experiment subfolders for seeds matching genotype.",
    )

    args = parser.parse_args()
    print(
        f"DEBUG: Parsed Args: resume={args.resume}, debug={args.debug}, chunk_size={args.chunk_size}"
    )

    # Dynamic Limits Override
    r_scale_val = args.r_scale
    d_scale_val = args.d_scale
    hx_scale_val = args.hx_scale
    corr_cutoff_val = args.correction_cutoff
    pnr_max_val = args.pnr_max

    if args.dynamic_limits:
        print("!!! Dynamic Limits Enabled !!!")

        # Adaptive r_scale based on cutoff:
        # Mean photon number for squeezed vacuum ~ sinh^2(r)
        # To keep mean photons at cutoff/3, use r ~ asinh(sqrt(cutoff/3))
        # This leaves ~67% of Fock space as headroom for mixing
        r_scale_val = float(np.arcsinh(np.sqrt(args.cutoff / 3.0)))

        # Adaptive d_scale based on cutoff:
        # Mean photon number for coherent state ~ |alpha|^2
        # With complex alpha, max |alpha|^2 = 2 * d_scale^2 when both real/imag maxed
        # To keep mean photons at cutoff/3, use d_scale ~ sqrt(cutoff/6)
        d_scale_val = float(np.sqrt(args.cutoff / 6.0))

        # Adaptive hx_scale based on cutoff:
        # Homodyne projection at position x selects Hermite functions phi_n(x)
        # For large |x|, this concentrates probability in high Fock states
        # After BS mixing, this can create states with mean_n >> cutoff/2
        # Very conservative to prevent artifacts (previously /8, caused issues)
        hx_scale_val = float(np.sqrt(args.cutoff / 16.0))

        print(
            f"  - Adaptive r_scale: {r_scale_val:.2f} (mean photons ~ {int(np.sinh(r_scale_val) ** 2)})"
        )
        print(
            f"  - Adaptive d_scale: {d_scale_val:.2f} (mean photons ~ {int(d_scale_val**2)})"
        )
        print(
            f"  - Adaptive hx_scale: {hx_scale_val:.2f} (conservative to prevent truncation artifacts)"
        )

        # Determine pnr_max:
        # 1. If user explicitly set --pnr-max, use that
        # 2. Otherwise use min(cutoff-1, 15) to avoid OOM
        # The default pnr_max is 3, so if it's still 3, user didn't override
        user_set_pnr = args.pnr_max != 3  # Check if user explicitly set it

        if user_set_pnr:
            pnr_max_val = args.pnr_max
            print(f"  - Using user-specified pnr_max: {pnr_max_val}")
        else:
            # Use sensible default that won't OOM
            pnr_override = min(args.cutoff - 1, 15)  # Cap at 15 to prevent OOM
            if pnr_override > 0:
                print(
                    f"  - Auto pnr_max: {pnr_override} (min(cutoff-1, 15) for memory safety)"
                )
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
        "hx_scale": hx_scale_val,
        "window": args.window,
        "pnr_max": pnr_max_val,
        "modes": args.modes,
        "alpha_expectation": args.alpha_expectation,
        "alpha_probability": args.alpha_probability,
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
                max_chunk_size=args.chunk_size,
                resume_path=args.resume,
                debug=args.debug,
                emitter_type=args.emitter,
                hybrid_ratio=args.hybrid_ratio,
                emitter_temp=args.emitter_temp,
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
            max_chunk_size=args.chunk_size,
            resume_path=args.resume,
            debug=args.debug,
            emitter_type=args.emitter,
            hybrid_ratio=args.hybrid_ratio,
            emitter_temp=args.emitter_temp,
            global_seed_scan=args.global_seed_scan,
        )


if __name__ == "__main__":
    # Assuming DEFAULT_CUTOFF is defined elsewhere or a reasonable default like 10
    try:
        DEFAULT_CUTOFF
    except NameError:
        DEFAULT_CUTOFF = 10

    main()
