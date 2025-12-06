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
import math
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import matplotlib

# === Project imports ===
from src.circuits.gaussian_herald_circuit import GaussianHeraldCircuit
from src.circuits.composer import Composer, SuperblockTopology
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
def decode_genotype(
    genotype: np.ndarray,
    *,
    max_signal: int = 2,
    max_control: int = 2,
    max_schmidt: int = 2,
    max_pnr: int = 3,
    max_modes_for_interf: int = 2,
    cutoff: int = DEFAULT_CUTOFF,
) -> Dict[str, Any]:
    """Decode a 1D real genotype into Gaussian block parameters."""
    g = np.asarray(genotype, dtype=float).flatten()
    # Calculate expected length
    expect_len = (
        2
        + max_schmidt
        + 2 * ((max_modes_for_interf * (max_modes_for_interf - 1)) // 2)
        + 2 * max_modes_for_interf
        + 2 * ((max_modes_for_interf * (max_modes_for_interf - 1)) // 2)
        + 2 * max_modes_for_interf
        + 2 * max_signal
        + 2 * max_control
        + max_control
        + 5
    )
    if g.size < expect_len:
        g = np.concatenate([g, np.zeros(expect_len - g.size, dtype=float)])

    idx = 0
    # Discrete modes
    # u_sig = np.clip(g[idx], 0.0, 1.0) # Unused since n_signal forced to 1
    idx += 1
    u_ctrl = np.clip(g[idx], 0.0, 1.0)
    idx += 1

    # FORCE pure-state pipeline single signal
    # Note: genotype bits for n_signal are ignored to keep genotype length stable.
    n_signal = 1
    n_control = (
        int(1 + math.floor(u_ctrl * (max_control - 1))) if max_control > 1 else 1
    )

    # Schmidt squeezings
    raw_r = g[idx : idx + max_schmidt]
    idx += max_schmidt
    tmss_squeezing = [float(np.tanh(x) * 1.2) for x in raw_r]
    schmidt_rank = min(len(tmss_squeezing), min(n_signal, n_control))
    tmss_squeezing = tmss_squeezing[:schmidt_rank]

    # Interferometer parameters
    Lmax = max_modes_for_interf * (max_modes_for_interf - 1) // 2
    us_theta_pack = g[idx : idx + Lmax]
    idx += Lmax
    us_phi_pack = g[idx : idx + Lmax]
    idx += Lmax
    us_varphi_pack = g[idx : idx + max_modes_for_interf]
    idx += max_modes_for_interf
    uc_theta_pack = g[idx : idx + Lmax]
    idx += Lmax
    uc_phi_pack = g[idx : idx + Lmax]
    idx += Lmax
    uc_varphi_pack = g[idx : idx + max_modes_for_interf]
    idx += max_modes_for_interf

    def map_angles(arr, scale=math.pi / 2):
        return [float(np.tanh(x) * scale) for x in arr.tolist()]

    M_us = max(n_signal, 1)
    M_uc = max(n_control, 1)
    us_theta = map_angles(us_theta_pack)[: (M_us * (M_us - 1) // 2)]
    us_phi = map_angles(us_phi_pack)[: (M_us * (M_us - 1) // 2)]
    us_varphi = map_angles(us_varphi_pack)[:M_us]
    uc_theta = map_angles(uc_theta_pack)[: (M_uc * (M_uc - 1) // 2)]
    uc_phi = map_angles(uc_phi_pack)[: (M_uc * (M_uc - 1) // 2)]
    uc_varphi = map_angles(uc_varphi_pack)[:M_uc]

    us_params = {
        "theta": np.array(us_theta),
        "phi": np.array(us_phi),
        "varphi": np.array(us_varphi),
    }
    uc_params = {
        "theta": np.array(uc_theta),
        "phi": np.array(uc_phi),
        "varphi": np.array(uc_varphi),
    }

    # Displacements
    disp_s = []
    for _ in range(max_signal):
        real = float(np.tanh(g[idx]) * 1.0)
        imag = float(np.tanh(g[idx + 1]) * 1.0)
        disp_s.append(real + 1j * imag)
        idx += 2
    disp_s = disp_s[:n_signal]
    disp_c = []
    for _ in range(max_control):
        real = float(np.tanh(g[idx]) * 1.0)
        imag = float(np.tanh(g[idx + 1]) * 1.0)
        disp_c.append(real + 1j * imag)
        idx += 2
    disp_c = disp_c[:n_control]

    # PNR outcomes
    pnr = []
    for j in range(max_control):
        u = float(np.clip(g[idx], 0.0, 1.0))
        idx += 1
        val = int(round(u * max_pnr))
        pnr.append(val)
    pnr = pnr[:n_control]

    # Homodyne params
    hom_x_raw = float(g[idx])
    idx += 1
    hom_win_raw = float(g[idx])
    idx += 1
    homodyne_x = float(np.tanh(hom_x_raw) * 4.0)
    homodyne_window = float(np.abs(np.tanh(hom_win_raw) * 2.0))

    # Round to ensure cache hits for quadrature matrices
    homodyne_x = round(homodyne_x, 6)
    homodyne_window = round(homodyne_window, 6)

    if homodyne_window < 1e-3:
        homodyne_window = None

    # Force fixed beam splitter parameters for consistent caching/architecture
    # We still consume the genotype slots to maintain compatibility
    _ = float(np.tanh(g[idx]) * (math.pi / 2))
    idx += 1
    _ = float(np.tanh(g[idx]) * math.pi)
    idx += 1
    mix_theta = math.pi / 4.0
    mix_phi = 0.0
    final_rot = float(np.tanh(g[idx]) * math.pi)
    idx += 1

    return {
        "n_signal": n_signal,
        "n_control": n_control,
        "tmss_squeezing": tmss_squeezing,
        "us_params": us_params,
        "uc_params": uc_params,
        "U_s": None,
        "U_c": None,
        "disp_s": disp_s,
        "disp_c": disp_c,
        "pnr_outcome": pnr,
        "homodyne_x": homodyne_x,
        "homodyne_window": homodyne_window,
        "mix_theta": mix_theta,
        "mix_phi": mix_phi,
        "final_rotation": final_rot,
        "signal_cutoff": cutoff,
    }


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
    ):
        self.composer = composer
        self.topology = topology
        self.operator = operator
        self.cutoff = int(cutoff)
        self.mode = mode
        self.homodyne_resolution = homodyne_resolution
        self.backend = backend
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
        params = decode_genotype(genotype, cutoff=self.cutoff)

        # If pure mode, we can optionally enforce n_signal=1 in decode or builder
        # For now, builder enforces it.

        try:
            vec, prob_block = gaussian_block_builder_from_params(
                params, cutoff=self.cutoff, backend=self.backend
            )
        except Exception as e:
            # invalid genotype: too small prob or other failure -> mark as invalid
            raise ValueError(f"Invalid block (herald/purity): {e}")

        # Replicate per leaf
        n_leaves = self.topology._count_leaves(self.topology.plan)
        fock_vecs = [vec.copy() for _ in range(n_leaves)]
        p_heralds = [prob_block for _ in range(n_leaves)]

        # Configure topology evaluation based on mode
        is_pure_mode = self.mode == "pure"

        # In pure mode, force window to None to ensure pure path
        h_window = params.get("homodyne_window", None)
        if is_pure_mode:
            h_window = None

        try:
            # Force pure_only True to ensure whole topology remains pure
            final_state, joint_prob = self.topology.evaluate_topology(
                composer=self.composer,
                fock_vecs=fock_vecs,
                p_heralds=p_heralds,
                homodyne_x=params.get("homodyne_x", None),
                homodyne_window=h_window,
                homodyne_resolution=self.homodyne_resolution if is_pure_mode else None,
                theta=params.get("mix_theta", math.pi / 4.0),
                phi=params.get("mix_phi", 0.0),
                exact_mixed=False,  # allow fast path
                n_hom_points=201,
                pure_only=is_pure_mode,
            )
        except Exception as e:
            # topology required mixed-state propagation -> mark invalid
            raise ValueError(f"Topology not pure for genotype: {e}")

        # Apply final rotation
        final_rot = params.get("final_rotation", 0.0)
        if abs(final_rot) > 1e-12:
            Urot = np.diag(np.exp(1j * final_rot * np.arange(self.cutoff)))
            if final_state.ndim == 1:
                final_state = Urot @ final_state
            else:
                final_state = Urot @ final_state @ Urot.conj().T

        # Expectation value
        if final_state.ndim == 1:
            exp_val = float(
                np.real(np.vdot(final_state, (self.operator @ final_state)))
            )
        else:
            exp_val = float(np.real(np.trace(final_state @ self.operator)))

        # Metrics
        def count_pairs(node):
            if node[0] == "leaf":
                return 0
            return 1 + count_pairs(node[1]) + count_pairs(node[2])

        topo = count_pairs(self.topology.plan)

        # Include mode count in complexity as requested
        # Complexity = tree_depth + number_of_modes
        # Tree depth for 2-layer superblock is roughly 2
        # Number of modes = n_signal + n_control
        n_modes = params["n_signal"] + params["n_control"]
        complexity = topo + n_modes

        pnr = params.get("pnr_outcome", [])
        total_ph = int(sum(pnr) * n_leaves)
        per_det_max = int(max(pnr) if len(pnr) > 0 else 0)

        # Log probability (clipped)
        prob_clipped = max(float(joint_prob), 1e-300)
        log_prob = -math.log10(prob_clipped)

        return {
            "expectation": exp_val,
            "joint_prob": float(joint_prob),
            "log_prob": log_prob,
            "complexity": complexity,
            "total_measured_photons": total_ph,
            "per_detector_max": per_det_max,
        }

    def scoring_fn_batch(
        self,
        genotypes: np.ndarray,
        rng_key: Any,  # jax.random.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Batched scoring function.
        Returns:
          fitnesses: (batch_size, N_objectives)
          descriptors: (batch_size, N_descriptors)
          extra_scores: dict of arrays
        """
        batch_size = genotypes.shape[0]
        fitnesses = np.zeros((batch_size, 4))  # 4 objectives
        descriptors = np.zeros((batch_size, 3))  # 3 descriptors
        extras = []

        # Parallel evaluation
        if self.backend == "jax" and jax is not None:
            # JAX batch execution
            from src.circuits.jax_runner import jax_scoring_fn_batch

            # Ensure genotypes are JAX array
            g_jax = jnp.array(genotypes)
            op_jax = jnp.array(self.operator)

            # Run batched scoring
            # jax_scoring_fn_batch returns (fitnesses, descriptors)
            # We need to handle potential compilation overhead on first run
            with jax.profiler.TraceAnnotation("jax_scoring_fn_batch"):
                fitnesses_jax, descriptors_jax = jax_scoring_fn_batch(
                    g_jax, self.cutoff, op_jax
                )
                # Force synchronization to ensure it's captured in profile
                fitnesses_jax.block_until_ready()
                descriptors_jax.block_until_ready()

            # Convert back to numpy for QDax/compatibility
            # (QDax might accept JAX arrays, but let's be safe if mixing)
            # Actually QDax runs on JAX, so it expects JAX arrays!
            # But run_mome.py seems to use numpy for adapter interface?
            # "genotypes: np.ndarray" in type hint.
            # If we are in "qdax" mode, genotypes might be JAX arrays.
            # If we are in "random" mode, they are numpy.

            # If input was numpy, return numpy. If jax, return jax?
            # The signature says np.ndarray.
            # But let's check if we need to convert.

            # If we return JAX arrays, QDax is happy.
            # If we return Numpy arrays, QDax might convert them.

            # Let's return JAX arrays if input was JAX, else convert?
            # Or just return JAX arrays and let caller handle.
            # But `extras` needs to be constructed.

            fitnesses = fitnesses_jax
            descriptors = descriptors_jax

            # Extras: we don't have per-individual metrics easily from vmap unless we modify it.
            # For now return list of empty dicts to satisfy random search loop (extras[i]).
            extras = [{} for _ in range(batch_size)]

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
            # 1. Total Photons
            d_total = float(metrics["total_measured_photons"])
            # 2. Max Photons
            d_max = float(metrics["per_detector_max"])
            # 3. Complexity
            d_complex = float(metrics["complexity"])

            descriptors[i, :] = np.array([d_total, d_max, d_complex])
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
):
    """Main runner supporting both QDax MOME and random search baseline."""
    np.random.seed(seed)

    # Setup
    from src.utils.gkp_operator import construct_gkp_operator

    composer = Composer(cutoff=cutoff, backend=backend)
    topology = SuperblockTopology.build_layered(2)

    # Construct GKP operator
    # Note: construct_gkp_operator handles backend-specific return types (numpy vs jax)
    # But HanamuraMOMEAdapter expects a numpy array for initialization usually?
    # Let's check. The adapter stores it.
    # If backend is jax, we might want jax array.
    # construct_gkp_operator(..., backend=backend) does the right thing.
    operator = construct_gkp_operator(
        cutoff, target_alpha, target_beta, backend=backend
    )

    # Adapter
    adapter = HanamuraMOMEAdapter(
        composer,
        topology,
        operator,
        cutoff=cutoff,
        mode=mode,
        homodyne_resolution=0.01,
        backend=backend,
    )

    D = 256  # genotype dimension (increased for full tree)

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
        from src.circuits.jax_runner import jax_scoring_fn_batch

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
                fitnesses, descriptors = jax_scoring_fn_batch(genotypes, cutoff, op_jax)

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
        genotypes = jax.random.uniform(
            subkey, (pop_size, D), minval=-2.0, maxval=2.0, dtype=jnp.float32
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

        # Grid-based Centroids
        # D1: Active Modes (Complexity): 1..8 (8 bins)
        # D2: Max PNR: 0..4 (5 bins)
        # D3: Total Photons: 0..24 (25 bins)
        # Total = 8 * 5 * 25 = 1000 cells
        d1 = jnp.linspace(1, 8, 8)
        d2 = jnp.linspace(0, 4, 5)
        d3 = jnp.linspace(0, 24, 25)
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

        print(
            f"Starting optimization: {n_iters} iterations in {n_chunks} chunks (sizes={chunks})."
        )

        start_time = time.time()
        history_fronts = []

        # Target ~50-100 frames for animation to keep overhead low
        snapshot_interval = max(1, n_iters // 50)

        completed = 0

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
                f"ETR: {etr_str} | "
                f"Cov: {cov:.1f}% | "
                f"Best Exp: {best_exp:.4f} | "
                f"Best Prob: {best_prob:.2e}"
            )

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
    }

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
    parser.add_argument("--profile", action="store_true", help="Enable JAX profiling")
    parser.add_argument(
        "--target-beta",
        type=complex,
        default=0.0,
        help="Target superposition beta (complex)",
    )
    args = parser.parse_args()

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
        )


if __name__ == "__main__":
    # Assuming DEFAULT_CUTOFF is defined elsewhere or a reasonable default like 10
    try:
        DEFAULT_CUTOFF
    except NameError:
        DEFAULT_CUTOFF = 10

    main()
