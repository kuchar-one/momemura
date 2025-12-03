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

import argparse
import math
import numpy as np
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

# === Project imports ===
from src.circuits.gaussian_herald_circuit import GaussianHeraldCircuit
from src.circuits.composer import Composer, SuperblockTopology
from src.utils.cache_manager import CacheManager

# Local defaults
DEFAULT_CUTOFF = 6
GLOBAL_CACHE = CacheManager(cache_dir="./.cache", size_limit_bytes=1024 * 1024 * 512)


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
    if homodyne_window < 1e-3:
        homodyne_window = None

    mix_theta = float(np.tanh(g[idx]) * (math.pi / 2))
    idx += 1
    mix_phi = float(np.tanh(g[idx]) * math.pi)
    idx += 1
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
    params: Dict[str, Any], cutoff: int = DEFAULT_CUTOFF
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
# Adapter: HanamuraMOME
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
    ):
        self.composer = composer
        self.topology = topology
        self.operator = operator
        self.cutoff = int(cutoff)
        self.mode = mode
        self.homodyne_resolution = homodyne_resolution
        if hasattr(composer, "cutoff") and composer.cutoff != self.cutoff:
            raise ValueError("composer.cutoff != operator cutoff")

    def evaluate_one(self, genotype: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single genotype and return metrics."""
        params = decode_genotype(genotype, cutoff=self.cutoff)

        # If pure mode, we can optionally enforce n_signal=1 in decode or builder
        # For now, builder enforces it.

        try:
            vec, prob_block = gaussian_block_builder_from_params(
                params, cutoff=self.cutoff
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
        def eval_single(i):
            try:
                metrics = self.evaluate_one(genotypes[i, :])
                return i, metrics, None
            except Exception as e:
                return i, None, str(e)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(eval_single, range(batch_size)))

        for i, metrics, error in results:
            if error:
                # invalid genotype — mark with -inf fitness so the repertoire cell stays empty
                # print(f"DEBUG: Genotype invalid: {error}") # Uncomment for debugging
                fitnesses[i, :] = -np.inf
                # use a descriptor sentinel outside grid bounds so it cannot populate centroids
                descriptors[i, :] = np.array([-9999.0, -9999.0, -9999.0], dtype=float)
                extras.append({"error": error})
                continue

            # Objectives (minimize all)
            # 1. Expectation (minimize)
            f_expect = metrics["expectation"]
            # 2. Log Prob (minimize -logP)
            f_prob = metrics["log_prob"]
            # 3. Complexity (minimize)
            f_complex = float(metrics["complexity"])
            # 4. Total Photons (minimize)
            f_photons = float(metrics["total_measured_photons"])

            fitnesses[i, :] = np.array(
                [-f_expect, -f_prob, -f_complex, -f_photons]
            )  # QDax maximizes, so negate to minimize

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
        # We just return empty dict or minimal info as QDax MOME doesn't store extras in repertoire
        return fitnesses, descriptors, {}


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
    x_bins = np.arange(0, 10)  # Total photons 0..9
    y_bins = np.arange(0, 10)  # Complexity 0..9
    heatmap_data = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)

    for i in range(len(objectives)):
        d_total = int(descriptors[i, 0])
        d_complex = int(descriptors[i, 2])
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
        annot=True,
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
def run(mode: str = "random", n_iters: int = 50, pop_size: int = 16, seed: int = 12345):
    """Main runner supporting both QDax MOME and random search baseline."""
    np.random.seed(seed)

    # Setup
    cutoff = DEFAULT_CUTOFF
    composer = Composer(cutoff=cutoff)
    topology = SuperblockTopology.build_layered(2)
    operator = np.diag(np.arange(cutoff, dtype=float))
    # Adapter
    adapter = HanamuraMOMEAdapter(
        composer, topology, operator, cutoff=cutoff, mode=mode, homodyne_resolution=0.01
    )

    D = 40  # genotype dimension

    if mode == "qdax":
        try:
            import jax
            import jax.numpy as jnp
            from qdax.core.mome import MOME
            from qdax.core.emitters.mutation_operators import (
                polynomial_mutation,
                polynomial_crossover,
            )
            from qdax.core.emitters.standard_emitters import MixingEmitter
        except Exception as e:
            print(f"Error importing jax/qdax: {e}")
            print("Falling back to random mode.")
            mode = "random"

    if mode == "random":
        # Random search baseline
        best = None
        best_f = -1e99
        for it in range(n_iters):
            pop = np.random.randn(pop_size, D)
            fitnesses, descs, extras = adapter.scoring_fn_batch(pop)
            for i in range(pop_size):
                if fitnesses[i, 0] > best_f:
                    best_f = float(fitnesses[i, 0])
                    best = (
                        pop[i].copy(),
                        fitnesses[i].copy(),
                        descs[i].copy(),
                        extras[i],
                    )
            if (it + 1) % 5 == 0:
                print(f"iter {it + 1}/{n_iters}, best f0 so far: {best_f:.6g}")
        print("RANDOM SEARCH DONE. Best found:")
        if best is not None:
            print("  fitness:", best[1])
            print("  descriptor:", best[2])
            print("  metrics:", best[3])
        else:
            print("  No valid solution found.")
        return best

    # QDax MOME mode
    print("Running QDax MOME...")

    # Define scoring function using jax.pure_callback
    def scoring_fn(genotypes, key):
        def numpy_scorer(genotypes_jax, k):
            gen_np = np.asarray(genotypes_jax)
            fitnesses, descriptors, extras = adapter.scoring_fn_batch(gen_np, k)
            return np.asarray(fitnesses), np.asarray(descriptors)

        result_shape = (
            jax.ShapeDtypeStruct((genotypes.shape[0], 4), jnp.float32),  # 4 objectives
            jax.ShapeDtypeStruct((genotypes.shape[0], 3), jnp.float32),  # 3 descriptors
        )
        fitnesses, descriptors = jax.pure_callback(
            numpy_scorer, result_shape, genotypes, key
        )
        return fitnesses, descriptors, {}

    # Custom metrics function
    def custom_metrics(repertoire):
        """Custom metrics for progress tracking."""
        # Coverage
        valid_mask = repertoire.fitnesses[:, 0] != -jnp.inf
        coverage = jnp.sum(valid_mask) / repertoire.fitnesses.shape[0]

        # Best expectation (min expectation = max fitness[0])
        # fitness[0] is -expectation
        min_expectation = -jnp.max(
            jnp.where(valid_mask, repertoire.fitnesses[:, 0], -jnp.inf)
        )

        # Best log prob (min log prob = max fitness[1])
        # fitness[1] is -log_prob
        min_log_prob = -jnp.max(
            jnp.where(valid_mask, repertoire.fitnesses[:, 1], -jnp.inf)
        )

        return {
            "coverage": coverage,
            "min_expectation": min_expectation,
            "min_log_prob": min_log_prob,
        }

    metrics_function = custom_metrics

    # Initialize RNG
    key = jax.random.key(seed)

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
    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_function,
        variation_fn=crossover_function,
        variation_percentage=1.0,
        batch_size=pop_size,
    )

    # Grid-based Centroids
    # D_total: 0..8 (9 bins)
    # D_max: 0..4 (5 bins)
    # Complexity: 1..6 (6 bins)
    # Total = 9 * 5 * 6 = 270
    d1 = jnp.linspace(0, 8, 9)
    d2 = jnp.linspace(0, 4, 5)
    d3 = jnp.linspace(1, 6, 6)
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
    repertoire, emitter_state, init_metrics = mome.init(genotypes, centroids, subkey)

    # Run loop with progress reporting
    # We run in chunks to print metrics
    chunk_size = 10
    n_chunks = n_iters // chunk_size

    all_metrics = {k: [] for k in init_metrics.keys()}

    print(f"Starting optimization: {n_iters} iterations in {n_chunks} chunks.")

    for chunk in range(n_chunks):
        (repertoire, emitter_state, key), metrics = jax.lax.scan(
            mome.scan_update,
            (repertoire, emitter_state, key),
            (),
            length=chunk_size,
        )

        # Accumulate metrics (taking the last value from the chunk)
        for k, v in metrics.items():
            # v is array of shape (chunk_size, ...)
            # We take the last one for printing, but could store all
            all_metrics[k].extend(v)  # Store all history

        # Print progress
        cov = float(metrics["coverage"][-1]) * 100
        best_exp = float(metrics["min_expectation"][-1])
        best_lp = float(metrics["min_log_prob"][-1])

        print(
            f"Iter {(chunk + 1) * chunk_size}/{n_iters} | "
            f"Cov: {cov:.1f}% | "
            f"Best Exp: {best_exp:.4f} | "
            f"Best LogProb: {best_lp:.2f}"
        )

    print("QDax MOME finished.")

    # Convert JAX metrics to numpy for plotting
    final_metrics = {k: np.array(v) for k, v in all_metrics.items()}

    # Visualization
    try:
        plot_mome_results(repertoire, final_metrics)
    except Exception as e:
        print(f"Visualization failed: {e}")

    return repertoire, emitter_state, final_metrics


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["random", "qdax"], default="random", help="Run mode"
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    run(mode=args.mode, n_iters=args.iters, pop_size=args.pop, seed=args.seed)
