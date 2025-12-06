# Hanamura Walkthrough

This guide provides an overview of the Hanamura codebase and the GKP state optimization pipeline.

## Architecture Overview

Hanamura uses a hybrid architecture to optimize quantum circuits:

1.  **Evolutionary Engine (QDax)**:
    - Uses Multi-Objective MAP-Elites (MOME) to explore the parameter space.
    - Optimizes for: Expectation Value (minimize -Exp), Log Probability (minimize -LogP), Complexity (minimize), and Photon Count (minimize).
    - Located in `run_mome.py` (adapter logic).

2.  **Circuit Composer (`src/circuits/composer.py`)**:
    - Defines the high-level circuit structure (`GaussianHeraldCircuit`, `SuperblockTopology`).
    - Manages the "genotype to phenotype" mapping (parameters -> circuit components).
    - Handles CPU-based simulation via `thewalrus`.

3.  **JAX Backend (`src/circuits/jax_composer.py`, `jax_runner.py`)**:
    - High-performance, GPU-accelerated simulation.
    - `jax_superblock`: Implements the core mixing logic (Depth-3 tree) using vectorized JAX operations (`vmap`).
    - `jax_herald`: Computes heralded state amplitudes.
    - `jax_runner`: Batches evaluations for QDax.

## Optimization Pipeline

1.  **Initialization**:
    - `run_mome.py` initializes a population of random genotypes.
    - Centroids for the MAP-Elites grid are generated based on descriptors (Complexity, Max PNR, Total Photons).

2.  **Evaluation Loop**:
    - Genotypes are decoded into circuit parameters (`decode_genotype`).
    - **JAX Mode**:
        - `jax_scoring_fn_batch` compiles the entire evaluation (circuit construction + simulation + expectation value).
        - `jax_superblock` processes the circuit layers efficiently using `vmap`.
    - **CPU Mode**:
        - `HanamuraMOMEAdapter` uses `ThreadPoolExecutor` to run `thewalrus` simulations in parallel.

3.  **Result Management**:
    - Results are stored in `OptimizationResult` (pickled).
    - `run_mome.py` tracks the Pareto front and metrics history.
    - An animation (`history.gif`) is generated to visualize the evolution of the Pareto front.

## Key Files

-   `run_mome.py`: Main entry point. Configures QDax and runs the loop.
-   `src/circuits/jax_composer.py`: Contains the optimized `jax_superblock` and `jax_u_bs` functions.
-   `src/circuits/jax_herald.py`: JAX implementation of the herald circuit (recurrence relations).
-   `src/utils/result_manager.py`: Handles saving/loading and visualization.

## Performance Tips

-   **GPU Utilization**: Use `--backend jax` and increase `--pop` (e.g., 100-500) to saturate the GPU.
-   **Memory**: Use `--low-mem` if you have limited VRAM (<8GB). This disables preallocation (precision is always `float32`).
-   **Chunking**: The loop automatically adjusts chunk size. If you see OOM errors, try reducing `--pop` or manually lowering `chunk_size` in `run_mome.py`.
