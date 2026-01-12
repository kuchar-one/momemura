# Hanamura Walkthrough

This guide provides an overview of the Hanamura codebase and the GKP state optimization pipeline.

## Architecture Overview

Hanamura uses a hybrid architecture to optimize quantum circuits:

1.  **Evolutionary Engine (QDax)**:
    - Uses Multi-Objective MAP-Elites (MOME) to explore the parameter space.
    - Optimizes for: Expectation Value (minimize -Exp), Log Probability (minimize -LogP), Complexity (minimize), and Photon Count (minimize).
    - Located in `run_mome.py` (adapter logic).

2.  **Circuit Composer (`src/simulation/cpu/composer.py`)**:
    - Defines the high-level circuit structure (`GaussianHeraldCircuit`, `SuperblockTopology`).
    - Manages the "genotype to phenotype" mapping (parameters -> circuit components).
    - Handles CPU-based simulation via `thewalrus`.

3.  **JAX Backend (`src/simulation/jax/composer.py`, `runner.py`)**:
    - High-performance, GPU-accelerated simulation.
    - `jax_superblock`: Implements the core mixing logic (Depth-3 tree) using vectorized JAX operations (`vmap`).
    - `jax_herald`: Computes heralded state amplitudes using **numerically stable recurrence relations** (normalized to prevent underflow/overflow).
    - `jax_runner`: Batches evaluations for QDax.

## Optimization Pipeline

1.  **Initialization**:
    - `run_mome.py` initializes a population of random genotypes.
    - Selects the genotype design (e.g., `--genotype A`) which defines the circuit topology and parameter space.
    - Parameter limits can be customized via CLI (e.g., `--depth 3`, `--r-scale 2.0`, `--d-scale 3.0`).
    - The number of modes can be set with `--modes N` (default 3), allowing scaling to any 1 Signal + (N-1) Control configuration.
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

## Seeding Strategy

The optimization can be seeded to accelerate convergence:
-   **Vacuum Seed**: The first individual is always initialized as a "Vacuum" state (identity parameters), ensuring a reliable baseline.
-   **Result Scanning**: Using `--seed-scan`, the optimizer scans the `output/` directory for high-fitness results from previous runs. It converts them to the current genotype design (e.g., C1 -> A) and injects them into the initial population.

## Key Files

-   `run_mome.py`: Main entry point. Configures QDax and runs the loop.
-   `src/circuits/jax_composer.py`: Contains the optimized `jax_superblock` and `jax_u_bs` functions.
-   `src/circuits/jax_herald.py`: JAX implementation of the herald circuit (recurrence relations).
-   `src/utils/result_manager.py`: Handles saving/loading and visualization.

## Performance Tips

-   **GPU Utilization**: Use `--backend jax` and increase `--pop` (e.g., 100-500) to saturate the GPU.
-   **Memory**: Use `--low-mem` if you have limited VRAM (<8GB). This disables preallocation (precision is always `float32`).
-   **Chunking**: The loop automatically adjusts chunk size. If you see OOM errors, try reducing `--pop` or manually lowering `chunk_size` in `run_mome.py`.
