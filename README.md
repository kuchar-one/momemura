# momemura

momemura is a high-performance framework for optimizing GKP (Gottesman-Kitaev-Preskill) state preparation circuits using QDax and JAX. It supports both pure-state analytical pipelines and realistic mixed-state simulations with homodyne detection.

**New:** Now supports **Differentiable Quality Diversity (DQD)** via exact gradient computation and the `OMG-MEGA` emitter!

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/momemura.git
    cd momemura
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: For GPU support, ensure you install the correct JAX version for your CUDA setup (e.g., `pip install "jax[cuda]"`).*

## Usage

### Running Optimization
*Note: Both modes use `float32` precision by default.*

**Arguments:**
- `--mode`: `qdax` (evolutionary) or `random` (baseline).
- `--backend`: `jax` (GPU-accelerated) or `thewalrus` (CPU reference).
- `--pop`: Population size.
- `--iters`: Number of generations.
- `--cutoff`: Fock space truncation cutoff.
- `--genotype`: Genotype design to use: `legacy` (default), `A` (Canonical), `B1`, `B2`, `B3`, `C1`, `C2`. See `docs/jax_genotype_and_circuit.md` for details.
- `--target-alpha`: Target GKP |0> coefficient (default 1.0).
- `--target-beta`: Target GKP |1> coefficient (default 0.0, complex).
- `--low-mem`: Disables JAX memory preallocation to save VRAM (precision remains `float32`).
- `--profile`: Enable JAX profiling (saves trace to `./profiles`).
- `--profile`: Enable JAX profiling (saves trace to `./profiles`).
- `--no-plot`: Disable plotting (useful for clusters).
- `--emitter`: Optimization strategy (Default: `standard`).
    - `standard`: Evolution Strategies (Mixing Emitter).
    - `biased`: Fitness-biased selection.
    - `hybrid`: Combination of Exploration and Intensification.
    - `gradient`: **[New]** Gradient-based OMG-MEGA Emitter (requires JAX).
    - `hybrid-gradient`: **[New]** Hybrid evolution + gradient descent.
    - `mega-hybrid`: **[New]** Nested Hybrid (Gradient + Biased + Standard).

**Parameter Limits:**
- `--depth`: Circuit depth (default 3).
- `--r-scale`: Max squeezing scale (default 2.0, approx 17dB).
- `--d-scale`: Max displacement scale (default 3.0).
- `--hx-scale`: Homodyne X scale (default 4.0).
- `--window`: Homodyne window width (default 0.1).
- `--pnr-max`: Max PNR outcome (default 3).
- `--modes`: Total number of modes (1 Signal + N-1 Controls) to simulate (default 3). Genotypes automatically scale to fit.
- `--seed-scan`: Scan `output/` directory for high-fitness seeds from previous runs and inject them into the initial population (converting genotype designs if needed).
- `--resume`: Path to a previous run directory (e.g., `output/2025...`) to resume optimization from the latest checkpoint.

### Robust Optimization (Watchdog)
For long-running optimizations, especially on multi-GPU setups where hangs can occur, use the **Watchdog** script. This script monitors the main process and restarts it if it hangs or crashes, ensuring the job completes.

```bash
python watchdog_mome.py --mode qdax --backend jax --pop 500 --iters 2000 --cutoff 25 --genotype A --debug
# Add any other MOME arguments as needed.
```
The watchdog automatically handles resuming from the last checkpoint.


**Dynamic Limits (Advanced):**
- `--dynamic-limits`: Enable dynamic parameter limits. Adapts `r_scale` and `d_scale` based on the simulation `cutoff` to allow maximum energy usage while maintaining a safety buffer (50% of cutoff) to minimize truncation artifacts.
- `--correction-cutoff`: (Optional) The larger cutoff dimension used for simulation when dynamic limits are active. Defaults to cutoff + 15.
- Note: When enabled, a leakage penalty (1.0 * leakage) is applied to the fitness function.

- Note: When enabled, a leakage penalty (1.0 * leakage) is applied to the fitness function.

### Differentiable Quality Diversity (DQD)
The framework now supports gradient-based optimization using JAX's auto-differentiation capabilities.
- **Exact Gradients**: The simulation computes the exact gradient of the Expectation Value with respect to all continuous circuit parameters.
- **MOME-OMG-MEGA**: A custom Emitter adapts the [OMG-MEGA](https://arxiv.org/abs/2205.10862) algorithm to the Multi-Objective MAP-Elites setting. It combines stochastic gradient descent (via random coefficient scaling) with the archive-based diversity maintenance of MOME.

To use DQD:
```bash
python run_mome.py --mode qdax --emitter gradient --genotype B30B
```

To profile the optimization loop:
```bash
python scripts/profile_mome.py
```

## Frontend Visualization

Momemura includes an interactive Streamlit frontend for analyzing optimization results.

```bash
streamlit run frontend/app.py
```

Features:
- **Global Pareto Front**: Explore trade-offs interactively.
- **Detailed Solution View**: Inspect circuit topology, parameters, and **Wigner functions** (simulated vs. target).
- **Heatmaps**: Analyze performace across the complexity/photon-count grid.

See `docs/frontend.md` for details.

## Project Structure

- `src/genotypes`: Genotype definitions (Canonical Designs A, B, C).
- `src/simulation`: Core simulation logic.
  - `jax`: GPU-accelerated runner and modules.
  - `cpu`: CPU reference implementation (TheWalrus/Numpy).
- `src/utils`: Helpers for caching, GKP operators, and result management.
- `tests`: Unified Pytest suite.
- `output`: Optimization results (checkpoints, plots, animations).

## Documentation

See `docs/walkthrough.md` for a detailed guide on the codebase and optimization pipeline.
