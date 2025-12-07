# momemura

momemura is a high-performance framework for optimizing GKP (Gottesman-Kitaev-Preskill) state preparation circuits using QDax and JAX. It supports both pure-state analytical pipelines and realistic mixed-state simulations with homodyne detection.

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

The main entry point is `run_mome.py`.

**Basic Run (QDax + JAX):**
```bash
python run_mome.py --mode qdax --backend jax --pop 100 --iters 500 --cutoff 10
```

**Low-Memory Mode (for limited VRAM):**
```bash
python run_mome.py --mode qdax --backend jax --pop 50 --iters 200 --cutoff 10 --low-mem
```
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
- `--no-plot`: Disable plotting (useful for clusters).

**Parameter Limits:**
- `--depth`: Circuit depth (default 3).
- `--r-scale`: Max squeezing scale (default 2.0, approx 17dB).
- `--d-scale`: Max displacement scale (default 3.0).
- `--hx-scale`: Homodyne X scale (default 4.0).
- `--window`: Homodyne window width (default 0.1).
- `--pnr-max`: Max PNR outcome (default 3).
- `--modes`: Total number of modes (1 Signal + N-1 Controls) to simulate (default 3). Genotypes automatically scale to fit.
- `--seed-scan`: Scan `output/` directory for high-fitness seeds from previous runs and inject them into the initial population (converting genotype designs if needed).

**Dynamic Limits (Advanced):**
- `--dynamic-limits`: Enable dynamic parameter limits. allows the optimizer to explore larger parameter ranges (e.g., r_scale=20, d_scale=20) by stimulating in a larger Hilbert space and penalizing leakage.
- `--correction-cutoff`: (Optional) The larger cutoff dimension used for simulation when dynamic limits are active. Defaults to cutoff + 15.
- Note: When enabled, a leakage penalty (1.0 * leakage) is applied to the fitness function.

### Profiling

To profile the optimization loop:
```bash
python scripts/profile_mome.py
```

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
