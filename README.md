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
- `--genotype`: Genotype design to use: `legacy` (default), `A` (Canonical), `B1`, `B2`, `C1`, `C2`. See `docs/jax_genotype_and_circuit.md` for details.
- `--target-alpha`: Target GKP |0> coefficient (default 1.0).
- `--target-beta`: Target GKP |1> coefficient (default 0.0, complex).
- `--low-mem`: Disables JAX memory preallocation to save VRAM (precision remains `float32`).
- `--profile`: Enable JAX profiling (saves trace to `./profiles`).
- `--no-plot`: Disable plotting (useful for clusters).

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
