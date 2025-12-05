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

    *Note: For GPU support, ensure you install the correct JAX version for your CUDA setup (e.g., `pip install "jax[cuda12]"`).*

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

**Arguments:**
- `--mode`: `qdax` (evolutionary) or `random` (baseline).
- `--backend`: `jax` (GPU-accelerated) or `thewalrus` (CPU reference).
- `--pop`: Population size.
- `--iters`: Number of generations.
- `--cutoff`: Fock space truncation cutoff.
- `--target-alpha`: Target GKP |0> coefficient (default 1.0).
- `--target-beta`: Target GKP |1> coefficient (default 0.0).
- `--low-mem`: Disables x64 precision and preallocation to save memory.

### Profiling

To profile the optimization loop:
```bash
python scripts/profile_mome.py
```

## Project Structure

- `src/circuits`: Core quantum circuit logic (`Composer`, `GaussianHeraldCircuit`).
- `src/circuits/jax_composer.py`: JAX-optimized circuit backend.
- `src/utils`: Helpers for caching, GKP operators, and result management.
- `tests`: Pytest suite for correctness and parity.
- `output`: Optimization results (checkpoints, plots, animations).

## Documentation

See `docs/walkthrough.md` for a detailed guide on the codebase and optimization pipeline.
