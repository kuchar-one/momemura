# Optimization Strategies

momemura supports several optimization strategies (Emitters) powered by QDax. All strategies utilize the Multi-Objective MAP-Elites (MOME) container to maintain a diverse archive of high-quality solutions.

## Combined Optimization Pipeline

For the most robust optimization results, we recommend using the **Combined Pipeline** (`run_pipeline.py`). This script orchestrates a sequence of Single-Objective (Adam) and Multi-Objective (QDAX) runs to progressively explore the trade-off between Expectation Value and Probability.

### Usage
```bash
python run_pipeline.py [run_mome.py arguments...]
```

### Example
```bash
python run_pipeline.py --genotype Design0 --pop 128 --iters 2000
```

### How it Works
The pipeline executes a sweep of 11 steps, varying the objective weights from **100% Exp / 0% Prob** to **0% Exp / 100% Prob**.

For each step:
1.  **Single Objective Phase**: Launches **two parallel** instances of `watchdog_restart.py` (which runs `run_mome.py` in `single` mode). This allows for deep exploration of specific regions of the objective space using gradient descent (Adam).
2.  **MOME Phase**: Launches **one** instance of `watchdog_restart.py` (running `run_mome.py` in `qdax` mode). This ensures diversity handling and broad coverage of the Pareto front.

**Features:**
- **Global Seeding**: All runs are configured to scan previous results for seeds, ensuring that good solutions found in earlier steps (or parallel runs) are propagated.
- **Robustness**: Uses `watchdog_restart.py` to handle stagnation and automatic restarts.
- **Graceful Skipping**: You can skip the current step by sending `SIGUSR1` to the pipeline process. The pipeline will cleanly terminate the running optimization steps (saving progress) and proceed to the next step.

---

## Emitter Types

You can select the emitter using the `--emitter` flag in `run_mome.py`.

### 1. Standard (`--emitter standard`)
Uses the default **Mixing Emitter**.
- **Logic**: Selects parents uniformly from the archive and applies mutation and crossover.
- **Use Case**: General purpose exploration. Good baseline.

### 2. Biased (`--emitter biased`)
Uses the **Biased Mixing Emitter**.
- **Logic**: Selects parents based on a softmax distribution of their fitness (Expectation Value). Higher quality solutions are selected more often.
- **Parameters**: 
    - Temperature control: High temp = uniform, Low temp = exploitative.
    - Supports dynamic temperature scaling based on archive coverage.
- **Use Case**: Faster convergence to high-quality solutions, potentially at the cost of diversity.

### 3. Gradient (`--emitter gradient`)
**[New]** Uses the **MOME-OMG-MEGA Emitter**.
- **Logic**: Differentiable Quality Diversity (DQD).
    1. Selects parents from the archive.
    2. Computes the **Exact Gradient** of the Expectation Value w.r.t parameters ($\nabla_{\theta} E$).
    3. Moves the parent in the direction of improving fitness (descending expectation):
       $$ \theta_{new} = \theta_{old} - \alpha \cdot \nabla_{\theta} E + \epsilon $$
       where $\alpha$ is a random coefficient sampled from $|N(0, \sigma_g)|$ and $\epsilon$ is Gaussian noise.
- **Requirements**: Requires `jax` backend.
- **Use Case**: Fine-tuning and finding local optima efficiently. Can discover high-performance solutions that are hard to hit via random mutation.

### 4. Hybrid (`--emitter hybrid`)
Uses the **Hybrid Emitter**.
- **Logic**: Combines two streams of evolution:
    1. **Exploration Stream**: Standard Mixing Emitter (Uniform selection).
    2. **Intensification Stream**: Biased Mixing Emitter (High pressure).
- **Use Case**: Best of both worlds—maintains broad coverage while pushing for state-of-the-art performance.

### 5. Hybrid Gradient (`--emitter hybrid-gradient`)
**[New]** Uses the Hybrid Emitter with Gradient Descent.
- **Logic**:
    1. **Exploration Stream**: Standard Mixing Emitter.
    2. **Intensification Stream**: MOME-OMG-MEGA Emitter (Gradient Descent).
- **Use Case**: Robust global search (evolution) coupled with efficient local optimization (gradients). Recommended for complex landscapes.

### 6. Mega Hybrid (`--emitter mega-hybrid`)
**[New]** Uses a nested Hybrid Emitter structure.
- **Logic**:
    1. **Intensification Stream (20%)**: Gradient Emitter (OMG-MEGA).
    2. **Exploration Stream (80%)**: Inner Hybrid Emitter.
        - **Intensification (20%)**: Biased Mixing Emitter.
        - **Exploration (80%)**: Standard Mixing Emitter.
- **Use Case**: The ultimate combination of Gradient-based local search, Biased exploitation, and Standard exploration. Designed for maximum performance on difficult problems.

## Technical Details

### Gradient Computation
The gradient is computed using `jax.value_and_grad` on the `expectation_value` component of the fitness. Note that the other objectives (Log Probability, Active Modes) are treated as non-differentiable or auxiliary for the purpose of the gradient step.

### MOME-OMG-MEGA
We adapted the [OMG-MEGA](https://arxiv.org/abs/2205.10862) algorithm for MOME. The standard implementation in QDax maintains a separate archive of gradients, which assumes a single-objective grid. Our implementation (`src.optimization.emitters.MOMEOMGMEGAEmitter`) performs the "Multi-objective Gradient-based Evolution" step directly using the `extras["gradients"]` returned by the scoring function, without needing a parallel archive.
