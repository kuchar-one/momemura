# JAX Optimization Genotype and Circuit Topology

This document details the genotype encoding, parameter mappings, and circuit topology used in the JAX-based optimization backend for the Momemura project. The implementation is primarily located in `src/simulation/jax/runner.py` and `src/genotypes/genotypes.py`.

## Overview

The optimization uses a real-valued vector (genotype) to map to a flexible quantum circuit architecture known as a "Maximal Superblock". The layout assumes a depth-3 binary tree structure leading to 8 leaf nodes.

**Constants:**
- `MAX_MODES`: 3 (1 Signal + 2 Control)
- `MAX_SIGNAL`: 1
- `MAX_CONTROL`: 2
- `MAX_PNR`: 3
- `SCHMIDT_RANK`: 2 (implicit in code, though usually effective rank is 1 for single signal)

## Genotype Designs

We support multiple genotype designs offering different trade-offs between expressivity and search space size.


| Design | Name | Description | Length (Depth 3) |
| :--- | :--- | :--- | :--- |
| **Legacy** | Original | Per-leaf unique parameters (Original logic mapped to canonical). | 256 |
| **A** | Original (Canonical) | Per-leaf unique. 1 TMSS/leaf + Final Gauss. | 162 |
| **B1** | Tied-Leaf (No Active) | Single shared block. 1 TMSS/block + Final Gauss. | 49 |
| **B2** | Tied-Leaf (Active) | Same as B1 but with per-leaf active flags. | 57 |
| **B3** | Semi-Tied (Per-Leaf PNR) | Shared continuous parameters, UNIQUE integer parameters (PNR, NCtrl) per leaf. | 37 (N=3) |
| **C1** | Tied-All (No Active) | Constant structure. | 25 |
| **C2** | Tied-All (Active) | Same as C1 but with per-leaf active flags. | 33 |

### 1. Global Parameters (All Designs)

All designs include global homodyne settings and a **Final Gaussian** block ($F=5$).
The structure scales with the number of control modes $N_C = N_{modes} - 1$.

### 2. Design A - Original (Per-Leaf Unique)

This design allows every leaf block and every mix node to have unique parameters.

**Formula**: $Length = G + L \cdot P_{leaf} + (L-1) \cdot P_{node} + F$
where:
- $P_{node} = 4$ (Mix params)
- $F = 5$ (Final Gaussian)
- $P_{leaf} = 1 (Active) + 1 (N_{Ctrl}) + 1 (TMSS) + 1 (US) + [N_C^2 + 3N_C + 2] (UC, Disp, PNR)$
  $= N_C^2 + 3N_C + 6$

**For N=3 (N_C=2)**: $P_{leaf} = 4 + 6 + 6 = 16$.
Total (Depth 3): $1 + 8 \times 16 + 7 \times 4 + 5 = 162$.

**Structure**:
- **Global**: `homodyne_x`.
- **Leaves (L blocks)**: Each block has $P_{leaf}$ params.
- **Mix Nodes (L-1 nodes)**: Each node has 4 unique params.
- **Final Gaussian**: 5 params (R, Phi, Varphi, Disp_Re, Disp_Im).

---

### 3. Design B - Tied-Leaf (Shared Blocks)

This design broadcasts a SINGLE set of block parameters to ALL leaves.

**Formula (B1)**: $Length = G + BP + (L-1) \cdot P_{node} + F$
where $BP = P_{leaf} - 1$ (No per-block active flag).
$BP = N_C^2 + 3N_C + 5$.

**For N=3**: $BP = 15$.
Total (Depth 3): $1 + 15 + 7 \times 4 + 5 = 49$.

**Formula (B2)**: Adds $L$ active flags. $Length = B1 + L$.
For Depth 3: $49 + 8 = 57$.

**Structure**:
- **Global**: `homodyne_x`.
- **Shared Block**: $BP$ Parameters applied to ALL leaves.
- **Mix Nodes**: $L-1$ Unique nodes.
- **Active Flags** (B2 only): $L$ booleans.
- **Final Gaussian**: 5 params.

### 3b. Design B3 - Semi-Tied (Shared Continuous, Unique Discrete)

This design addresses the redundancy in B2 by separating parameters into "Shared" and "Unique" groups.
- **Shared**: Expensive continuous parameters (TMSS, Unitaries, Displacements).
- **Unique**: Integer/Boolean parameters (Active, NCtrl, PNR).

**Benefit**: Allows every leaf to target a different photon signature (e.g., `|3,0>` or `|1,2>`) while using the same optical hardware settings.

**Formula**: $Length = G + Shared + L \cdot Unique + (L-1) \cdot P_{node} + F$
Where:
- $Shared = 1(TMSS) + 1(US) + Len(UC) + Len(Disp)$
- $Unique = 1(Active) + 1(NCtrl) + Len(PNR)$

**For N=3 Modes (N_C=2)**:
- $Shared = 12$ ($1 + 1 + 4 + 6$).
- $Unique = 4$ ($1 + 1 + 2$).
- Total (Depth 3, 8 leaves): $1 + 12 + 32 + 28 + 5 = 78$.

**For N=2 Modes (N_C=1)**:
- $Shared = 7$.
- $Unique = 3$.
- Total (Depth 2, 4 leaves): $1 + 7 + 12 + 12 + 5 = 37$.

---

### 4. Design C - Tied-All (Shared Blocks & Mixing)

This design forces maximum symmetry. All blocks are identical, and all mix nodes are identical.

**Formula (C1)**: $Length = G + BP + P_{node} + F$
Constant Length: $1 + BP + 4 + 5 = BP + 10$.

**For N=3**: $15 + 10 = 25$.

**Formula (C2)**: Adds $L$ active flags. $Length = C1 + L$.
For Depth 3: $25 + 8 = 33$.

**Structure**:
- **Global**: `homodyne_x`.
- **Shared Block**: $BP$ params.
- **Shared Mix Node**: 4 params applied to ALL mix nodes.
- **Active Flags** (C2 only): $L$ booleans.

---

## Parameter Mappings (New Designs)
 
 New designs (A, B, C) use the following **defaults**, which can be customized via CLI arguments in `run_mome.py`:
 - `H_X_SCALE` (Default `4.0`, CLI `--hx-scale`)
 - `R_SCALE` (Default `2.0`, CLI `--r-scale`)
 - `D_SCALE` (Default `3.0`, CLI `--d-scale`)
 - `MAX_PNR` (Default `3`, CLI `--pnr-max`)
 - `H_WINDOW` (Default `0.1`, CLI `--window`)

### Leaf Block Parameters (Design A, P=16)

Unique per leaf. Includes `Active` flag at index 0.

| Index | Name | Map | Description |
| :--- | :--- | :--- | :--- |
| 0 | `active` | > 0.0 | Boolean Active Flag |
| 1 | `n_ctrl` | Thresholds $\pm 0.33$ | {0, 1, 2} |
| 2 | `tmss_r` | $\tanh \times R\_SCALE$ | Squeezing (Single) |
| 3 | `us_phase` | $\tanh \times \pi/2$ | Signal Phase |
| 4-7 | `uc_params` | $\tanh \times \pi/2$ | Control Unitary (Theta, Phi, Varphi1, Varphi2) |
| 8-9 | `disp_s` | $\tanh \times D\_SCALE$ | Signal Disp (Re, Im) |
| 10-13 | `disp_c` | $\tanh \times D\_SCALE$ | Control Disp (Re1, Im1, Re2, Im2) |
| 14-15 | `pnr` | Integers | {0..`MAX_PNR`} x 2 |

### Shared Block Parameters (Design B/C, BP=15)

Shared across all leaves. **No Active flag** (handled separately or implicitly True).

| Index | Name | Map | Description |
| :--- | :--- | :--- | :--- |
| 0 | `n_ctrl` | Thresholds $\pm 0.33$ | {0, 1, 2} |
| 1 | `tmss_r` | $\tanh \times R\_SCALE$ | Squeezing |
| 2 | `us_phase` | $\tanh \times \pi/2$ | Signal Phase |
| 3-6 | `uc_params` | $\tanh \times \pi/2$ | Control Unitary |
| 7-8 | `disp_s` | $\tanh \times D\_SCALE$ | Signal Disp |
| 9-12 | `disp_c` | $\tanh \times D\_SCALE$ | Control Disp |
| 13-14 | `pnr` | Integers | {0..`MAX_PNR`} |

### Mix Node Parameters (PN = 4)

| Index | Name | Map | Description |
| :--- | :--- | :--- | :--- |
| 0 | `theta` | $\tanh \times \pi/2$ | Mixing Angle |
| 1 | `phi` | $\tanh \times \pi/2$ | Mixing Phase |
| 2 | `varphi` | $\tanh \times \pi/2$ | Output Phase |
| 3 | `source` | Thresholds $\pm 0.33$ | {0=Mix, 1=Left, 2=Right} |

---

## Circuit Evaluation

Evaluation logic remains consistent across designs. State generation (`jax_get_heralded_state`) and Superblock combination (`jax_superblock`) consume the decoded parameters to produce a final state and score.

Fitness objectives:
1. Maximize Expectation (Minimize $-E$).
2. Maximize Probability (Minimize $-\log_{10}(P)$).
3. Minimize Complexity (Active Modes).
4. Minimize Total Photons.

## Dynamic Parameter Limits (Advanced)

The backend supports a **Dynamic Limits** mode to allow exploration of parameter spaces that would normally overflow the finite Fock basis.

### Mechanism: Dual-Cutoff Simulation
When enabled via `--dynamic-limits`:
1.  **Scaling**: Parameter limits (`R_SCALE`, `D_SCALE`) are relaxed significantly (e.g., $2.0 \to 20.0$), allowing the generation of high-energy states.
2.  **Simulation (`correction_cutoff`)**: The circuit is simulated in a larger Hilbert space dimension (Defaults to `cutoff + 15`).
3.  **Leakage Detection**: The system calculates the probability mass "leaking" outside the target `cutoff`.
    $$ Leakage = 1.0 - \sum_{n=0}^{cutoff-1} | \psi_{sim}[n] |^2 $$
4.  **Penalty**: A soft penalty is added to the fitness function (minimizing expectation).
    $$ Penalty = 1.0 \times Leakage $$
5.  **Truncation**: The state is effectively truncated and renormalized to the target `cutoff` for final processing.

This allows the optimizer to find solutions that naturally fit within the bounds without being constrained by artificial parameter clamps.
