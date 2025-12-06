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
| **C1** | Tied-All (No Active) | Constant structure. | 25 |
| **C2** | Tied-All (Active) | Same as C1 but with per-leaf active flags. | 33 |

### 1. Global Parameters (All Designs)

All designs include global homodyne settings and a **Final Gaussian** block.

| Parameter | Gene Mapping | Range | Description |
| :--- | :--- | :--- | :--- |
---


### 2. Design A - Original (Per-Leaf Unique)

This design allows every leaf block and every mix node to have unique parameters.

**Formula**: $Length = G + L \cdot P_{leaf} + (L-1) \cdot P_{node} + F$
For Depth 3 ($L=8$): $1 + 8 \times 16 + 7 \times 4 + 5 = 162$.

**Structure**:
- **Global**: `homodyne_x`.
- **Leaves (8 blocks)**: Each block has 16 unique params (Active, N_Ctrl, TMSS, US, UC, Disp, PNR).
- **Mix Nodes (7 nodes)**: Each node has 4 unique params (Theta, Phi, Varphi, Source).
- **Final Gaussian**: 5 params.

---

### 3. Design B - Tied-Leaf (Shared Blocks)

This design broadcasts a SINGLE set of block parameters to ALL leaves. This assumes identical physical resources for generation, but allows unique routing (Mix Nodes).

**Formula (B1)**: $Length = G + BP + (L-1) \cdot P_{node} + F$
For Depth 3: $1 + 15 + 7 \times 4 + 5 = 49$.

**Formula (B2)**: Adds $L$ active flags. $Length = 49 + 8 = 57$.

**Structure**:
- **Global**: `homodyne_x`.
- **Shared Block**: 15 Parameters (N_Ctrl, TMSS, US, UC, Disp, PNR) applied to ALL leaves.
- **Mix Nodes**: 7 Unique nodes (same as Design A).
- **Active Flags** (B2 only): 8 booleans at the end of the genotype to turn generic leaves on/off.
- **Final Gaussian**: 5 params.

---

### 4. Design C - Tied-All (Shared Blocks & Mixing)

This design forces maximum symmetry. All blocks are identical, and all mix nodes are identical.

**Formula (C1)**: $Length = G + BP + P_{node} + F$
Constant Length: $1 + 15 + 4 + 5 = 25$.

**Formula (C2)**: Adds $L$ active flags. $Length = 25 + 8 = 33$.

**Structure**:
- **Global**: `homodyne_x`.
- **Shared Block**: 1 parameter set for all leaves.
- **Shared Mix Node**: 1 parameter set (Theta, Phi, Varphi, Source) applied to ALL 7 mix nodes.
- **Active Flags** (C2 only): 8 booleans.

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
| 2 | `tmss_r` | $\tanh \times 2.0$ | Squeezing (Single) |
| 3 | `us_phase` | $\tanh \times \pi/2$ | Signal Phase |
| 4-7 | `uc_params` | $\tanh \times \pi/2$ | Control Unitary (Theta, Phi, Varphi1, Varphi2) |
| 8-9 | `disp_s` | $\tanh \times 3.0$ | Signal Disp (Re, Im) |
| 10-13 | `disp_c` | $\tanh \times 3.0$ | Control Disp (Re1, Im1, Re2, Im2) |
| 14-15 | `pnr` | Integers | {0..3} x 2 |

### Shared Block Parameters (Design B/C, BP=15)

Shared across all leaves. **No Active flag** (handled separately or implicitly True).

| Index | Name | Map | Description |
| :--- | :--- | :--- | :--- |
| 0 | `n_ctrl` | Thresholds $\pm 0.33$ | {0, 1, 2} |
| 1 | `tmss_r` | $\tanh \times 2.0$ | Squeezing |
| 2 | `us_phase` | $\tanh \times \pi/2$ | Signal Phase |
| 3-6 | `uc_params` | $\tanh \times \pi/2$ | Control Unitary |
| 7-8 | `disp_s` | $\tanh \times 3.0$ | Signal Disp |
| 9-12 | `disp_c` | $\tanh \times 3.0$ | Control Disp |
| 13-14 | `pnr` | Integers | {0..3} |

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
