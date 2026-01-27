# JAX Optimization Genotype and Circuit Topology

This document details the genotype encoding, parameter mappings, and circuit topology used in the JAX-based optimization backend for the Momemura project. The implementation is primarily located in `src/simulation/jax/runner.py` and `src/genotypes/genotypes.py`.

## Overview

The optimization uses a real-valued vector (genotype) to map to a flexible quantum circuit architecture known as a "Maximal Superblock". The layout assumes a depth-3 binary tree structure leading to 8 leaf nodes.

**Constants:**
- `MAX_MODES`: 6 (1 Signal + 5 Control)
- `DEFAULT_MODES`: 3 (1 Signal + 2 Control)
- `MAX_PNR`: 3 (Default, configurable)

## Genotype Designs

We support multiple genotype designs offering different trade-offs between expressivity and search space size.

| Design | Name | Description | Length (N=3) |
| :--- | :--- | :--- | :--- |
| **Legacy** | Original | **REMOVED** | N/A |
| **0** | Per-Node Homodyne | Variant of A with independent homodyne detection at each mixing node. | 209 |
| **00B** | 0 + Balanced | Design 0 with fixed 50:50 mixing. | 188 |
| **A** | Original (Canonical) | Per-leaf unique. General Gaussian per leaf. | 203 |
| **B1** | Tied-Leaf (No Active) | Single shared block. 1 GG/block + Final Gauss. | 48 |
| **B2** | Tied-Leaf (Active) | Same as B1 but with per-leaf active flags. | 56 |
| **B3** | Semi-Tied (Per-Leaf PNR) | Shared continuous parameters, UNIQUE discrete parameters (Active, PNR, NCtrl) per leaf. | 77 |
| **B30** | B3 + Per-Node Homodyne | Like B3 but with independent homodyne detection at each mixing node. | 83 |
| **B3B** | B3 + Balanced | B3 with fixed 50:50 mixing. | 56 |
| **B30B** | B30 + Balanced | B30 with fixed 50:50 mixing. | 62 |
| **C1** | Tied-All (No Active) | Constant structure (Shared Leaf, Shared Mix). | 30 |
| **C2** | Tied-All (Active) | Same as C1 but with per-leaf active flags. | 38 |
| **C20** | C2 + Per-Node Homodyne | Like C2 but with independent homodyne detection at each mixing node. | 44 |
| **C2B** | C2 + Balanced | C2 with fixed 50:50 mixing. | 35 |
| **C20B** | C20 + Balanced | C20 with fixed 50:50 mixing. | 41 |

### 1. General Gaussian Leaf ($N$ Modes)

Each leaf prepares an $N$-mode Gaussian state via Bloch-Messiah decomposition:
$$ |\psi\rangle = D(\alpha) U(\theta) S(r) |0\rangle $$

Parameters for $N$ modes:
- **Squeezing ($r$)**: $N$ params.
- **Pass. Unitary ($U$)**: $N^2$ params (Clements phases).
- **Displacement ($D$)**: $2N$ params (Re, Im).
- **Total Continuous**: $N^2 + 3N$.

For Default $N=3$: $9 + 9 = 18$ params.

Auxiliary Discrete Params:
- **Active**: 1 Boolean.
- **NCtrl**: 1 Integer (Active control modes).
- **PNR**: $N-1$ Integers (Photon outcomes).

### 2. Design A - Original (Per-Leaf Unique)

This design allows every leaf block and every mix node to have unique parameters.

**Formula**: $Length = 1(Hom) + L \cdot P_{leaf} + (L-1) \cdot 3 + 5(Final)$
where $P_{leaf} = 1(Act) + 1(NC) + (N-1)(PNR) + (N^2+3N)(GG)$.

**For N=3**: $P_{leaf} = 1 + 1 + 2 + 18 = 22$.
Total (Depth 3): $1 + 8 \times 22 + 7 \times 3 + 5 = 1 + 176 + 21 + 5 = 203$.

---

### 3. Design B - Tied-Leaf (Shared Blocks)

This design broadcasts a SINGLE set of block parameters to ALL leaves.

**Formula (B1)**: $Length = 1 + BP + (L-1) \cdot 3 + 5$
where $BP = P_{leaf} - 1$ (No per-block active flag).
For N=3: $BP = 21$.
Total: $1 + 21 + 21 + 5 = 48$.

**Formula (B2)**: Adds $L$ active flags. $Length = 48 + 8 = 56$.

### 3b. Design B3 - Semi-Tied (Shared Continuous, Unique Discrete)

- **Shared**: GG ($N^2+3N$).
- **Unique**: Active(1) + NCtrl(1) + PNR($N-1$).

**For N=3**:
- $Shared = 18$.
- $Unique = 4$.
- Total: $1 + 18 + 8 \times 4 + 21 + 5 = 77$.

---

### 4. Design C - Tied-All (Shared Blocks & Mixing)

All blocks are identical, and all mix nodes are identical (shared mix params).

**Formula (C1)**: $Length = 1 + BP + 3(SharedMix) + 5$
For N=3: $1 + 21 + 3 + 5 = 30$.

**Formula (C2)**: Adds $L$ active flags. $Length = 30 + 8 = 38$.

---

### 5. Balanced Variants (Suffix B)

New variants with suffix `B` enforce **Balanced Beam Splitters** ($\theta=\pi/4, \phi=0$) for all mixing nodes.
Reduces length by removing mix parameters.

**B3B**: $77 - 21 = 56$.
**C2B**: $38 - 3 = 35$.

---

## Parameter Mappings

- `H_X_SCALE` (Default `4.0`)
- `R_SCALE` (Default `2.0`)
- `D_SCALE` (Default `3.0`)
- `MAX_PNR` (Default `3`)
- `H_WINDOW` (Default `0.1`)

### Leaf Block (General Gaussian)

| Name | Map | Description |
| :--- | :--- | :--- |
| `active` | > 0.0 | Boolean Active Flag |
| `n_ctrl` | Thresholds | {0..N-1} |
| `pnr` | Integers | {0..MAX_PNR} x (N-1) |
| `r` | $\tanh \times R\_SCALE$ | Squeezing Vector (N) |
| `phases` | $(\tanh + 1) \times \pi$ | Unitary Phases (N^2) |
| `disp` | $\tanh \times D\_SCALE$ | Displacement Vector (2N) |

### Mix Node Parameters (PN = 3)

| Name | Map | Description |
| :--- | :--- | :--- |
| `theta` | $\tanh \times \pi/2$ | Mixing Angle |
| `phi` | $\tanh \times \pi/2$ | Mixing Phase |
| `varphi` | $\tanh \times \pi/2$ | Output Phase |

## Dynamic Parameter Limits

The system supports `correction_cutoff` and `pnr_max` expansion via CLI arguments (`--dynamic-limits`), enabling simulation in larger Hilbert spaces to avoid truncation artifacts during optimization.

- **Adaptive Limits**:
  - **Squeezing**: $r_{max} = \text{asinh}(\sqrt{\text{cutoff}/2})$. Prevents squeezing energy from exceeding 50% of the basis capacity.
  - **Displacement**: $d_{max} = \sqrt{\text{cutoff}/2}$. Prevents displacement energy from exceeding 50% of the basis capacity.
  - **PNR Resolution**: Automatically scales `pnr_max` up to 15 based on cutoff to capture higher photon numbers.

## Circuit Logic Details

### Dynamic Mode Allocation (`n_ctrl`)
To improve performance and physical accuracy, the number of modes simulated for each leaf is dynamically determined by its `n_ctrl` parameter:
- **`n_ctrl = 0`**: 1 Mode (Signal only). Prepared as Vacuum $|0\rangle$. (Prob = 1.0).
- **`n_ctrl = 1`**: 2 Modes.
- **`n_ctrl = 2`**: 3 Modes.
This removes the computational overhead of simulating unused control modes and correctly handles the vacuum baseline.

### Inactive Leaves
Leaves flagged as **Inactive** (via genotype boolean) are treated as "Empty":
- They produce a valid `state` vector (Vacuum) but are **skipped** in the mixing logic (treated as identity pass-through or zero-vector depending on context), ensuring they do not affect the main circuit state.
