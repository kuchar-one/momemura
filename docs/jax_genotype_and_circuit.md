# JAX Optimization Genotype and Circuit Topology

This document details the genotype encoding, parameter mappings, and circuit topology used in the JAX-based optimization backend for the Momemura project. The implementation is primarily located in `src/circuits/jax_runner.py`.

## Overview

The optimization uses a fixed-length real-valued vector (genotype) to map to a flexible quantum circuit architecture known as a "Maximal Superblock". The layout assumes a depth-3 binary tree structure leading to 8 leaf nodes.

**Constants:**
- `MAX_MODES`: 3 (1 Signal + 2 Control)
- `MAX_SIGNAL`: 1
- `MAX_CONTROL`: 2
- `MAX_PNR`: 3
- `SCHMIDT_RANK`: 2 (implicit in code, though usually effective rank is 1 for single signal)

## Genotype Structure

The genotype is a 1D `float32` array. If the input genotype is shorter than the target length (256), it is zero-padded.

| Section | Description | Parameter Count | Indices (Approx) |
| :--- | :--- | :--- | :--- |
| **Global** | Homodyne measurement settings | 2 | 0-1 |
| **Mix Nodes** | Internal tree nodes (mixing/routing) | 28 (7 nodes $\times$ 4 params) | 2-29 |
| **Leaves** | Gaussian herald blocks | 136 (8 leaves $\times$ 17 params) | 30-165 |
| **Padding** | Unused / Reserved | Variable | 166-255 |

---

## 1. Global Parameters

These parameters control the final homodyne measurement performed on the output state.

| Parameter | Gene Mapping | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `homodyne_x` | $\tanh(g_0) \times 4.0$ | $[-4.0, 4.0]$ | Center position of the homodyne measurement window (or point). Rounded to 6 decimal places. |
| `homodyne_window` | $|\tanh(g_1) \times 2.0|$ | $[0.0, 2.0]$ | Width of the homodyne window. Rounded to 6 decimal places. |

---

## 2. Mix Nodes (Internal Topology)

There are 7 mix nodes in the binary tree structure. Each node accepts inputs from two children (or sources) and produces one output.

**Per-Node Parameters (4 genes):**

| Parameter | Gene Mapping | Range | Description |
| :--- | :--- | :--- | :--- |
| `theta` | $\tanh(g_{i}) \times \frac{\pi}{2}$ | $[-\frac{\pi}{2}, \frac{\pi}{2}]$ | Beamsplitter mixing angle. |
| `phi` | $\tanh(g_{i+1}) \times \frac{\pi}{2}$ | $[-\frac{\pi}{2}, \frac{\pi}{2}]$ | Beamsplitter phase. |
| `varphi` | $\tanh(g_{i+2}) \times \frac{\pi}{2}$ | $[-\frac{\pi}{2}, \frac{\pi}{2}]$ | Output phase rotation. |
| `source` | Thresholds on $g_{i+3}$ | $\{0, 1, 2\}$ | Source selector. <br> $< -0.33 \to 1$ (Left Child) <br> $> 0.33 \to 2$ (Right Child) <br> Else $\to 0$ (Default/Mix Both) |

---

## 3. Leaf Nodes (Gaussian Blocks)

There are 8 leaf nodes. Each leaf represents a "Gaussian Herald Circuit" that generates a pure state to be fed into the tree.

**Per-Leaf Parameters (17 genes):**

The 17 genes are mapped as follows (indices relative to the start of the leaf block):

### A. Configuration
| Index | Parameter | mapping | Range / Description |
| :--- | :--- | :--- | :--- |
| 0 | `active` | $g > 0.0$ | `bool`. If False, this leaf contributes vacuum/zero state. |
| 1 | `n_ctrl` | Thresholds on $g$ | $\{0, 1, 2\}$ <br> $< -0.33 \to 0$ <br> $> 0.33 \to 2$ <br> Else $\to 1$ |

### B. Resources
| Index | Parameter | Mapping | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| 2-3 | `tmss_r` | $\tanh(g) \times 2.0$ | $[-2.0, 2.0]$ | Squeezing magnitude for TMSS pairs. `tmss_r[0]` used if `n_ctrl >= 1`, `tmss_r[1]` if `n_ctrl >= 2`. |
| 4 | `us_phase` | $\tanh(g) \times \frac{\pi}{2}$ | $[-\frac{\pi}{2}, \frac{\pi}{2}]$ | Phase rotation for the Signal mode. |
| 5-8 | `uc_params` | $\tanh(g) \times \frac{\pi}{2}$ | $[-\frac{\pi}{2}, \frac{\pi}{2}]$ | Unitary parameters for Control modes (2-mode unitary). <br> 5: `theta` <br> 6: `phi` <br> 7-8: `varphi` (2 values) |

### C. Displacements
| Index | Parameter | Mapping | Range |
| :--- | :--- | :--- | :--- |
| 9-10 | `disp_s` | $\tanh(g) \times 3.0$ | $[-3.0, 3.0]$ (Real, Imag) for Signal mode. |
| 11-14 | `disp_c` | $\tanh(g) \times 3.0$ | $[-3.0, 3.0]$ (Real, Imag) for Control modes 1 & 2. |

### D. Measurements
| Index | Parameter | Mapping | Range |
| :--- | :--- | :--- | :--- |
| 15-16 | `pnr` | `round(clip(g, 0, 1) * 3)` | $\{0, 1, 2, 3\}$. Photon number resolution outcome to herald. |

---

## Circuit Construction & Evaluation

### 1. Leaf State Generation (`jax_get_heralded_state`)
For each active leaf:
1. **Vacuum**: Start with $N = 1 + n_{ctrl}$ modes in vacuum.
2. **TMSS**: Apply Two-Mode Squeezing between Signal (Mode 0) and Control modes (1, 2) based on `tmss_r`.
3. **Unitaries**: apply 1-mode rotation to Signal and 2-mode unitary to Controls.
4. **Displacement**: Apply complex displacements `disp_s` and `disp_c`.
5. **Heralding**: Project Control modes onto Fock states specified by `pnr`.
    - If `n_ctrl < 1`, `pnr[0]` is ignored (effectively 0).
    - If `n_ctrl < 2`, `pnr[1]` is effectively 0.
6. **Normalization**: The resulting signal state slice is normalized.

### 2. Superblock Combination (`jax_superblock`)
The leaves are combined according to the tree topology defined by the Mix Nodes.
- **Tree Structure**: A fixed depth-3 binary tree.
- **Propagation**: States propagate from leaves up to the root.
- **Mixing**: At each node, if `source=0`, the two inputs are mixed on a beamsplitter defined by `theta`, `phi` and output phase `varphi`. If `source=1` or `2`, only the left or right input is passed through (with phase applied).

### 3. Scoring
The final state at the root is evaluated against the target operator.
- **Expectation**: $E = \langle \psi_{final} | O | \psi_{final} \rangle$.
- **Fitness Objectives**:
    1. Maximize Expectation (Minimize $-E$).
    2. Maximize Probability (Minimize $-\log_{10}(P)$).
    3. Minimize Complexity (Active Modes).
    4. Minimize Total Photons.
