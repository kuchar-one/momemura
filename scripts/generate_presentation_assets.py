#!/usr/bin/env python3
"""
Generate presentation assets for selected genotypes.

For each genotype:
  - Wigner function plot (matplotlib, inferno colormap with TwoSlopeNorm)
  - Simplified GBS circuit TikZ/LaTeX

For each experiment:
  - Pareto front plot with selected genotypes highlighted

Uses the EXACT same loading pipeline as the Streamlit frontend.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "frontend"))

from src.utils.result_manager import AggregatedOptimizationResult
from src.utils.plotter import PlateauTwoSlopeNorm
import frontend.utils as utils
import qutip as qt

# ---------------------------------------------------------------------------
# Configuration: experiments and genotypes
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "B30_c30_a1p00_b1p41": {
        "path": os.path.join(
            PROJECT_ROOT, "output", "experiments", "B30_c30_a1p00_b1p41"
        ),
        "genotypes": [
            {"idx": 8415, "expected_exp": 0.5058},
            {"idx": 11907, "expected_exp": 0.6590},
        ],
    },
    "B30_c30_a1p41_b1p41": {
        "path": os.path.join(
            PROJECT_ROOT, "output", "experiments", "B30_c30_a1p41_b1p41"
        ),
        "genotypes": [
            {"idx": 6089, "expected_exp": 0.6492},
            {"idx": 10410, "expected_exp": 0.4790},
        ],
    },
    "B30B_c30_a1p00_b1p00": {
        "path": os.path.join(
            PROJECT_ROOT, "output", "experiments", "B30B_c30_a1p00_b1p00"
        ),
        "genotypes": [
            {"idx": 3473, "expected_exp": 0.3971},
            {"idx": 21579, "expected_exp": 0.6256},
        ],
    },
}

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "presentation_assets")
TOLERANCE = 0.002  # tolerance for expectation matching


# ---------------------------------------------------------------------------
# Wigner Plot (matplotlib, inferno, TwoSlopeNorm)
# ---------------------------------------------------------------------------


def plot_wigner_matplotlib(
    psi,
    xvec,
    pvec,
    title,
    save_path,
    cmap="inferno",
    vmin=-0.23,
    vmax=0.23,
    vcenter=0,
    plateau_size=0.03,
):
    """Wigner plot matching src/utils/plotter.plot_single_state exactly."""
    fig = plt.figure(figsize=(5, 4), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05], figure=fig)

    norm = PlateauTwoSlopeNorm(
        vcenter=vcenter, plateau_size=plateau_size, vmin=vmin, vmax=vmax
    )

    ax = fig.add_subplot(gs[0, 0])
    W = qt.wigner(qt.Qobj(psi), xvec, pvec)
    ax.contourf(xvec, pvec, W, 1000, cmap=cmap, norm=norm, zorder=-1)
    ax.grid(False)
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel(r"$p$", fontsize=10)
    ax.set_title(title, pad=12)
    ax.set_rasterization_zorder(0)
    ax.set_aspect("equal")

    cax = fig.add_subplot(gs[0, 1])
    ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Wigner saved: {save_path}")


def compile_circuit_tex(tex_path, out_dir):
    """Compile a TikZ .tex snippet to a standalone PDF and PNG."""
    basename = os.path.splitext(os.path.basename(tex_path))[0]

    # Read the TikZ snippet
    with open(tex_path, "r") as f:
        tikz_code = f.read()

    # Wrap in standalone document
    standalone = (
        r"""\documentclass[border=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{braket}
\begin{document}
"""
        + tikz_code
        + "\n"
        + r"\end{document}"
        + "\n"
    )

    # Write standalone .tex
    standalone_tex = os.path.join(out_dir, f"{basename}_standalone.tex")
    with open(standalone_tex, "w") as f:
        f.write(standalone)

    # Compile with pdflatex
    try:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                out_dir,
                standalone_tex,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        pdf_path = os.path.join(out_dir, f"{basename}_standalone.pdf")
        if os.path.exists(pdf_path):
            # Convert PDF to PNG with pdftoppm
            png_prefix = os.path.join(out_dir, basename)
            subprocess.run(
                ["pdftoppm", "-png", "-r", "600", "-singlefile", pdf_path, png_prefix],
                capture_output=True,
                timeout=15,
            )
            png_path = f"{png_prefix}.png"
            if os.path.exists(png_path):
                print(f"    ✓ Circuit PNG compiled: {png_path}")
            else:
                print(f"    ⚠ PDF created but pdftoppm failed")
        else:
            print(
                f"    ⚠ pdflatex failed: {result.stderr[-200:] if result.stderr else 'unknown error'}"
            )
    except FileNotFoundError:
        print(f"    ⚠ pdflatex not found — skipping compilation")
    except Exception as e:
        print(f"    ⚠ Compilation error: {e}")
    finally:
        # Cleanup auxiliary files
        for ext in [".aux", ".log"]:
            aux = os.path.join(out_dir, f"{basename}_standalone{ext}")
            if os.path.exists(aux):
                os.remove(aux)


# ---------------------------------------------------------------------------
# GBS Circuit TikZ Generator
# ---------------------------------------------------------------------------


def get_active_nonzero_pnrs(params):
    """
    Extract non-zero PNR values from active leaves.
    Returns a list of (photon_number,) for each detected mode across all active leaves.
    """
    leaf_params = params.get("leaf_params", {})
    leaf_active = params.get("leaf_active", [])
    pnr_array = np.array(leaf_params.get("pnr", []))
    n_ctrl_raw = leaf_params.get("n_ctrl", [])

    if hasattr(n_ctrl_raw, "tolist"):
        n_ctrl_list = n_ctrl_raw.tolist()
    elif isinstance(n_ctrl_raw, list):
        n_ctrl_list = n_ctrl_raw
    else:
        n_ctrl_list = [n_ctrl_raw]

    nonzero_pnrs = []

    for i in range(len(leaf_active)):
        # Check if leaf is active (same check as frontend)
        if not leaf_active[i]:
            continue

        # Check squeezing threshold (same as frontend is_leaf_active)
        r_vals = leaf_params.get("r", [[]])[i]
        if hasattr(r_vals, "__len__"):
            r_max = float(np.max(np.abs(np.array(r_vals))))
        else:
            r_max = abs(float(r_vals))
        if r_max < 0.01:
            continue

        # Active leaf — get its PNRs masked by n_ctrl
        n_ctrl = int(n_ctrl_list[i]) if i < len(n_ctrl_list) else pnr_array.shape[1]
        pnr_row = pnr_array[i]

        for j in range(n_ctrl):
            pnr_val = int(pnr_row[j])
            if pnr_val > 0:
                nonzero_pnrs.append(pnr_val)

    return nonzero_pnrs


def generate_circuit_tikz(nonzero_pnrs, genotype_idx, prob=None):
    """
    Generate TikZ code for a simplified GBS circuit.

    The circuit has:
    - (len(nonzero_pnrs) + 1) modes total
    - All inputs are vacuum |0⟩
    - One green unitary box Û_G
    - Top output: arrow out (signal mode)
    - All other outputs: PNR detectors with ⟨n_i| labels
    - P(n_c) probability label on the right
    """
    n_detected = len(nonzero_pnrs)
    n_modes = n_detected + 1  # +1 for the signal output mode

    # Vertical spacing between modes
    spacing = 0.5
    total_height = (n_modes - 1) * spacing

    # Y positions: top mode at top, going down
    y_positions = [total_height / 2 - i * spacing for i in range(n_modes)]

    # Box bounds
    box_left = 0.0
    box_right = 1.0
    box_top = y_positions[0] + 0.2
    box_bottom = y_positions[-1] - 0.2

    lines = []
    lines.append(f"% GBS Circuit for genotype {genotype_idx}")
    lines.append(f"% {n_modes} modes: 1 signal + {n_detected} detected")
    lines.append(r"\begin{tikzpicture}[scale=0.9]")

    # Vacuum inputs
    for i, y in enumerate(y_positions):
        lines.append(
            f"    \\node[font=\\normalsize] at (-1, {y:.2f}) {{$\\ket{{0}}$}};"
        )

    # Input lines
    for i, y in enumerate(y_positions):
        lines.append(f"    \\draw[thick] (-0.8, {y:.2f}) -- ({box_left}, {y:.2f});")

    # Unitary box
    lines.append(
        f"    \\filldraw[fill=green!15, draw=green!60!black, thick] "
        f"({box_left}, {box_bottom:.2f}) rectangle ({box_right}, {box_top:.2f});"
    )
    box_center_y = (box_top + box_bottom) / 2
    lines.append(
        f"    \\node[font=\\Large] at ({(box_left + box_right) / 2}, {box_center_y:.2f}) "
        f"{{$\\hat{{U}}_{{G}}$}};"
    )

    # Output lines
    for i, y in enumerate(y_positions):
        lines.append(f"    \\draw[thick] ({box_right}, {y:.2f}) -- (1.6, {y:.2f});")

    # Top output: arrow out (signal mode)
    lines.append(
        f"    \\draw[thick, -Latex] (1.6, {y_positions[0]:.2f}) -- (1.9, {y_positions[0]:.2f});"
    )

    # PNR detectors for all other outputs
    for i in range(1, n_modes):
        y = y_positions[i]
        pnr_val = nonzero_pnrs[i - 1]
        # Detector arc (half-circle)
        lines.append(
            f"    \\draw[thick, fill=gray!50] (1.6, {y + 0.1:.2f}) arc (90:-90:0.1) -- cycle;"
        )
        # Label
        lines.append(
            f"    \\node[font=\\normalsize] at (2.1, {y:.2f}) {{$\\bra{{{pnr_val}}}$}};"
        )

    # Probability label to the right
    if prob is not None:
        # Format probability in scientific notation for LaTeX
        exp = int(np.floor(np.log10(abs(prob))))
        mantissa = prob / (10**exp)
        prob_label = f"{mantissa:.2f} \\times 10^{{{exp}}}"
        label_x = 3.0
        label_y = (y_positions[0] + y_positions[-1]) / 2
        lines.append(
            f"    \\node[font=\\normalsize, anchor=west] at ({label_x}, {label_y:.2f}) "
            f"{{$P(\\mathbf{{n}}_c) = {prob_label}$}};"
        )

    lines.append(r"\end{tikzpicture}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pareto Front Plot
# ---------------------------------------------------------------------------


def plot_pareto_front(df, highlighted_genotypes, exp_name, save_path):
    """
    Pareto front scatter plot with highlighted genotypes.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # All points
    ax.scatter(
        df["LogProb"],
        df["Expectation"],
        c=df["Complexity"],
        cmap="viridis",
        s=8,
        alpha=0.5,
        edgecolors="none",
        label="All solutions",
    )

    # Highlight selected genotypes
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for i, g in enumerate(highlighted_genotypes):
        idx = g["idx"]  # resolved index
        label_idx = g.get("original_idx", idx)  # original user-provided index
        row = df.iloc[idx]
        color = colors[i % len(colors)]
        ax.scatter(
            row["LogProb"],
            row["Expectation"],
            s=200,
            marker="*",
            c=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label=f"Genotype {label_idx} (Exp={row['Expectation']:.4f})",
        )

    ax.set_xlabel(r"$-\log_{10}(P)$", fontsize=14)
    ax.set_ylabel("Expectation Value", fontsize=14)
    ax.set_title(f"Pareto Front — {exp_name}", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Pareto front saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_experiment_sorted(exp_path):
    """
    Load an aggregated experiment with sub-runs sorted alphabetically.
    This ensures deterministic genotype indices regardless of filesystem order.
    """
    from pathlib import Path

    experiment_path = Path(exp_path)
    sub_run_dirs = []

    for item in sorted(os.listdir(experiment_path)):  # sorted for determinism
        sub_path = experiment_path / item
        if sub_path.is_dir() and (sub_path / "results.pkl").exists():
            sub_run_dirs.append(str(sub_path))

    if not sub_run_dirs:
        raise ValueError(f"No valid runs found in {experiment_path}")

    from src.utils.result_manager import OptimizationResult

    sub_runs = []
    for d in sub_run_dirs:
        try:
            run = OptimizationResult.load(d)
            sub_runs.append(run)
        except Exception as e:
            print(f"  Skipping corrupt run {d}: {e}")

    print(f"  Aggregated {len(sub_runs)} runs (sorted) from {exp_path}")
    return AggregatedOptimizationResult(sub_runs)


def find_genotype(df, result, g_idx, expected_exp):
    """
    Find the correct genotype: try the given index first, then search by expectation.
    Returns (resolved_idx, row, matched_directly).
    """
    # Try direct index first
    if g_idx < len(df):
        row = df.iloc[g_idx]
        actual_exp = float(row["Expectation"])
        if abs(actual_exp - expected_exp) < TOLERANCE:
            return g_idx, row, True

    # Fallback: search by expectation value
    diffs = np.abs(df["Expectation"].values - expected_exp)
    candidates = np.where(diffs < TOLERANCE)[0]

    if len(candidates) == 0:
        # Widen search
        best = np.argmin(diffs)
        print(
            f"    ⚠ No exact match for exp={expected_exp:.4f}. "
            f"Closest: idx={best}, exp={df.iloc[best]['Expectation']:.4f}"
        )
        return best, df.iloc[best], False

    # Among candidates, prefer GlobalDominant ones
    if "GlobalDominant" in df.columns:
        dominant = [c for c in candidates if df.iloc[c]["GlobalDominant"]]
        if dominant:
            candidates = dominant

    # Pick the one with lowest LogProb (best probability)
    best_cand = min(candidates, key=lambda c: df.iloc[c]["LogProb"])
    return best_cand, df.iloc[best_cand], False


def main():
    print("=" * 60)
    print("Generating Presentation Assets")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for exp_name, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'─' * 50}")
        print(f"Experiment: {exp_name}")
        print(f"{'─' * 50}")

        exp_path = exp_cfg["path"]
        if not os.path.isdir(exp_path):
            print(f"  ✗ SKIPPED: Directory not found: {exp_path}")
            continue

        # Output directory
        out_dir = os.path.join(OUTPUT_ROOT, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        # Load with sorted sub-runs for deterministic ordering
        print(f"  Loading experiment data (this may take a moment)...")
        result = load_experiment_sorted(exp_path)
        df = result.get_pareto_front()
        print(f"  Total solutions in Pareto front: {len(df)}")

        # Get config from first run for cutoff/pnr_max
        config = result.runs[0].config
        base_cutoff = int(config.get("cutoff", 12))
        corr_cutoff = config.get("correction_cutoff")
        sim_cutoff = base_cutoff
        if corr_cutoff is not None:
            cc = int(corr_cutoff)
            if cc > base_cutoff:
                sim_cutoff = cc
        pnr_max = int(config.get("pnr_max", 3))
        print(
            f"  Config: cutoff={base_cutoff}, sim_cutoff={sim_cutoff}, pnr_max={pnr_max}"
        )

        # Wigner grid
        grid_size = 200
        limit = 5
        xvec = np.linspace(-limit, limit, grid_size)
        pvec = np.linspace(-limit, limit, grid_size)

        # --- Target state Wigner (ground state of GKP operator) ---
        # Exactly matches frontend/app.py lines 176-230
        from src.utils.gkp_operator import construct_gkp_operator

        t_alpha_str = str(config.get("target_alpha", "2.0"))
        t_beta_str = str(config.get("target_beta", "0.0"))

        t_alpha = complex(t_alpha_str) if "j" in t_alpha_str else float(t_alpha_str)
        t_beta = complex(t_beta_str) if "j" in t_beta_str else float(t_beta_str)

        print(f"  Target: α={t_alpha}, β={t_beta}, cutoff={base_cutoff}")

        op_matrix = construct_gkp_operator(
            base_cutoff, t_alpha, t_beta, backend="thewalrus"
        )
        vals, vecs = np.linalg.eigh(op_matrix)
        ground_state = vecs[:, 0]

        target_path = os.path.join(out_dir, "wigner_target.png")
        plot_wigner_matplotlib(
            ground_state,
            xvec,
            pvec,
            title=f"Target GS — Eig: {vals[0]:.4f}",
            save_path=target_path,
        )

        # Track resolved indices for Pareto front highlighting
        resolved_genotypes = []

        for g_info in exp_cfg["genotypes"]:
            g_idx = g_info["idx"]
            expected_exp = g_info["expected_exp"]

            print(f"\n  Genotype {g_idx} (expected exp={expected_exp:.4f}):")

            # Find correct genotype (direct index or search by expectation)
            resolved_idx, row, matched_directly = find_genotype(
                df, result, g_idx, expected_exp
            )
            actual_exp = float(row["Expectation"])

            if matched_directly:
                print(f"    ✓ Direct index match: exp={actual_exp:.4f}")
            else:
                print(
                    f"    ⚠ Index {g_idx} had exp={float(df.iloc[g_idx]['Expectation']) if g_idx < len(df) else 'N/A'}. "
                    f"Resolved to idx={resolved_idx} with exp={actual_exp:.4f}"
                )

            resolved_genotypes.append(
                {
                    "idx": resolved_idx,
                    "original_idx": g_idx,
                    "expected_exp": expected_exp,
                }
            )

            # Get circuit params (same as frontend)
            params = result.get_circuit_params(resolved_idx)

            # Ensure params have needed defaults (same as app.py lines 82-87)
            params["n_control"] = params.get("n_control", 3)
            params["pnr_outcome"] = params.get(
                "pnr_outcome", [0] * int(params.get("n_control", 1))
            )

            # --- Wigner plot ---
            print(f"    Computing state (sim_cutoff={sim_cutoff})...")
            psi, prob = utils.compute_heralded_state(
                params, cutoff=sim_cutoff, pnr_max=pnr_max
            )
            print(f"    Herald probability: {prob:.4e}")

            # Use original user-provided index for filenames
            wigner_path = os.path.join(out_dir, f"wigner_{g_idx}.png")
            plot_wigner_matplotlib(
                psi,
                xvec,
                pvec,
                title=f"Genotype {g_idx} — Exp: {actual_exp:.4f}",
                save_path=wigner_path,
            )

            # --- Circuit TikZ ---
            nonzero_pnrs = get_active_nonzero_pnrs(params)
            print(f"    Active non-zero PNRs: {nonzero_pnrs}")

            tikz_code = generate_circuit_tikz(nonzero_pnrs, g_idx, prob=prob)
            tikz_path = os.path.join(out_dir, f"circuit_{g_idx}.tex")
            with open(tikz_path, "w") as f:
                f.write(tikz_code)
            print(f"    ✓ Circuit TikZ saved: {tikz_path}")
            compile_circuit_tex(tikz_path, out_dir)

        # --- Pareto front plot ---
        pareto_path = os.path.join(out_dir, "pareto_front.png")
        plot_pareto_front(df, resolved_genotypes, exp_name, pareto_path)

    print(f"\n{'=' * 60}")
    print(f"All done! Assets saved to: {OUTPUT_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
