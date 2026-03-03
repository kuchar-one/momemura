"""
Wigner function visualization utilities.

Provides:
  - PlateauTwoSlopeNorm: Custom colormap normalization with a central a plateau.
  - plot_states: Plot multiple states side-by-side.
  - plot_single_state: Plot a single state with colorbar.
  - plot_wigner_with_marginals: Plot Wigner function with marginal distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, gridspec
from matplotlib.colorbar import ColorbarBase
import qutip as qt
import math


def _to_qobj(state):
    """Convert a state to a qutip Qobj if it isn't one already."""
    if isinstance(state, qt.Qobj):
        return state
    arr = np.asarray(state)
    if arr.ndim == 1:
        return qt.Qobj(arr[:, np.newaxis])  # ket
    return qt.Qobj(arr)


try:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage{amsmath}
            \usepackage{physics}
            \usepackage{braket}
        """,
            "axes.titlesize": 14,
        }
    )
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
except Exception as e:
    print(f"Warning: Could not configure LaTeX for matplotlib: {e}")


class PlateauTwoSlopeNorm(colors.TwoSlopeNorm):
    def __init__(self, vcenter, plateau_size, vmin=None, vmax=None):
        super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
        self.plateau_size = plateau_size

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")

        plateau_lower = self.vcenter - self.plateau_size / 2
        plateau_upper = self.vcenter + self.plateau_size / 2

        x_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]
        y_points = [0, 0.5, 0.5, 1]

        result = np.ma.masked_array(
            np.interp(result, x_points, y_points, left=-np.inf, right=np.inf),
            mask=np.ma.getmask(result),
        )

        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result


def plot_states(
    states,
    titles,
    xvec=None,
    yvec=None,
    cmap="inferno",
    vmin=-0.23,
    vmax=0.23,
    vcenter=0,
    plateau_size=0.03,
    figsize=(12, 7.2),
):
    """Plot multiple states in subplots with shared colorbar."""
    if xvec is None:
        xvec = np.linspace(-5, 5, 1000)
    if yvec is None:
        yvec = np.linspace(-5, 5, 1000)

    num_states = len(states)
    rows = math.ceil(num_states / 3)
    cols = 3

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig, width_ratios=[1] * cols + [0.05])

    norm = PlateauTwoSlopeNorm(
        vcenter=vcenter, plateau_size=plateau_size, vmin=vmin, vmax=vmax
    )

    axes = []
    for i in range(num_states):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        W = qt.wigner(_to_qobj(states[i]), xvec, yvec)
        ax.contourf(xvec, yvec, W, 1000, cmap=cmap, norm=norm, zorder=-1)
        ax.grid(False)
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_ylabel("$p$", fontsize=10)
        ax.set_title(titles[i], pad=12)
        ax.set_rasterization_zorder(0)

    cax = fig.add_subplot(gs[:, -1])
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    plt.tight_layout()
    return fig, axes, cbar


def plot_single_state(
    state,
    title=None,
    xvec=None,
    yvec=None,
    cmap="inferno",
    vmin=-0.23,
    vmax=0.23,
    vcenter=0,
    plateau_size=0.03,
    figsize=(5, 4),
):
    """Plot a single state with dedicated colorbar."""
    if xvec is None:
        xvec = np.linspace(-5, 5, 1000)
    if yvec is None:
        yvec = np.linspace(-5, 5, 1000)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05], figure=fig)

    norm = PlateauTwoSlopeNorm(
        vcenter=vcenter, plateau_size=plateau_size, vmin=vmin, vmax=vmax
    )

    ax = fig.add_subplot(gs[0, 0])
    state = _to_qobj(state)
    W = qt.wigner(state, xvec, yvec)
    ax.contourf(xvec, yvec, W, 1000, cmap=cmap, norm=norm, zorder=-1)
    ax.grid(False)
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$p$", fontsize=10)
    if title is not None:
        ax.set_title(title, pad=12)
    ax.set_rasterization_zorder(0)

    cax = fig.add_subplot(gs[0, 1])
    ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    plt.tight_layout()
    return fig, ax, cax


if __name__ == "__main__":
    # Example usage for plotting the ground state of an operator
    from src.utils.gkp_operator import construct_gkp_operator_angle

    # 1. Construct the GKP (+Z) magic operator for N=60
    N = 60
    # +Z is theta=0, phi=0 => alpha=1, beta=0 => state |0_L>
    H = construct_gkp_operator_angle(N, theta=0.0, phi=0.0, backend="numpy")

    # 2. Get the ground state of the Hamiltonian
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]

    # 3. Plot the ground state
    print("Plotting the ground state of the GKP operator...")
    fig, ax, cax = plot_single_state(gs, title=r"GKP $|0_L\rangle$ Wigner Function")
    plt.savefig("example_gkp_groundstate.png", dpi=300)
    print("Saved plot to 'example_gkp_groundstate.png'")
