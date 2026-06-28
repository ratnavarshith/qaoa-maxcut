"""Plot helpers for QAOA experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .qaoa import expectation_value, maxcut_hamiltonian


def plot_landscape(g, output_dir: str | Path, grid: int = 25) -> Path:
    """p=1 cost landscape over (gamma, beta) ∈ [0, π]^2."""
    H = maxcut_hamiltonian(g)
    gs = np.linspace(0, np.pi, grid)
    bs = np.linspace(0, np.pi, grid)
    Z = np.zeros((grid, grid))
    for i, gam in enumerate(gs):
        for j, bet in enumerate(bs):
            Z[i, j] = expectation_value(g, 1, np.array([gam, bet]), H=H)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cs = ax.contourf(bs, gs, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="<H_C>")
    ax.set_xlabel("beta"); ax.set_ylabel("gamma")
    ax.set_title("QAOA p=1 cost landscape")
    fig.tight_layout()
    out = Path(output_dir) / "landscape.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_depth_sweep(
    ratios: dict[int, float],
    output_dir: str | Path,
    rand_ratio: Optional[float] = None,
    gw_ratio: Optional[float] = None,
) -> Path:
    """Approximation ratio vs QAOA depth, with optional baselines."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ps = sorted(ratios)
    ax.plot(ps, [ratios[p] for p in ps], "o-", label="QAOA + INTERP")
    if gw_ratio is not None:
        ax.axhline(gw_ratio, color="tab:green", ls="--",
                   label=f"GW ({gw_ratio:.3f})")
    if rand_ratio is not None:
        ax.axhline(rand_ratio, color="tab:gray", ls=":",
                   label=f"random ({rand_ratio:.3f})")
    ax.axhline(1.0, color="k", ls="--", alpha=0.3, label="optimum")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("approximation ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(output_dir) / "depth_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_optimization_history(
    histories: dict[int, list[float]],
    output_dir: str | Path,
) -> Path:
    """COBYLA cost-function trace at each depth."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p, hist in sorted(histories.items()):
        ax.plot(hist, label=f"p={p}", alpha=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("<H_C>")
    ax.set_title("COBYLA optimization history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(output_dir) / "optimization_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_noise_sweep(
    p_vals: list[float],
    ratios_by_depth: dict[int, list[float]],
    output_dir: str | Path,
) -> Path:
    """Approximation ratio vs depolarizing rate, one line per QAOA depth."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for depth, ys in sorted(ratios_by_depth.items()):
        ax.plot(p_vals, ys, "o-", label=f"QAOA p={depth}")
    if any(p > 0 for p in p_vals):
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("depolarizing error rate")
    ax.set_ylabel("approximation ratio")
    ax.set_title("QAOA under gate noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(output_dir) / "noise_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
