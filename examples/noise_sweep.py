"""CLI: how does QAOA approximation ratio degrade with depolarizing noise?

Optimize once noiselessly via INTERP, then evaluate the optimized circuits at
each depth under a sweep of depolarizing rates.

Examples (PowerShell):

    python -m examples.noise_sweep
    python -m examples.noise_sweep --graph random_regular --n 8 --regular-degree 3
    python -m examples.noise_sweep --noise-rates 0 0.001 0.003 0.01 0.03
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qaoa_maxcut.graphs import get_graph, available_graphs
from qaoa_maxcut.qaoa import (
    run_qaoa_interp, sample_distribution, approx_ratio,
)
from qaoa_maxcut.persistence import save_results
from qaoa_maxcut.plotting import plot_noise_sweep


def noise_model(rate: float) -> NoiseModel:
    m = NoiseModel()
    m.add_all_qubit_quantum_error(depolarizing_error(rate, 1),
                                  ["id", "h", "rx", "rz", "sx", "x"])
    m.add_all_qubit_quantum_error(depolarizing_error(rate, 2), ["cx", "rzz"])
    return m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QAOA noise sweep.")
    p.add_argument("--graph", default="erdos_renyi",
                   choices=available_graphs())
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--edge-prob", type=float, default=0.5)
    p.add_argument("--regular-degree", type=int, default=3)
    p.add_argument("--p-max", type=int, default=3,
                   help="Optimize and evaluate depths 1..p_max.")
    p.add_argument("--noise-rates", nargs="+", type=float,
                   default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.05])
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = get_graph(args.graph, args.n,
                  p=args.edge_prob, d=args.regular_degree, seed=args.seed)
    print(f"graph: {args.graph} n={g.number_of_nodes()} "
          f"|E|={g.number_of_edges()}")

    print("INTERP optimization (noiseless)...")
    results = run_qaoa_interp(g, p_max=args.p_max, seed=args.seed)

    ratios_by_depth: dict[int, list[float]] = {}
    summary_rows = []
    for depth in range(1, args.p_max + 1):
        params = results[depth].params
        ys = []
        for rate in args.noise_rates:
            nm = None if rate == 0 else noise_model(rate)
            dist = sample_distribution(g, depth, params, shots=args.shots,
                                       seed=args.seed, noise_model=nm)
            r = approx_ratio(g, dist)
            ys.append(r)
            summary_rows.append({
                "p": depth, "noise_rate": rate, "approximation_ratio": r,
            })
        ratios_by_depth[depth] = ys
        print(f"  p={depth}: ratios={[round(y, 3) for y in ys]}")

    plot_noise_sweep(args.noise_rates, ratios_by_depth, output_dir)
    print(f"  saved {output_dir / 'noise_sweep.png'}")

    payload = {
        "config": vars(args),
        "noise_rates": list(args.noise_rates),
        "ratios_by_depth": {
            str(d): ys for d, ys in ratios_by_depth.items()
        },
        "summary_rows": summary_rows,
    }
    paths = save_results(payload, output_dir, stem="noise_sweep")
    for k, v in paths.items():
        print(f"  saved {v}")


if __name__ == "__main__":
    main()
