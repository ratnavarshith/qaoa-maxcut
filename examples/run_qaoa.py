"""CLI entry point: depth sweep + p=1 landscape + classical baselines.

Examples (PowerShell):

    # default: ER(6, 0.5), p_max=5, GW + random baselines if cvxpy installed
    python -m examples.run_qaoa

    # different graph family
    python -m examples.run_qaoa --graph cycle --n 8 --p-max 4

    # custom output directory
    python -m examples.run_qaoa --output-dir results --no-landscape
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qaoa_maxcut.graphs import get_graph, available_graphs
from qaoa_maxcut.qaoa import (
    run_qaoa_interp, sample_distribution, approx_ratio,
)
from qaoa_maxcut.baselines import (
    brute_maxcut, random_bitstring_ratio, goemans_williamson, has_cvxpy,
)
from qaoa_maxcut.persistence import save_results
from qaoa_maxcut.plotting import (
    plot_landscape, plot_depth_sweep, plot_optimization_history,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QAOA Max-Cut depth sweep with INTERP + baselines.")
    p.add_argument("--graph", default="erdos_renyi",
                   choices=available_graphs(),
                   help="Graph family. Default: erdos_renyi.")
    p.add_argument("--n", type=int, default=6, help="Number of nodes.")
    p.add_argument("--edge-prob", type=float, default=0.5,
                   help="Edge probability (erdos_renyi only).")
    p.add_argument("--regular-degree", type=int, default=3,
                   help="Degree d (random_regular only).")
    p.add_argument("--p-max", type=int, default=5,
                   help="Maximum QAOA depth.")
    p.add_argument("--shots", type=int, default=4096,
                   help="Shots for the final distribution.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=200,
                   help="COBYLA max iterations per depth.")
    p.add_argument("--landscape-grid", type=int, default=20,
                   help="Grid resolution for the p=1 landscape.")
    p.add_argument("--no-landscape", action="store_true",
                   help="Skip the landscape plot.")
    p.add_argument("--no-gw", action="store_true",
                   help="Skip the Goemans-Williamson baseline.")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = get_graph(
        args.graph, args.n,
        p=args.edge_prob, d=args.regular_degree, seed=args.seed,
    )
    print(f"graph: {args.graph} n={g.number_of_nodes()} "
          f"|E|={g.number_of_edges()}")
    opt = brute_maxcut(g)
    print(f"true max-cut: {opt}")

    rand_ratio = random_bitstring_ratio(g, trials=2000, seed=args.seed)
    print(f"random bitstring baseline: {rand_ratio:.3f}")

    gw_ratio = None
    if not args.no_gw and has_cvxpy():
        try:
            gw_ratio = goemans_williamson(g, trials=100, seed=args.seed)
            print(f"GW SDP baseline:           {gw_ratio:.3f}")
        except Exception as e:
            print(f"GW SDP failed: {e}")
    elif not args.no_gw:
        print("cvxpy not installed; skipping GW baseline. "
              "Install with: pip install cvxpy")

    if not args.no_landscape:
        print("computing p=1 landscape...")
        plot_landscape(g, output_dir, grid=args.landscape_grid)
        print(f"  saved {output_dir / 'landscape.png'}")

    print("INTERP depth sweep...")
    results = run_qaoa_interp(g, p_max=args.p_max, seed=args.seed,
                              max_iter=args.max_iter)

    ratios: dict[int, float] = {}
    summary_rows = []
    for p, res in results.items():
        dist = sample_distribution(g, p, res.params, shots=args.shots,
                                   seed=args.seed)
        r = approx_ratio(g, dist)
        res.distribution = dist
        res.approximation_ratio = r
        ratios[p] = r
        print(f"  p={p}: cost={res.cost:.3f}  approx ratio={r:.3f}")
        summary_rows.append({
            "p": p,
            "cost": res.cost,
            "approximation_ratio": r,
            "n_iter": len(res.history),
        })

    plot_depth_sweep(ratios, output_dir,
                     rand_ratio=rand_ratio, gw_ratio=gw_ratio)
    print(f"  saved {output_dir / 'depth_sweep.png'}")

    plot_optimization_history({p: r.history for p, r in results.items()},
                              output_dir)
    print(f"  saved {output_dir / 'optimization_history.png'}")

    payload = {
        "config": vars(args),
        "graph": {
            "family": args.graph,
            "n": g.number_of_nodes(),
            "edges": list(g.edges()),
            "max_cut": opt,
        },
        "baselines": {"random": rand_ratio, "gw": gw_ratio},
        "summary_rows": summary_rows,
        "depths": {
            str(p): {
                "params": res.params,
                "cost": res.cost,
                "approximation_ratio": res.approximation_ratio,
                "history": res.history,
            }
            for p, res in results.items()
        },
    }
    paths = save_results(payload, output_dir, stem="run_qaoa")
    for k, v in paths.items():
        print(f"  saved {v}")


if __name__ == "__main__":
    main()
