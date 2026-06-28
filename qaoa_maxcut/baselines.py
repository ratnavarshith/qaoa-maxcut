"""Classical baselines for Max-Cut.

`cvxpy` is treated as an optional dependency: importing this module never
imports cvxpy at module load, and `has_cvxpy()` lets callers branch cleanly.
"""
from __future__ import annotations

import importlib.util
import numpy as np
import networkx as nx


def has_cvxpy() -> bool:
    """True iff cvxpy is importable in the current environment."""
    return importlib.util.find_spec("cvxpy") is not None


def brute_maxcut(g: nx.Graph) -> int:
    """Exact Max-Cut by brute force. O(2^(n-1)) — only safe for small n."""
    n = g.number_of_nodes()
    best = 0
    # Loop only half the assignment space because the cut is symmetric under
    # bit-flip of every vertex.
    for mask in range(1, 2 ** (n - 1)):
        s = format(mask, f"0{n}b")
        cut = sum(1 for i, j in g.edges() if s[i] != s[j])
        if cut > best:
            best = cut
    return best


def cut_value(g: nx.Graph, bitstring: str) -> int:
    """Cut size of an assignment string. Indexing matches `brute_maxcut`."""
    asn = bitstring[::-1]
    return sum(1 for i, j in g.edges() if asn[i] != asn[j])


def random_bitstring_ratio(g: nx.Graph, trials: int = 1000, seed: int = 0) -> float:
    """Best-of-`trials` random bitstrings, normalized by the true Max-Cut."""
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    opt = brute_maxcut(g)
    if opt == 0:
        return 0.0
    best = 0
    for _ in range(trials):
        s = "".join(rng.choice(["0", "1"], size=n))
        cut = sum(1 for i, j in g.edges() if s[i] != s[j])
        if cut > best:
            best = cut
    return best / opt


def goemans_williamson(g: nx.Graph, trials: int = 50, seed: int = 0) -> float:
    """GW SDP relaxation + random hyperplane rounding.

    Solves max sum_{(i,j) in E} (1 - x_i.x_j)/2 with ||x_i||=1 on the unit
    sphere via SDP, then draws random hyperplanes and rounds. The reported
    value is the best cut found across `trials` divided by the true optimum.

    Raises `ImportError` if cvxpy is not installed; callers should guard with
    `has_cvxpy()` first.
    """
    if not has_cvxpy():
        raise ImportError(
            "cvxpy is required for the Goemans-Williamson baseline. "
            "Install with: pip install cvxpy",
        )
    import cvxpy as cp

    n = g.number_of_nodes()
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0] + [X[i, i] == 1 for i in range(n)]
    objective = 0.25 * sum((1 - X[i, j]) for i, j in g.edges())
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    W, V = np.linalg.eigh(X.value)
    W = np.clip(W, 0, None)
    vectors = V @ np.diag(np.sqrt(W))

    rng = np.random.default_rng(seed)
    opt = brute_maxcut(g)
    if opt == 0:
        return 0.0
    best = 0
    for _ in range(trials):
        r = rng.standard_normal(n)
        assignment = (vectors @ r > 0).astype(int)
        cut = sum(1 for i, j in g.edges() if assignment[i] != assignment[j])
        if cut > best:
            best = cut
    return best / opt
