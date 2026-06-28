"""Graph families used as Max-Cut instances.

Each generator returns a `networkx.Graph` with integer node labels in
`range(n)`. We expose a registry so the CLI and tests can iterate by name
without hardcoding.
"""
from __future__ import annotations

import networkx as nx


def erdos_renyi(n: int, p: float = 0.5, seed: int = 0) -> nx.Graph:
    """Random graph with edge probability `p`."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def cycle(n: int) -> nx.Graph:
    """Simple cycle on n nodes — Max-Cut optimum is n if n is even, n-1 if odd."""
    return nx.cycle_graph(n)


def complete(n: int) -> nx.Graph:
    """K_n. Max-Cut optimum is floor(n/2) * ceil(n/2)."""
    return nx.complete_graph(n)


def random_regular(n: int, d: int = 3, seed: int = 0) -> nx.Graph:
    """Random d-regular graph on n nodes (requires n*d even)."""
    if (n * d) % 2 != 0:
        raise ValueError(f"random_regular: n*d must be even (got n={n}, d={d}).")
    return nx.random_regular_graph(d, n, seed=seed)


_REGISTRY = {
    "erdos_renyi": erdos_renyi,
    "cycle": cycle,
    "complete": complete,
    "random_regular": random_regular,
}


def available_graphs() -> list[str]:
    return list(_REGISTRY.keys())


def get_graph(name: str, n: int, **kwargs) -> nx.Graph:
    """Build a graph by family name.

    Recognized kwargs:
        erdos_renyi:    p (float), seed (int)
        cycle:          (none)
        complete:       (none)
        random_regular: d (int), seed (int)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown graph family: {name!r}. "
            f"Choose from: {', '.join(_REGISTRY.keys())}.",
        )
    fn = _REGISTRY[name]
    if name == "erdos_renyi":
        return fn(n, p=kwargs.get("p", 0.5), seed=kwargs.get("seed", 0))
    if name == "cycle":
        return fn(n)
    if name == "complete":
        return fn(n)
    if name == "random_regular":
        return fn(n, d=kwargs.get("d", 3), seed=kwargs.get("seed", 0))
    raise AssertionError("unreachable")  # pragma: no cover
