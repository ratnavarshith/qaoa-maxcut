"""Tests for the graph registry."""
import pytest

from qaoa_maxcut.graphs import get_graph, available_graphs


def test_available_graphs_nonempty():
    assert "erdos_renyi" in available_graphs()
    assert "cycle" in available_graphs()
    assert "complete" in available_graphs()
    assert "random_regular" in available_graphs()


def test_erdos_renyi_basic():
    g = get_graph("erdos_renyi", 6, p=0.5, seed=0)
    assert g.number_of_nodes() == 6


def test_cycle_basic():
    g = get_graph("cycle", 6)
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 6


def test_complete_basic():
    g = get_graph("complete", 5)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 10  # n*(n-1)/2


def test_random_regular_valid():
    g = get_graph("random_regular", 6, d=3, seed=0)
    assert g.number_of_nodes() == 6
    assert all(deg == 3 for _, deg in g.degree())


def test_random_regular_rejects_odd_product():
    with pytest.raises(ValueError):
        get_graph("random_regular", 5, d=3)  # 5*3=15 is odd


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        get_graph("not_a_graph", 6)
