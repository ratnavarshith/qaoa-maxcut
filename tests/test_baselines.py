"""Tests for classical baselines and helpers."""
import networkx as nx

from qaoa_maxcut.baselines import (
    brute_maxcut, cut_value, random_bitstring_ratio, has_cvxpy,
)


def test_brute_maxcut_cycle_4():
    g = nx.cycle_graph(4)
    assert brute_maxcut(g) == 4


def test_brute_maxcut_cycle_5():
    # Odd cycle: max-cut is 4 (one edge must be uncut).
    g = nx.cycle_graph(5)
    assert brute_maxcut(g) == 4


def test_brute_maxcut_complete_4():
    # K_n: max cut is floor(n/2) * ceil(n/2).
    g = nx.complete_graph(4)
    assert brute_maxcut(g) == 4


def test_cut_value_alternating():
    g = nx.cycle_graph(4)
    # "0101" alternates -> all 4 edges cut.
    assert cut_value(g, "0101") == 4
    # "0000" cuts nothing.
    assert cut_value(g, "0000") == 0


def test_random_bitstring_ratio_in_unit_interval():
    g = nx.cycle_graph(4)
    r = random_bitstring_ratio(g, trials=100, seed=0)
    assert 0 <= r <= 1


def test_has_cvxpy_returns_bool():
    assert isinstance(has_cvxpy(), bool)
