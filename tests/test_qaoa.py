"""Tests for QAOA construction and INTERP warm-start."""
import numpy as np
import networkx as nx
import pytest

from qaoa_maxcut.qaoa import (
    qaoa_circuit, maxcut_hamiltonian, interp_init, approx_ratio,
)


def test_qaoa_circuit_size():
    g = nx.cycle_graph(4)
    qc = qaoa_circuit(g, p=2, params=np.array([0.1, 0.2, 0.3, 0.4]))
    assert qc.num_qubits == 4


def test_maxcut_hamiltonian_terms():
    g = nx.cycle_graph(4)
    H = maxcut_hamiltonian(g)
    # 4 ZZ terms + 1 identity offset
    assert len(H) == 5


def test_interp_init_length_grows():
    params = np.array([0.5, 0.7])  # p=1
    nxt = interp_init(params)
    assert len(nxt) == 4  # gammas + betas at p=2


def test_interp_init_p2_to_p3():
    params = np.array([0.1, 0.2, 0.3, 0.4])  # p=2
    nxt = interp_init(params)
    assert len(nxt) == 6


def test_approx_ratio_with_perfect_distribution():
    # On C_4, "0101" is an optimal cut (all 4 edges cut). Putting all mass
    # on it should give ratio 1.0.
    g = nx.cycle_graph(4)
    dist = {"0101": 1.0}
    assert approx_ratio(g, dist) == pytest.approx(1.0)


def test_approx_ratio_uniform_distribution_in_range():
    g = nx.cycle_graph(4)
    # Uniform over all 16 bitstrings — ratio must be in [0, 1].
    dist = {format(i, "04b"): 1 / 16 for i in range(16)}
    r = approx_ratio(g, dist)
    assert 0 <= r <= 1
