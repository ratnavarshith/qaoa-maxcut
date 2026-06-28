"""QAOA for Max-Cut with INTERP warm-start.

The implementation is intentionally small. The pieces are:

    maxcut_hamiltonian   build the cost Hamiltonian as a SparsePauliOp
    qaoa_circuit         build the variational ansatz at depth p
    expectation_value    run the AerEstimator on a circuit
    optimize_params      classical outer loop (COBYLA)
    interp_init          Zhou et al. 2020 INTERP warm-start
    run_qaoa_interp      sequential p=1..p_max optimization with INTERP seeding
    sample_distribution  shot-based bitstring distribution for ratio computation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

from .baselines import brute_maxcut, cut_value


@dataclass
class QAOAResult:
    """Outcome of one optimization run at a single depth `p`."""
    p: int
    params: np.ndarray
    cost: float                       # negated minimizer value (i.e., <H_C>)
    history: list[float] = field(default_factory=list)
    distribution: dict[str, float] = field(default_factory=dict)
    approximation_ratio: Optional[float] = None


def maxcut_hamiltonian(g: nx.Graph) -> SparsePauliOp:
    """Cost Hamiltonian for Max-Cut on `g`.

    H_C = sum_{(i,j) in E} (I - Z_i Z_j) / 2
        = -0.5 * sum Z_i Z_j  +  0.5 * |E| * I
    """
    n = g.number_of_nodes()
    terms = []
    for i, j in g.edges():
        s = ["I"] * n
        s[i] = "Z"; s[j] = "Z"
        terms.append(("".join(reversed(s)), -0.5))
    terms.append(("I" * n, 0.5 * g.number_of_edges()))
    return SparsePauliOp.from_list(terms)


def qaoa_circuit(g: nx.Graph, p: int, params: np.ndarray) -> QuantumCircuit:
    """Standard QAOA ansatz: H, then p layers of RZZ-cost + RX-mixer."""
    n = g.number_of_nodes()
    gammas, betas = params[:p], params[p:]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for k in range(p):
        for i, j in g.edges():
            qc.rzz(2 * gammas[k], i, j)
        for q in range(n):
            qc.rx(2 * betas[k], q)
    return qc


def expectation_value(
    g: nx.Graph,
    p: int,
    params: np.ndarray,
    estimator: Optional[EstimatorV2] = None,
    H: Optional[SparsePauliOp] = None,
) -> float:
    """<H_C> at the given parameters using the AerEstimator."""
    if estimator is None:
        estimator = EstimatorV2()
    if H is None:
        H = maxcut_hamiltonian(g)
    circ = qaoa_circuit(g, p, params)
    res = estimator.run([(circ, H)]).result()
    return float(res[0].data.evs)


def optimize_params(
    g: nx.Graph,
    p: int,
    init_params: np.ndarray,
    estimator: Optional[EstimatorV2] = None,
    max_iter: int = 200,
    method: str = "COBYLA",
) -> QAOAResult:
    """Run one round of classical optimization from a given starting point."""
    if estimator is None:
        estimator = EstimatorV2()
    H = maxcut_hamiltonian(g)
    history: list[float] = []

    def cost(x):
        v = expectation_value(g, p, x, estimator, H)
        history.append(v)
        return -v

    out = minimize(cost, x0=init_params, method=method,
                   options={"maxiter": max_iter})
    return QAOAResult(p=p, params=out.x, cost=-out.fun, history=history)


def interp_init(params_p: np.ndarray) -> np.ndarray:
    """INTERP schedule (Zhou et al. 2020).

    Given optimal `(gamma_1..gamma_p, beta_1..beta_p)` at depth `p`, build a
    `(p+1)`-depth starting point by linearly interpolating each schedule.
    """
    p = len(params_p) // 2
    gammas = params_p[:p]
    betas = params_p[p:]
    new_p = p + 1

    def interp(arr):
        new = np.zeros(new_p)
        for i in range(1, new_p + 1):
            left = arr[i - 2] if i >= 2 else 0.0
            right = arr[i - 1] if i - 1 < p else 0.0
            new[i - 1] = ((i - 1) / p) * left + ((p - i + 1) / p) * right
        return new

    return np.concatenate([interp(gammas), interp(betas)])


def run_qaoa_interp(
    g: nx.Graph,
    p_max: int,
    seed: int = 0,
    max_iter: int = 200,
) -> dict[int, QAOAResult]:
    """Sweep p=1..p_max, INTERP-seeding each depth from the previous optimum."""
    rng = np.random.default_rng(seed)
    estimator = EstimatorV2()
    results: dict[int, QAOAResult] = {}
    params = rng.uniform(0, np.pi, size=2)  # p=1 start
    for p in range(1, p_max + 1):
        res = optimize_params(g, p, params, estimator, max_iter)
        results[p] = res
        if p < p_max:
            params = interp_init(res.params)
    return results


def sample_distribution(
    g: nx.Graph,
    p: int,
    params: np.ndarray,
    shots: int = 4096,
    seed: int = 0,
    noise_model=None,
) -> dict[str, float]:
    """Shot-based bitstring distribution from the QAOA circuit at `params`."""
    circ = qaoa_circuit(g, p, params)
    circ.measure_all()
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    tqc = transpile(circ, sim)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    return {k: v / shots for k, v in counts.items()}


def approx_ratio(g: nx.Graph, distribution: dict[str, float]) -> float:
    """E[cut] / true Max-Cut from a sampled distribution."""
    opt = brute_maxcut(g)
    if opt == 0:
        return 0.0
    exp_cut = sum(prob * cut_value(g, b) for b, prob in distribution.items())
    return exp_cut / opt
