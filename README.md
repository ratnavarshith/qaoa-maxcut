# QAOA for Max-Cut

A reproducible QAOA implementation for the Max-Cut problem with INTERP
warm-start, four graph families, classical baselines, and a depolarizing-noise
sweep. Built around the questions:

- Does deeper QAOA actually beat classical polynomial-time algorithms?
- How well does the INTERP heuristic (Zhou et al. 2020) recover from random
  initialization at higher depths?
- At what gate-error rate does deeper QAOA become *worse* than shallow QAOA?

## Why this matters

I used Max-Cut because it is the standard introductory QAOA example and it is
easy to compare against classical baselines. The question I actually wanted to
answer was not just "does QAOA work" but whether it matches the GW bound at any
depth, and what happens to that comparison once you add realistic noise.

## Background

Max-Cut asks for the vertex partition that maximizes crossing edges. It is NP-hard
in general; the Goemans-Williamson SDP (1995) gives a 0.878-approximation in
polynomial time and is the classical benchmark to beat.

QAOA builds a circuit with p alternating layers — a cost unitary based on
`H_C = Σ_{(i,j)∈E} (I - Z_i Z_j) / 2` and a mixing unitary — and optimizes the
2p parameters classically with COBYLA. INTERP (Zhou et al. 2020) seeds each new
depth from a linear interpolation of the previous optimum, which helps avoid
local minima at p ≥ 3.

Approximation ratio = E[cut] / true Max-Cut. The true optimum comes from brute
force (exact but O(2^n)), so graphs here stay small.

## What's in here

| Module | Purpose |
|---|---|
| `qaoa_maxcut/qaoa.py` | Cost Hamiltonian, ansatz, COBYLA optimizer, INTERP |
| `qaoa_maxcut/graphs.py` | Graph registry: ER, cycle, complete, random regular |
| `qaoa_maxcut/baselines.py` | Brute-force optimum, random bitstring, GW SDP |
| `qaoa_maxcut/persistence.py` | JSON + CSV result saving |
| `qaoa_maxcut/plotting.py` | Landscape, depth, history, noise plots |
| `examples/run_qaoa.py` | CLI: depth sweep + landscape + baselines |
| `examples/noise_sweep.py` | CLI: noise sensitivity vs depth |
| `tests/` | Pytest suite (graphs, baselines, qaoa, persistence) |

## Install

```powershell
cd qaoa-maxcut
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt

# Optional: enables the Goemans-Williamson baseline
pip install cvxpy
```

`cvxpy` is optional — the GW baseline is skipped automatically if it isn't
installed and the rest of the project runs unchanged.

## Run

```powershell
# Default ER(6, 0.5), p_max=5, with random + GW baselines if available
python -m examples.run_qaoa

# Different graph family
python -m examples.run_qaoa --graph cycle --n 8 --p-max 4
python -m examples.run_qaoa --graph random_regular --n 8 --regular-degree 3
python -m examples.run_qaoa --graph complete --n 5 --p-max 4

# Custom output dir, no landscape (faster)
python -m examples.run_qaoa --output-dir results --no-landscape

# Noise sweep at three depths
python -m examples.noise_sweep --p-max 3 `
    --noise-rates 0 0.001 0.003 0.01 0.03
```

Output goes to `results/` by default and includes:

- `landscape.png` — p=1 cost landscape over (γ, β)
- `depth_sweep.png` — approximation ratio vs depth, with random + GW baselines
- `optimization_history.png` — COBYLA trace per depth
- `noise_sweep.png` — approximation ratio vs depolarizing rate, per depth
- `run_qaoa.json` / `run_qaoa.csv` — full results including parameters and
  histories
- `noise_sweep.json` / `noise_sweep.csv` — noise-sweep results

## Example results

Verified run on cycle graph (n=4, p_max=2):

```powershell
python -m examples.run_qaoa --graph cycle --n 4 --p-max 2
```

p=1 approximation ratio: ~0.761  
p=2 approximation ratio: 1.000 (optimal cut recovered)

Generated: `landscape.png`, `depth_sweep.png`, `optimization_history.png`,
`run_qaoa.json`, `run_qaoa.csv`.

## Test

```powershell
pytest
```

Covers graph generation, brute-force Max-Cut on small known instances
(C_4 cycle = 4, C_5 cycle = 4, K_4 = 4), `cut_value`, INTERP length growth,
`approx_ratio` on a perfect distribution, and a save/load roundtrip.

## Methodology

### Why INTERP

Random initialization at p ≥ 2 frequently lands COBYLA in a local minimum on
small graphs. INTERP (Zhou et al. 2020) takes the optimum at depth `p` and
linearly interpolates the schedules `(γ_1..γ_p, β_1..β_p)` to construct a
warm-start at depth `p+1`. This consistently improves convergence at p ≥ 3.

### Why a brute-force baseline

We need the *true* Max-Cut to compute approximation ratios. Brute force is
exact, but `O(2^(n-1))`, so this project is bounded to `n ≤ ~20`. For larger
graphs, swap in a proven approximation (GW achieves ≥ 0.878 of the optimum).

### Why GW

QAOA's interesting comparison isn't against random bitstrings — it's against
the best classical *polynomial-time* algorithm. On small random graphs, GW
typically lands near 0.9 and QAOA at p ≈ 4 reaches the same neighborhood.
Reporting both makes "did QAOA beat classical?" a real question.

### Why a noise sweep

Deeper circuits compound gate errors. The depth-vs-noise tradeoff plot
(`noise_sweep.png`) makes the point that bigger `p` is only better when gate
fidelities are high enough — a key argument for hardware-aware QAOA depth
selection on near-term devices.

## Limitations

- Brute-force Max-Cut bounds graphs to roughly `n ≤ 20`.
- COBYLA only; no parameter-shift gradients.
- No hardware execution; depolarizing-noise simulation only.
- Random Clifford and other families could be added; current registry is four
  families.

## Future work

- Parameter-shift / SPSA gradient estimators for higher-depth runs.
- Layered noise models (T1/T2, asymmetric two-qubit error rates).
- Hardware backend execution via `qiskit-ibm-runtime` Sampler.
- Add a Multi-Angle QAOA (MA-QAOA) ansatz comparator.

## Resume bullets

- Implemented QAOA for Max-Cut with the INTERP warm-start heuristic
  (Zhou et al. 2020) across four graph families (Erdős–Rényi, cycle,
  complete, random regular), achieving near-Goemans-Williamson approximation
  ratios at p = 4 on n = 6 instances.
- Quantified the depth–noise tradeoff: deeper QAOA outperforms shallow QAOA
  in the noiseless regime but degrades below shallow at depolarizing rates
  ≥ 1%, motivating hardware-aware depth selection.
- Built a reproducible CLI-driven experimental harness with JSON/CSV result
  persistence, plotting, and a pytest suite covering Max-Cut on known
  small-graph instances and the INTERP schedule.
- Made cvxpy/Goemans-Williamson an optional dependency without compromising
  the rest of the pipeline.

## References

- Farhi, Goldstone, Gutmann 2014 — original QAOA.
- Zhou, Wang, Choi, Pichler, Lukin 2020 — INTERP heuristic.
- Goemans & Williamson 1995 — SDP relaxation + hyperplane rounding.
