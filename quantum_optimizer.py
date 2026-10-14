"""
quantum_optimizer.py
--------------------
Hybrid quantum-classical signal-phase optimizer.

WHERE THE QUANTUM PART LIVES
============================
We choose, for every intersection, which of its two phases (NS or EW) gets the
green light this cycle. That is one binary variable x_i per intersection:

    x_i = 0  -> give green to the North-South phase
    x_i = 1  -> give green to the East-West phase

The cost we want to minimise is the total *unserved* queue (vehicles kept
waiting on the red phase), plus coordination and emergency terms:

    C(x) = sum_i [ (1 - x_i) * ew_load_i + x_i * ns_load_i ]      # unserved queue
         + lambda_coord * coordination_penalty(x)                # green-wave
         + lambda_emerg * emergency_penalty(x)                   # green corridor

This maps directly onto a QUBO / Ising Hamiltonian (linear terms h_i and
quadratic couplings J_ij between neighbouring intersections). We build that
QUBO and hand it to QAOA (Qiskit) running on the Aer simulator.

If Qiskit is not installed, we transparently fall back to classical simulated
annealing over the same QUBO so the app always runs on a plain Windows laptop.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from traffic_network import TrafficNetwork


# --------------------------------------------------------------------------- #
# Optional Qiskit import
# --------------------------------------------------------------------------- #
QISKIT_AVAILABLE = False
try:
    from qiskit_aer.primitives import SamplerV2 as AerSampler  # type: ignore
    from qiskit_optimization import QuadraticProgram  # type: ignore
    from qiskit_optimization.algorithms import MinimumEigenOptimizer  # type: ignore
    from qiskit_algorithms import QAOA  # type: ignore
    from qiskit_algorithms.optimizers import COBYLA  # type: ignore
    from qiskit.primitives import Sampler as RefSampler  # type: ignore

    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    QISKIT_AVAILABLE = False


@dataclass
class QUBO:
    """Quadratic Unconstrained Binary Optimization problem.

    Cost = sum_i linear[i] x_i + sum_{i<j} quad[(i,j)] x_i x_j + offset
    """

    variables: List[str]
    linear: Dict[int, float]
    quadratic: Dict[Tuple[int, int], float]
    offset: float = 0.0

    def energy(self, bits: Dict[int, int]) -> float:
        e = self.offset
        for i, w in self.linear.items():
            e += w * bits[i]
        for (i, j), w in self.quadratic.items():
            e += w * bits[i] * bits[j]
        return e


@dataclass
class OptResult:
    assignment: Dict[str, int]   # node_id -> chosen phase (0 NS / 1 EW)
    energy: float
    method: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# Build the QUBO from live traffic state
# --------------------------------------------------------------------------- #
def build_qubo(net: TrafficNetwork,
               emergency_path: List[str] | None = None,
               lambda_coord: float = 0.15,
               lambda_emerg: float = 50.0) -> QUBO:
    nodes = list(net.intersections.keys())
    index = {n: i for i, n in enumerate(nodes)}

    linear: Dict[int, float] = {i: 0.0 for i in range(len(nodes))}
    quadratic: Dict[Tuple[int, int], float] = {}

    # 1) Serve the busier phase: unserved queue cost.
    #    x_i=0 -> NS green -> EW stays queued -> pay ew_load
    #    x_i=1 -> EW green -> NS stays queued -> pay ns_load
    #    cost_i = (1-x_i)*ew + x_i*ns = ew + x_i*(ns - ew)
    for n in nodes:
        ns, ew = net.phase_load(n)
        i = index[n]
        linear[i] += (ns - ew)          # coefficient of x_i
        # constant `ew` folded into offset below

    offset = sum(net.phase_load(n)[1] for n in nodes)  # sum of ew loads

    # 2) Coordination (green-wave): neighbouring intersections that carry a lot
    #    of shared traffic prefer opposite/aligned phases. We softly penalise
    #    neighbours holding conflicting phases weighted by the road density.
    for (u, v), road in net.roads.items():
        if u not in index or v not in index or road.closed:
            continue
        i, j = index[u], index[v]
        if i >= j:
            continue
        # xor penalty encoded as x_i + x_j - 2 x_i x_j (=1 when they differ)
        w = lambda_coord * road.density
        linear[i] += w
        linear[j] += w
        quadratic[(i, j)] = quadratic.get((i, j), 0.0) - 2 * w

    # 3) Emergency green corridor: intersections on the ambulance path must be
    #    forced onto the phase that serves the corridor direction. We add a
    #    strong linear bias so the optimizer keeps those greens open.
    if emergency_path:
        for a, b in zip(emergency_path, emergency_path[1:]):
            for n in (a, b):
                if n not in index:
                    continue
                i = index[n]
                ns, ew = net.phase_load(n)
                node = net.intersections[n]
                nb = net.intersections[b]
                # decide corridor direction between a and b
                prefer_ew = abs(node.lon - nb.lon) >= abs(node.lat - nb.lat)
                if prefer_ew:
                    linear[i] -= lambda_emerg   # push x_i -> 1 (EW green)
                else:
                    linear[i] += lambda_emerg   # push x_i -> 0 (NS green)

    return QUBO(variables=nodes, linear=linear, quadratic=quadratic, offset=offset)


# --------------------------------------------------------------------------- #
# Brute force (exact) - only for tiny problems, used as ground truth
# --------------------------------------------------------------------------- #
def solve_bruteforce(qubo: QUBO) -> OptResult:
    n = len(qubo.variables)
    best_bits, best_e = None, float("inf")
    for combo in itertools.product([0, 1], repeat=n):
        bits = {i: combo[i] for i in range(n)}
        e = qubo.energy(bits)
        if e < best_e:
            best_e, best_bits = e, bits
    assignment = {qubo.variables[i]: best_bits[i] for i in range(n)}
    return OptResult(assignment, best_e, "Exact (brute force)")


# --------------------------------------------------------------------------- #
# Classical simulated annealing over the QUBO (quantum fallback)
# --------------------------------------------------------------------------- #
def solve_annealing(qubo: QUBO, iters: int = 4000,
                    seed: int | None = None) -> OptResult:
    rng = random.Random(seed)
    n = len(qubo.variables)
    bits = {i: rng.randint(0, 1) for i in range(n)}
    cur = qubo.energy(bits)
    best_bits = dict(bits)
    best_e = cur
    for step in range(iters):
        t = max(0.01, 1.0 - step / iters)
        i = rng.randrange(n)
        bits[i] ^= 1
        e = qubo.energy(bits)
        if e < cur or rng.random() < np.exp(-(e - cur) / (t + 1e-9)):
            cur = e
            if e < best_e:
                best_bits, best_e = dict(bits), e
        else:
            bits[i] ^= 1  # revert
    assignment = {qubo.variables[i]: best_bits[i] for i in range(n)}
    return OptResult(assignment, best_e, "Simulated Annealing (classical fallback)")


# --------------------------------------------------------------------------- #
# QAOA over the QUBO (the quantum path)
# --------------------------------------------------------------------------- #
def solve_qaoa(qubo: QUBO, reps: int = 2, seed: int = 42) -> OptResult:
    """Solve the QUBO with QAOA on the Aer simulator via qiskit-optimization."""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is not available")

    qp = QuadraticProgram("traffic_signal_qubo")
    for v in qubo.variables:
        qp.binary_var(name=v)

    linear = {qubo.variables[i]: qubo.linear.get(i, 0.0)
              for i in range(len(qubo.variables))}
    quadratic = {
        (qubo.variables[i], qubo.variables[j]): w
        for (i, j), w in qubo.quadratic.items()
    }
    qp.minimize(linear=linear, quadratic=quadratic, constant=qubo.offset)

    try:
        sampler = AerSampler()
    except Exception:
        sampler = RefSampler()

    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=reps)
    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)

    assignment = {v: int(round(result.variables_dict[v]))
                  for v in qubo.variables}
    return OptResult(assignment, float(result.fval),
                     f"QAOA (Qiskit Aer, reps={reps})")


# --------------------------------------------------------------------------- #
# Public entry point used by the app
# --------------------------------------------------------------------------- #
def optimize_signals(net: TrafficNetwork,
                     emergency_path: List[str] | None = None,
                     prefer_quantum: bool = True) -> OptResult:
    """
    Build the QUBO from the current network state and solve it.
    Uses QAOA when Qiskit is available, otherwise simulated annealing.
    """
    qubo = build_qubo(net, emergency_path=emergency_path)
    if prefer_quantum and QISKIT_AVAILABLE:
        try:
            res = solve_qaoa(qubo)
            res.detail = "Quantum QAOA optimised the signal-phase QUBO."
            return res
        except Exception as exc:  # pragma: no cover
            fallback = solve_annealing(qubo)
            fallback.detail = f"QAOA failed ({exc}); used classical annealing."
            return fallback
    res = solve_annealing(qubo)
    res.detail = ("Qiskit not installed - solved the same QUBO with "
                  "classical simulated annealing.")
    return res


def apply_assignment(net: TrafficNetwork, result: OptResult,
                     min_green: int = 15, max_green: int = 60) -> None:
    """
    Write the optimizer's phase decision back into the network and compute an
    adaptive green time proportional to the load of the served phase.
    """
    for node_id, phase in result.assignment.items():
        node = net.intersections[node_id]
        node.green_phase = phase
        ns, ew = net.phase_load(node_id)
        served = ns if phase == 0 else ew
        total = ns + ew + 1e-9
        frac = served / total
        node.green_time = int(min_green + frac * (max_green - min_green))
