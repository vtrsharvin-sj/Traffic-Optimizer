"""
classical_baseline.py
----------------------
Simple classical controllers used as a comparison baseline against the
hybrid quantum-classical (QAOA) optimizer.

Two baselines are provided:

1. Fixed-timing control  -> every intersection always gives green to the
   North-South phase with a fixed 30 s green, ignoring live traffic.

2. Rule-based (greedy) control -> each intersection independently serves
   whichever of its two phases currently has the larger load. This is a
   sensible, purely local heuristic with no coordination between junctions.

The quantum optimizer additionally adds cross-intersection coordination and an
emergency-corridor term, which is where it can beat these baselines.
"""

from __future__ import annotations

from typing import Dict

from traffic_optimizer.traffic.traffic_network import TrafficNetwork
from traffic_optimizer.quantum.quantum_optimizer import OptResult, build_qubo


def solve_fixed(net: TrafficNetwork) -> OptResult:
    """Every intersection: NS green (phase 0), fixed timing."""
    assignment: Dict[str, int] = {n: 0 for n in net.intersections}
    qubo = build_qubo(net)  # score it on the same cost function for fairness
    bits = {i: assignment[qubo.variables[i]] for i in range(len(qubo.variables))}
    energy = qubo.energy(bits)
    return OptResult(assignment, energy, "Fixed timing (classical baseline)",
                     "All signals fixed to North-South, 30 s green.")


def solve_rule_based(net: TrafficNetwork) -> OptResult:
    """Each intersection greedily serves its busier phase (no coordination)."""
    assignment: Dict[str, int] = {}
    for n in net.intersections:
        ns, ew = net.phase_load(n)
        assignment[n] = 0 if ns >= ew else 1
    qubo = build_qubo(net)
    bits = {i: assignment[qubo.variables[i]] for i in range(len(qubo.variables))}
    energy = qubo.energy(bits)
    return OptResult(assignment, energy, "Rule-based greedy (classical baseline)",
                     "Each junction serves its busier direction, no coordination.")


def apply_fixed_timing(net: TrafficNetwork, green: int = 30) -> None:
    for node in net.intersections.values():
        node.green_phase = 0
        node.green_time = green
