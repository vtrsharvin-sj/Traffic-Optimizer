"""
live_intersection.py
---------------------
Bridges the REAL detection counts (from video_detector) into the quantum
optimization layer.

The four videos map to the four approaches of a single 4-way junction:

        North
          |
  West -- + -- East
          |
        South

The junction runs two phases:
    Phase 0 (North-South green)  serves North + South approaches
    Phase 1 (East-West green)    serves East + West approaches

We also expose a small multi-intersection helper so the emergency green
corridor / QUBO code (which expect several intersections) still works: we model
a short arterial of 4 intersections whose approach loads are seeded from the
four live video counts. This keeps everything driven by real detections while
letting the QAOA optimizer coordinate multiple signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from quantum_optimizer import QUBO, OptResult, solve_annealing, solve_qaoa, \
    QISKIT_AVAILABLE


PHASE_NAME = {0: "North-South", 1: "East-West"}


@dataclass
class LiveJunction:
    """One real 4-way junction fed by four detected vehicle counts."""
    north: int = 0
    south: int = 0
    east: int = 0
    west: int = 0

    def ns_load(self) -> int:
        return self.north + self.south

    def ew_load(self) -> int:
        return self.east + self.west

    def total(self) -> int:
        return self.north + self.south + self.east + self.west

    def busier_phase(self) -> int:
        return 0 if self.ns_load() >= self.ew_load() else 1


def counts_to_junction(counts: Dict[str, int]) -> LiveJunction:
    return LiveJunction(
        north=int(counts.get("North", 0)),
        south=int(counts.get("South", 0)),
        east=int(counts.get("East", 0)),
        west=int(counts.get("West", 0)),
    )


# --------------------------------------------------------------------------- #
# Single-junction QUBO (2 variables: phase choice per direction pair)
# --------------------------------------------------------------------------- #
def build_junction_qubo(j: LiveJunction, emergency_dir: str | None = None,
                        lambda_emerg: float = 100.0) -> QUBO:
    """
    One binary variable x for the junction:
        x = 0 -> serve North-South (pay the East-West queue)
        x = 1 -> serve East-West   (pay the North-South queue)

    cost(x) = (1-x)*ew_load + x*ns_load
            = ew_load + x*(ns_load - ew_load)

    Emergency: if an ambulance approaches from a given direction we bias the
    junction to keep that direction's phase green.
    """
    variables = ["junction_phase"]
    ns, ew = j.ns_load(), j.ew_load()
    linear = {0: float(ns - ew)}
    offset = float(ew)

    if emergency_dir in ("North", "South"):
        linear[0] += lambda_emerg      # push x -> 0 (NS green)
    elif emergency_dir in ("East", "West"):
        linear[0] -= lambda_emerg      # push x -> 1 (EW green)

    return QUBO(variables=variables, linear=linear, quadratic={}, offset=offset)


def optimize_junction(j: LiveJunction, emergency_dir: str | None = None,
                      prefer_quantum: bool = True) -> Tuple[OptResult, int]:
    """
    Solve the single-junction phase choice with QAOA (or annealing fallback).
    Returns the OptResult and the chosen phase (0 NS / 1 EW).
    """
    qubo = build_junction_qubo(j, emergency_dir=emergency_dir)
    if prefer_quantum and QISKIT_AVAILABLE:
        try:
            res = solve_qaoa(qubo, reps=1)
            res.detail = "QAOA chose the green phase from real detection counts."
        except Exception as exc:
            res = solve_annealing(qubo, iters=500)
            res.detail = f"QAOA failed ({exc}); classical annealing used."
    else:
        res = solve_annealing(qubo, iters=500)
        res.detail = ("Qiskit not present - same QUBO solved by classical "
                      "annealing on real detection counts.")
    phase = int(res.assignment["junction_phase"])
    return res, phase


def adaptive_green_time(load_served: int, min_green: int = 8,
                        max_green: int = 45, sec_per_vehicle: float = 2.0) -> int:
    """Green duration grows with the number of vehicles on the served phase."""
    return int(min(max_green, max(min_green, min_green + load_served * sec_per_vehicle)))


# --------------------------------------------------------------------------- #
# Classical baselines for the single junction (for comparison)
# --------------------------------------------------------------------------- #
def fixed_phase(j: LiveJunction) -> Tuple[int, int]:
    """Fixed timing: always NS green, 30 s."""
    return 0, 30


def rule_based_phase(j: LiveJunction) -> Tuple[int, int]:
    """Greedy: serve busier direction pair with adaptive time."""
    phase = j.busier_phase()
    served = j.ns_load() if phase == 0 else j.ew_load()
    return phase, adaptive_green_time(served)
