"""
environmental.py
----------------
Estimates the operational and environmental impact of a signal plan:

    * average vehicle waiting time
    * traffic throughput (vehicles cleared per cycle)
    * idle fuel consumption
    * CO2 emissions

These are simplified but internally consistent models so that improvements from
one signal plan to another are meaningful for comparison. They are NOT meant to
be traffic-engineering-grade absolute values.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from traffic_network import TrafficNetwork
from quantum_optimizer import OptResult


# Simple physical constants for the idle-emission model.
FUEL_IDLE_L_PER_S = 0.0003       # litres burned per second per idling vehicle
CO2_KG_PER_L = 2.31              # kg CO2 per litre of petrol burned
DISCHARGE_RATE = 0.5            # vehicles cleared per second of green per lane


@dataclass
class Metrics:
    avg_wait_s: float
    total_wait_s: float
    throughput: float          # vehicles cleared this cycle
    served_vehicles: int
    waiting_vehicles: int
    fuel_litres: float
    co2_kg: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_plan(net: TrafficNetwork, result: OptResult,
                  cycle_len: int = 60) -> Metrics:
    """
    Evaluate a phase assignment against the current network state.

    For each intersection we look at the served phase (given green) and the
    opposing phase (kept red). Served vehicles are discharged based on green
    time; the rest keep waiting and burn idle fuel.
    """
    served_vehicles = 0
    waiting_vehicles = 0
    total_wait = 0.0

    for node_id, phase in result.assignment.items():
        node = net.intersections[node_id]
        ns, ew = net.phase_load(node_id)
        served_load = ns if phase == 0 else ew
        red_load = ew if phase == 0 else ns

        green = max(1, node.green_time)
        capacity_cleared = DISCHARGE_RATE * green
        cleared = min(served_load, capacity_cleared)

        served_vehicles += int(cleared)
        remaining_on_green = max(0.0, served_load - cleared)

        # everyone on red waits the whole cycle; some on green also wait
        waiting = red_load + remaining_on_green
        waiting_vehicles += int(waiting)

        # average wait for waiters ~ half the red duration + spillover penalty
        red_time = cycle_len - green
        total_wait += red_load * (red_time * 0.5 + 5.0)
        total_wait += remaining_on_green * (green * 0.5)

    avg_wait = total_wait / max(1, waiting_vehicles)
    fuel = total_wait * FUEL_IDLE_L_PER_S
    co2 = fuel * CO2_KG_PER_L

    return Metrics(
        avg_wait_s=round(avg_wait, 2),
        total_wait_s=round(total_wait, 1),
        throughput=round(served_vehicles, 1),
        served_vehicles=served_vehicles,
        waiting_vehicles=waiting_vehicles,
        fuel_litres=round(fuel, 3),
        co2_kg=round(co2, 3),
    )


def improvement(baseline: Metrics, optimized: Metrics) -> Dict[str, float]:
    """Percentage improvement of optimized over baseline (positive = better)."""

    def pct(old: float, new: float, lower_is_better: bool = True) -> float:
        if old == 0:
            return 0.0
        change = (old - new) / old * 100.0
        return round(change if lower_is_better else -change, 1)

    return {
        "waiting_time_pct": pct(baseline.avg_wait_s, optimized.avg_wait_s),
        "total_wait_pct": pct(baseline.total_wait_s, optimized.total_wait_s),
        "throughput_pct": pct(baseline.throughput, optimized.throughput,
                              lower_is_better=False),
        "fuel_pct": pct(baseline.fuel_litres, optimized.fuel_litres),
        "co2_pct": pct(baseline.co2_kg, optimized.co2_kg),
    }
