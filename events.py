"""
events.py
---------
Dynamic event handling. The system reacts to disruptive events by mutating the
live traffic state, after which the optimizer is re-run to adapt signals.

Supported events (problem statement asks for at least one):
    * congestion    - a sudden surge of vehicles on random roads
    * accident      - one road blocked, its queue spills onto neighbours
    * road_closure  - a road is fully closed (removed from routing)
    * emergency     - an emergency vehicle appears (handled via emergency.py)
"""

from __future__ import annotations

import random
from typing import List, Tuple

from traffic_network import TrafficNetwork


EVENT_TYPES = ["none", "congestion", "accident", "road_closure", "emergency"]


def trigger_congestion(net: TrafficNetwork, seed: int | None = None) -> str:
    rng = random.Random(seed)
    roads = [r for r in net.roads.values() if not r.closed]
    picked = rng.sample(roads, k=max(1, len(roads) // 4))
    for road in picked:
        road.density = min(road.capacity, int(road.density * 1.8) + 15)
        road.queue = min(road.capacity, int(road.queue * 1.9) + 10)
    names = ", ".join(f"{r.source}->{r.target}" for r in picked)
    return f"Sudden congestion surge on {len(picked)} roads: {names}"


def trigger_accident(net: TrafficNetwork,
                     seed: int | None = None) -> Tuple[str, Tuple[str, str]]:
    rng = random.Random(seed)
    roads = [r for r in net.roads.values() if not r.closed]
    road = rng.choice(roads)
    road.closed = True
    spilled = road.queue + road.density
    road.density = 0
    road.queue = 0
    # spill onto other approaches of the accident node
    for other in net.approaches(road.target):
        if not other.closed:
            other.queue = min(other.capacity, other.queue + spilled // 3)
    msg = (f"Accident on road {road.source}->{road.target}. Road blocked; "
           f"{spilled} vehicles rerouted to nearby approaches.")
    return msg, (road.source, road.target)


def trigger_road_closure(net: TrafficNetwork,
                         seed: int | None = None) -> Tuple[str, Tuple[str, str]]:
    rng = random.Random(seed)
    roads = [r for r in net.roads.values() if not r.closed]
    road = rng.choice(roads)
    road.closed = True
    road.density = 0
    road.queue = 0
    msg = f"Road closure: {road.source}->{road.target} is now closed."
    return msg, (road.source, road.target)


def reopen_all(net: TrafficNetwork) -> None:
    for road in net.roads.values():
        road.closed = False
