"""
emergency.py
------------
Emergency Green Corridor System.

Given a start and destination intersection, we compute the fastest route for an
emergency vehicle (ambulance/fire) using a travel-time weighted shortest path on
the NetworkX graph. Edge travel time grows with road saturation, so the router
naturally avoids congested roads and closed roads.

The list of nodes on that route becomes the `emergency_path` that the quantum
optimizer receives, forcing those intersections onto the corridor phase (green
wave). After the vehicle passes, normal optimization is restored.
"""

from __future__ import annotations

from typing import List

import networkx as nx

from traffic_network import TrafficNetwork


FREE_FLOW_TIME = 20.0  # seconds to cross a road with no traffic


def _edge_travel_time(net: TrafficNetwork, u: str, v: str) -> float:
    road = net.roads.get((u, v))
    if road is None or road.closed:
        return float("inf")
    # BPR-style delay: time rises sharply as saturation approaches capacity.
    sat = min(road.saturation, 1.5)
    return FREE_FLOW_TIME * (1.0 + 0.75 * (sat ** 4))


def compute_emergency_route(net: TrafficNetwork, start: str,
                            dest: str) -> List[str]:
    """Return the fastest node path from start to dest, avoiding closures."""
    if start == dest:
        return [start]
    g = nx.DiGraph()
    for (u, v), road in net.roads.items():
        if road.closed:
            continue
        g.add_edge(u, v, weight=_edge_travel_time(net, u, v))
    try:
        return nx.shortest_path(g, start, dest, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def route_travel_time(net: TrafficNetwork, path: List[str]) -> float:
    """Total travel time along a node path (seconds)."""
    return sum(_edge_travel_time(net, u, v) for u, v in zip(path, path[1:]))


def mark_corridor(net: TrafficNetwork, path: List[str]) -> None:
    """Flag intersections on the corridor so the dashboard can highlight them."""
    for node in net.intersections.values():
        node.is_emergency = False
    for n in path:
        if n in net.intersections:
            net.intersections[n].is_emergency = True


def clear_corridor(net: TrafficNetwork) -> None:
    for node in net.intersections.values():
        node.is_emergency = False
