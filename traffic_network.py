"""
traffic_network.py
-------------------
Models a multi-intersection urban traffic network using NetworkX.

Each node is an intersection with real-world (lat, lon) coordinates so it can be
drawn on an OpenStreetMap/Folium map. Each directed edge is a road approach that
carries traffic with a density, queue length and capacity.

The network is intentionally small (4-8 intersections) as suggested by the
problem statement, which keeps the QUBO small enough for QAOA on a simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Road:
    """A directed road approach between two intersections."""

    source: str
    target: str
    capacity: int          # max vehicles the road can hold
    density: int = 0       # current vehicles on the road
    queue: int = 0         # vehicles waiting at the stop line
    closed: bool = False   # road closure event

    @property
    def saturation(self) -> float:
        """Fraction of capacity currently used (0..1+)."""
        if self.capacity <= 0:
            return 0.0
        return self.density / self.capacity


@dataclass
class Intersection:
    """A signalised intersection (network node)."""

    node_id: str
    name: str
    lat: float
    lon: float
    # Two logical phases per intersection: NS (0) and EW (1).
    # `green_phase` says which phase currently has the green light.
    green_phase: int = 0
    # Assigned green duration in seconds for the active phase.
    green_time: int = 30
    is_emergency: bool = False  # part of an active green corridor


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class TrafficNetwork:
    """
    A grid-like network of intersections. Default layout is a 2x3 grid (6 nodes)
    placed on a real map area so Folium can render it. You can request 4-8 nodes.
    """

    # A compact downtown-ish area (Bengaluru MG Road region) used only for
    # placing markers on the map; nothing about the logic depends on the city.
    BASE_LAT = 12.9750
    BASE_LON = 77.6050
    LAT_STEP = 0.0040
    LON_STEP = 0.0050

    def __init__(self, n_intersections: int = 6, seed: int | None = None):
        if not (4 <= n_intersections <= 8):
            raise ValueError("n_intersections must be between 4 and 8")
        self.rng = random.Random(seed)
        self.n = n_intersections
        self.graph = nx.DiGraph()
        self.intersections: Dict[str, Intersection] = {}
        self.roads: Dict[Tuple[str, str], Road] = {}
        self._build_grid()
        self.randomize_traffic()

    # ---- construction ---------------------------------------------------- #
    def _grid_shape(self) -> Tuple[int, int]:
        """Pick rows x cols that hold n nodes (as close to a rectangle)."""
        cols = math.ceil(math.sqrt(self.n))
        rows = math.ceil(self.n / cols)
        return rows, cols

    def _build_grid(self) -> None:
        rows, cols = self._grid_shape()
        idx = 0
        positions: Dict[str, Tuple[int, int]] = {}
        for r in range(rows):
            for c in range(cols):
                if idx >= self.n:
                    break
                node_id = f"I{idx + 1}"
                lat = self.BASE_LAT + (rows - 1 - r) * self.LAT_STEP
                lon = self.BASE_LON + c * self.LON_STEP
                self.intersections[node_id] = Intersection(
                    node_id=node_id, name=f"Junction {idx + 1}",
                    lat=lat, lon=lon,
                )
                self.graph.add_node(node_id)
                positions[node_id] = (r, c)
                idx += 1

        # Connect grid neighbours with bidirectional roads.
        coord_to_node = {v: k for k, v in positions.items()}
        for node_id, (r, c) in positions.items():
            for dr, dc in ((0, 1), (1, 0)):
                neigh = coord_to_node.get((r + dr, c + dc))
                if neigh is not None:
                    self._add_road_pair(node_id, neigh)

    def _add_road_pair(self, a: str, b: str) -> None:
        for src, dst in ((a, b), (b, a)):
            cap = self.rng.choice([40, 50, 60, 70])
            road = Road(source=src, target=dst, capacity=cap)
            self.roads[(src, dst)] = road
            self.graph.add_edge(src, dst, road=road)

    # ---- traffic state --------------------------------------------------- #
    def randomize_traffic(self, level: float = 0.5) -> None:
        """
        Populate density/queue for every road.
        `level` (0..1) scales how busy the network is.
        """
        for road in self.roads.values():
            if road.closed:
                road.density = 0
                road.queue = 0
                continue
            base = int(road.capacity * level)
            road.density = max(0, min(road.capacity,
                                      base + self.rng.randint(-10, 15)))
            road.queue = max(0, int(road.density * self.rng.uniform(0.2, 0.6)))

    def approaches(self, node_id: str) -> List[Road]:
        """All incoming roads that feed an intersection."""
        return [self.roads[(u, node_id)]
                for u in self.graph.predecessors(node_id)]

    def phase_load(self, node_id: str) -> Tuple[float, float]:
        """
        Return (ns_load, ew_load) for an intersection based on incoming queues.
        We split approaches into two phases using their geometric direction.
        """
        node = self.intersections[node_id]
        ns = ew = 0.0
        for road in self.approaches(node_id):
            src = self.intersections[road.source]
            d_lat = abs(src.lat - node.lat)
            d_lon = abs(src.lon - node.lon)
            weight = road.queue + road.density * 0.5
            if d_lat >= d_lon:      # mostly north-south approach
                ns += weight
            else:                   # mostly east-west approach
                ew += weight
        return ns, ew

    def total_queue(self) -> int:
        return sum(r.queue for r in self.roads.values())

    def total_density(self) -> int:
        return sum(r.density for r in self.roads.values())

    def summary(self) -> Dict[str, float]:
        return {
            "intersections": self.n,
            "roads": len(self.roads),
            "total_queue": self.total_queue(),
            "total_density": self.total_density(),
            "avg_saturation": (
                sum(r.saturation for r in self.roads.values())
                / max(1, len(self.roads))
            ),
        }
