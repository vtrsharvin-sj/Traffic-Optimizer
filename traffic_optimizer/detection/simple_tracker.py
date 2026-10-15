"""
simple_tracker.py
-----------------
A tiny per-direction centroid tracker so we can use the FAST YOLO `predict`
path (with FP16) instead of the much slower `track` (ByteTrack) call, while
still assigning stable IDs needed for speed estimation.

Matching: each new detection is matched to the nearest existing track whose
centroid is within a distance gate; unmatched detections start new tracks;
tracks not seen for a few frames are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class _Track:
    cx: float
    cy: float
    misses: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_misses: int = 5):
        self.max_distance = max_distance
        self.max_misses = max_misses
        self._tracks: Dict[str, Dict[int, _Track]] = {}
        self._next_id: Dict[str, int] = {}

    def update(self, direction: str,
               boxes: List[Tuple[float, float, float, float]]
               ) -> List[int]:
        """Return a tracker id for each input box (same order)."""
        tracks = self._tracks.setdefault(direction, {})
        if direction not in self._next_id:
            self._next_id[direction] = 1

        centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        assigned: List[int] = [-1] * len(boxes)
        used = set()

        # greedily match each detection to the closest unused track
        for i, (cx, cy) in enumerate(centroids):
            best_id, best_d = None, self.max_distance
            for tid, tr in tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr.cx, cy - tr.cy)
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id is not None:
                tr = tracks[best_id]
                tr.cx, tr.cy, tr.misses = cx, cy, 0
                assigned[i] = best_id
                used.add(best_id)
            else:
                nid = self._next_id[direction]
                self._next_id[direction] += 1
                tracks[nid] = _Track(cx=cx, cy=cy)
                assigned[i] = nid
                used.add(nid)

        # age out tracks that weren't matched this frame
        for tid in list(tracks.keys()):
            if tid not in used:
                tracks[tid].misses += 1
                if tracks[tid].misses > self.max_misses:
                    del tracks[tid]

        return assigned
