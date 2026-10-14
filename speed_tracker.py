"""
speed_tracker.py
----------------
Estimates per-vehicle speed from ByteTrack tracker IDs and flags speeding
vehicles, mirroring the approach in the original Jetson project:

  * track each vehicle's box centre over time (keyed by tracker id)
  * distance moved in pixels / pixels-per-metre -> metres
  * metres / dt -> m/s -> km/h  (with an exponential moving average)
  * if speed > SPEED_LIMIT_KMH, it's a violation (with a per-vehicle cooldown)

One tracker is kept per approach/direction. Boxes are supplied in the ORIGINAL
frame's pixel coordinates so the speed reflects real motion, not the letterboxed
inference frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config


@dataclass
class _VState:
    cx: float
    cy: float
    t: float
    speed: float = 0.0
    obs: int = 0
    last_violation: float = 0.0


@dataclass
class Violation:
    direction: str
    vehicle_id: int
    speed_kmh: float
    box: Tuple[int, int, int, int]   # x1,y1,x2,y2 in original-frame pixels


class SpeedTracker:
    """Per-direction speed estimator. Call update() once per processed frame."""

    def __init__(self):
        self._state: Dict[str, Dict[int, _VState]] = {}

    def update(self, direction: str,
               dets: List[Tuple[int, Tuple[float, float, float, float]]]
               ) -> Tuple[Dict[int, float], List[Violation]]:
        """
        dets: list of (tracker_id, (x1,y1,x2,y2)) in ORIGINAL-frame pixels.
        Returns (speeds_by_id, new_violations).
        """
        now = time.time()
        lane = self._state.setdefault(direction, {})
        speeds: Dict[int, float] = {}
        violations: List[Violation] = []
        seen = set()

        for tid, box in dets:
            if tid is None:
                continue
            seen.add(tid)
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            prev = lane.get(tid)
            if prev is None:
                lane[tid] = _VState(cx=cx, cy=cy, t=now)
                speeds[tid] = 0.0
                continue

            dt = now - prev.t
            if dt < 0.05:                     # too soon; keep last speed
                speeds[tid] = prev.speed
                continue

            dist_px = math.hypot(cx - prev.cx, cy - prev.cy)
            if dist_px < 2.0:                 # jitter -> stationary
                inst_kmh = 0.0
            else:
                metres = dist_px / config.PIXELS_PER_METER
                inst_kmh = (metres / dt) * 3.6

            # Reject teleport jumps: ByteTrack occasionally re-assigns an ID to
            # a different vehicle, producing an impossible position jump. Ignore
            # any instantaneous reading above a physical sanity cap.
            if inst_kmh > config.SPEED_MAX_PLAUSIBLE_KMH:
                prev.cx, prev.cy, prev.t = cx, cy, now   # resync, keep speed
                speeds[tid] = prev.speed
                continue

            prev.obs += 1
            # exponential moving average for stability
            a = config.SPEED_SMOOTHING
            speed = prev.speed * a + inst_kmh * (1 - a)

            prev.cx, prev.cy, prev.t, prev.speed = cx, cy, now, speed
            # only trust the reading after a few consistent observations
            speeds[tid] = speed if prev.obs >= config.SPEED_MIN_OBS else 0.0

            # violation check with per-vehicle cooldown (only trusted speeds)
            if (prev.obs >= config.SPEED_MIN_OBS
                    and speed > config.SPEED_LIMIT_KMH
                    and speed <= config.SPEED_MAX_PLAUSIBLE_KMH
                    and now - prev.last_violation > config.VIOLATION_COOLDOWN):
                prev.last_violation = now
                violations.append(Violation(
                    direction=direction, vehicle_id=int(tid),
                    speed_kmh=speed,
                    box=(int(box[0]), int(box[1]), int(box[2]), int(box[3]))))

        # forget vehicles that disappeared (keep the dict small)
        for tid in list(lane.keys()):
            if tid not in seen and now - lane[tid].t > 5.0:
                del lane[tid]

        return speeds, violations
