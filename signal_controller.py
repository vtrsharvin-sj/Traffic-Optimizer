"""
signal_controller.py
--------------------
Deployment-grade adaptive traffic signal controller for a real 4-way
intersection. ONE approach is green at a time (North / South / East / West);
the other three are RED, with proper GREEN -> YELLOW -> ALL-RED clearance.

Control logic (how a real adaptive controller behaves)
------------------------------------------------------
The controller works in CYCLES. At the start of each cycle it:

1. Measures the current demand (vehicle count) on all four approaches.
2. Builds a SERVICE ORDER for this cycle: busiest approach first, then the
   next busiest, ... down to the least busy. (Empty approaches are skipped;
   if everything is empty it falls back to a fixed N->E->S->W round robin so
   the lights still cycle.)
3. Assigns each approach a STABLE green time for the whole cycle using a
   Webster-style split of a target cycle length proportional to demand,
   clamped to [MIN_GREEN, MAX_GREEN]. Because the split is computed once per
   cycle, the timings don't jitter frame-to-frame.

It then serves the approaches in that order, one at a time, inserting a fixed
YELLOW + ALL-RED between each. When the cycle finishes it recomputes a fresh
order and fresh timings from the latest counts.

Emergency vehicles preempt: the emergency approach is moved to the front and
gets served immediately (cutting the current green to yellow if needed).

All timing constants are realistic and tunable for deployment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DIRECTIONS = ["North", "South", "East", "West"]
ROUND_ROBIN = ["North", "East", "South", "West"]   # default order when idle

# ---- realistic timing constants (seconds) -- tune for your deployment ---- #
MIN_GREEN = 15            # a green never shorter than this
MAX_GREEN = 60            # ... never longer than this
TARGET_CYCLE = 120        # total green budget shared across served approaches
YELLOW = 4                # amber clearance
ALL_RED = 2               # all-red safety clearance
SEC_PER_VEHICLE = 2.5     # demand weighting used inside the Webster split
EMPTY_GREEN = MIN_GREEN   # green given to an approach with no vehicles


@dataclass
class SignalState:
    phase: str = "GREEN"                 # GREEN | YELLOW | ALL_RED
    green_dir: str = "North"
    green_time: int = MIN_GREEN
    time_left: float = 0.0
    lights: Dict[str, str] = field(default_factory=dict)
    reason: str = ""
    order: List[str] = field(default_factory=list)     # this cycle's service order
    plan: Dict[str, int] = field(default_factory=dict)  # dir -> green seconds
    red_left: Dict[str, float] = field(default_factory=dict)   # dir -> s until its green
    full_time: Dict[str, int] = field(default_factory=dict)    # dir -> its green length


class AdaptiveSignalController:
    def __init__(self):
        now = time.time()
        self._order: List[str] = list(ROUND_ROBIN)
        self._plan: Dict[str, int] = {d: MIN_GREEN for d in DIRECTIONS}
        self._idx = 0
        self._green_dir = self._order[0]
        self._phase = "GREEN"
        self._green_time = self._plan[self._green_dir]
        self._phase_end = now + self._green_time
        self._reason = "cycle start"
        self._emergency_pending: Optional[str] = None
        self._planned = False    # plan from real demand on the first update

    # ---- plan a whole cycle from current demand ------------------------ #
    def _plan_cycle(self, counts: Dict[str, int],
                    emergency_dir: Optional[str]) -> None:
        demand = {d: max(0, int(counts.get(d, 0))) for d in DIRECTIONS}
        active = [d for d in DIRECTIONS if demand[d] > 0]

        if not active:
            # nothing detected anywhere -> keep the lights cycling round-robin
            self._order = list(ROUND_ROBIN)
            self._plan = {d: EMPTY_GREEN for d in DIRECTIONS}
            self._reason = "idle round-robin"
        else:
            # BUSIEST FIRST, then descending demand
            self._order = sorted(active, key=lambda d: demand[d], reverse=True)
            total = sum(demand[d] for d in self._order)
            # Webster-style split of the cycle budget, proportional to demand
            plan: Dict[str, int] = {}
            for d in self._order:
                share = demand[d] / total
                g = MIN_GREEN + share * (TARGET_CYCLE - MIN_GREEN * len(self._order))
                # per-vehicle sanity floor
                g = max(g, MIN_GREEN + demand[d] * SEC_PER_VEHICLE * 0.4)
                plan[d] = int(max(MIN_GREEN, min(MAX_GREEN, round(g))))
            self._plan = plan
            self._reason = f"busiest-first (demand { {d: demand[d] for d in self._order} })"

        # emergency approach jumps to the front of the order
        if emergency_dir and emergency_dir in DIRECTIONS:
            if emergency_dir in self._order:
                self._order.remove(emergency_dir)
            self._order.insert(0, emergency_dir)
            self._plan[emergency_dir] = max(self._plan.get(emergency_dir,
                                                            MIN_GREEN), 25)
            self._reason = f"emergency priority: {emergency_dir}"

        self._idx = 0

    def _start_green(self, now: float) -> None:
        self._green_dir = self._order[self._idx]
        self._green_time = int(self._plan.get(self._green_dir, MIN_GREEN))
        self._phase = "GREEN"
        self._phase_end = now + self._green_time

    # ---- main tick ------------------------------------------------------ #
    def update(self, counts: Dict[str, int],
               emergency_dir: Optional[str] = None) -> SignalState:
        now = time.time()

        # First real tick with detections: plan a demand-based cycle so even
        # the very first green goes to the busiest approach.
        if not self._planned and sum(max(0, counts.get(d, 0))
                                     for d in DIRECTIONS) > 0:
            self._plan_cycle(counts, emergency_dir)
            self._start_green(now)
            self._planned = True

        remaining = self._phase_end - now

        # Emergency that appears mid-green on a DIFFERENT approach -> preempt.
        if (emergency_dir and self._phase == "GREEN"
                and emergency_dir != self._green_dir
                and self._emergency_pending != emergency_dir):
            self._emergency_pending = emergency_dir
            self._phase = "YELLOW"
            self._phase_end = now + YELLOW
            remaining = YELLOW
            self._reason = f"emergency preemption -> {emergency_dir}"

        if remaining > 0:
            return self._render(remaining)

        # phase finished -> advance the state machine
        if self._phase == "GREEN":
            self._phase = "YELLOW"
            self._phase_end = now + YELLOW
        elif self._phase == "YELLOW":
            self._phase = "ALL_RED"
            self._phase_end = now + ALL_RED
        else:  # ALL_RED -> move to the next approach in the plan
            if self._emergency_pending:
                # replan immediately with the emergency at the front
                self._plan_cycle(counts, self._emergency_pending)
                self._emergency_pending = None
                self._start_green(now)
            else:
                self._idx += 1
                if self._idx >= len(self._order):
                    # cycle complete -> replan from fresh counts
                    self._plan_cycle(counts, emergency_dir)
                self._start_green(now)

        return self._render(self._phase_end - now)

    def _render(self, remaining: float) -> SignalState:
        lights = {d: "RED" for d in DIRECTIONS}
        if self._phase == "GREEN":
            lights[self._green_dir] = "GREEN"
        elif self._phase == "YELLOW":
            lights[self._green_dir] = "YELLOW"

        remaining = max(0.0, remaining)
        full_time = {d: int(self._plan.get(d, MIN_GREEN)) for d in DIRECTIONS}

        # Estimate seconds until EACH approach next turns green, by walking the
        # service order forward from the current position.
        red_left: Dict[str, float] = {}
        clearance = YELLOW + ALL_RED
        # time still left before the CURRENT green dir finishes its full slot
        if self._phase == "GREEN":
            cur_finish = remaining + clearance
        elif self._phase == "YELLOW":
            cur_finish = remaining + ALL_RED
        else:  # ALL_RED
            cur_finish = remaining

        n = len(self._order)
        for d in DIRECTIONS:
            if lights[d] == "GREEN":
                red_left[d] = 0.0
                continue
            if d not in self._order:
                red_left[d] = 0.0
                continue
            di = self._order.index(d)
            ci = self._idx
            # number of slots between now and d's green
            steps = (di - ci) % n
            if steps == 0:
                # d is the current dir but not green (in its yellow/all-red) ->
                # it comes again next full cycle
                steps = n
            wait = cur_finish
            # add the full slots (green+clearance) of the approaches served
            # before d in the remaining order
            for k in range(1, steps):
                nd = self._order[(ci + k) % n]
                wait += full_time.get(nd, MIN_GREEN) + clearance
            red_left[d] = round(wait, 1)

        return SignalState(
            phase=self._phase, green_dir=self._green_dir,
            green_time=self._green_time, time_left=remaining,
            lights=lights, reason=self._reason,
            order=list(self._order), plan=dict(self._plan),
            red_left=red_left, full_time=full_time)
