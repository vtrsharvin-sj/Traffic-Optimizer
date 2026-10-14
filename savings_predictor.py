"""
savings_predictor.py
--------------------
Predicts FUEL SAVED and CO2 SAVED by the adaptive signal controller compared to
a dumb FIXED-TIMING controller, using the live detection counts and the
controller's current cycle plan.

Prediction model
-----------------
The dominant, controllable source of fuel burn and emissions at a junction is
vehicles IDLING while stopped at a red light. So we predict the total idle
(waiting) vehicle-seconds for two strategies each cycle and take the difference.

For a cycle of length C seconds with a per-approach green split g_d:

    red_time_d      = C - g_d
    idle_veh_sec_d  = vehicles_d * red_time_d          (veh waiting * time)
    total_idle      = sum over approaches d

  * FIXED baseline: every approach gets the SAME green  g = C / 4,
    regardless of how many vehicles are actually waiting. Busy approaches
    therefore sit on red far longer than they should.

  * ADAPTIVE: greens are proportional to demand (the controller's plan), so
    busy approaches spend less of the cycle on red.

Idle seconds saved = total_idle_fixed - total_idle_adaptive  (>= 0)

Convert:
    fuel_saved_litres = idle_seconds_saved * FUEL_IDLE_L_PER_S
    co2_saved_kg      = fuel_saved_litres  * CO2_KG_PER_L

Savings are accumulated across the session so the dashboard shows a growing
running total (predicted litres / kg avoided so far).

Physical constants are typical published figures for a petrol car idling; they
give realistic-order estimates, not lab-grade values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Idling burns ~0.0002-0.0004 L/s for a typical petrol car; use a mid value.
FUEL_IDLE_L_PER_S = 0.00028
# Burning 1 L of petrol releases ~2.31 kg of CO2.
CO2_KG_PER_L = 2.31

DIRECTIONS = ["North", "South", "East", "West"]


@dataclass
class SavingsState:
    # per-cycle (instantaneous prediction)
    idle_saved_s: float = 0.0
    fuel_saved_l: float = 0.0
    co2_saved_kg: float = 0.0
    # cumulative over the session
    total_fuel_saved_l: float = 0.0
    total_co2_saved_kg: float = 0.0
    cycles: int = 0


class SavingsPredictor:
    def __init__(self):
        self._state = SavingsState()
        self._last_cycle_key = None

    def _cycle_length(self, plan: Dict[str, int], yellow: int, all_red: int,
                      order: List[str]) -> float:
        greens = sum(plan.get(d, 0) for d in order)
        clearances = (yellow + all_red) * max(1, len(order))
        return max(1.0, greens + clearances)

    def predict(self, counts: Dict[str, int], plan: Dict[str, int],
                order: List[str], yellow: int, all_red: int) -> SavingsState:
        """
        Predict fuel/CO2 saved for the CURRENT cycle plan vs a fixed-timing
        baseline. Accumulates a running total once per new cycle plan.
        """
        cycle = self._cycle_length(plan, yellow, all_red, order)
        clearance = (yellow + all_red)

        # We estimate the total TIME VEHICLES SPEND QUEUED over one cycle for
        # each strategy. A vehicle keeps idling until its approach's green has
        # discharged it. If an approach gets `g` green seconds it can clear
        # about `g * DISCHARGE_RATE` vehicles; any that don't clear wait the
        # whole next cycle too.
        #
        # ADAPTIVE gives busy approaches MORE green (from the plan), so they
        # clear their queue and stop idling sooner.
        # FIXED gives every approach the SAME green regardless of its queue, so
        # busy approaches leave many vehicles idling into the next cycle.
        DISCHARGE = 0.5  # vehicles cleared per green-second (one lane)

        def idle_seconds(green_of) -> float:
            total = 0.0
            for d in DIRECTIONS:
                veh = max(0, counts.get(d, 0))
                if veh == 0:
                    continue
                g = max(1.0, green_of(d))
                cleared = min(veh, g * DISCHARGE)
                leftover = max(0.0, veh - cleared)
                # cleared vehicles idle on average half the red before green;
                # leftover vehicles idle a whole extra cycle.
                red_before = max(0.0, cycle - g)
                total += cleared * (red_before / 2.0)
                total += leftover * cycle
            return total

        # ADAPTIVE: demand-proportional greens (the controller plan)
        idle_adaptive = idle_seconds(lambda d: plan.get(d, cycle / 4.0))
        # FIXED baseline: equal green share for all four approaches
        equal_green = (cycle - clearance * 4) / 4.0
        idle_fixed = idle_seconds(lambda d: equal_green)

        idle_saved = max(0.0, idle_fixed - idle_adaptive)
        fuel_saved = idle_saved * FUEL_IDLE_L_PER_S
        co2_saved = fuel_saved * CO2_KG_PER_L

        self._state.idle_saved_s = round(idle_saved, 1)
        self._state.fuel_saved_l = round(fuel_saved, 4)
        self._state.co2_saved_kg = round(co2_saved, 4)

        # accumulate once per distinct cycle plan (avoid double counting on the
        # sub-second UI refresh)
        key = (tuple(order), tuple(sorted(plan.items())),
               tuple(sorted(counts.items())))
        if key != self._last_cycle_key:
            self._last_cycle_key = key
            self._state.total_fuel_saved_l = round(
                self._state.total_fuel_saved_l + fuel_saved, 4)
            self._state.total_co2_saved_kg = round(
                self._state.total_co2_saved_kg + co2_saved, 4)
            self._state.cycles += 1

        return self._state
