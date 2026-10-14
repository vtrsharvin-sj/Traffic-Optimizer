"""
logger.py
---------
Logs traffic events to BOTH a JSON file and a CSV file, matching the style of
the original Jetson project.

JSON  (traffic_log.json):  a growing list of event objects
    { "timestamp": ISO8601, "type": ..., "direction": ..., "data": {...} }

CSV   (traffic_logs.csv):  one row per event
    Timestamp, Event_Type, Direction, Vehicles, Density, Detail

Event types used by this app:
    COUNTS         - periodic per-direction vehicle counts snapshot
    SIGNAL_CHANGE  - quantum controller changed the green phase
    ACCIDENT       - accident detected on an approach
    EMERGENCY      - emergency corridor activated

Thread-safe. Writes are appended to CSV immediately; the JSON file is rewritten
(kept to the last N events) so it stays valid JSON at all times.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List

import config

_lock = threading.Lock()
_events: List[Dict[str, Any]] = []

JSON_PATH = os.path.join(config.BASE_DIR, "traffic_log.json")
CSV_PATH = os.path.join(config.BASE_DIR, "traffic_logs.csv")
_MAX_JSON = 5000


def _ensure_csv_header() -> None:
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["Timestamp", "Event_Type", "Direction",
                 "Vehicles", "Density", "Detail"])


def _load_existing_json() -> None:
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                _events.extend(data[-_MAX_JSON:])
        except Exception:
            pass
    else:
        # create an empty but valid JSON file so it always exists
        try:
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception:
            pass


_ensure_csv_header()
_load_existing_json()


def log_event(event_type: str, direction: str,
              vehicles: int = 0, density: str = "",
              detail: str = "", extra: Dict[str, Any] | None = None) -> None:
    """Append one event to both JSON and CSV."""
    ts = datetime.now().isoformat()
    data: Dict[str, Any] = {"vehicles": vehicles, "density": density}
    if detail:
        data["detail"] = detail
    if extra:
        data.update(extra)

    event = {"timestamp": ts, "type": event_type,
             "direction": direction, "data": data}

    with _lock:
        _events.append(event)
        # rewrite JSON (trimmed) so it is always valid
        try:
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(_events[-_MAX_JSON:], f, indent=2)
        except Exception as exc:
            print(f"[logger] JSON write failed: {exc}")
        # append CSV row
        try:
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [ts, event_type, direction, vehicles, density, detail])
        except Exception as exc:
            print(f"[logger] CSV write failed: {exc}")


def log_counts(counts: Dict[str, int], densities: Dict[str, str]) -> None:
    """Log a snapshot of all four approach counts as one event per direction."""
    for d, c in counts.items():
        log_event("COUNTS", d, vehicles=c, density=densities.get(d, ""))


def log_signal_change(direction_pair: str, green_time: int,
                      served: int, method: str) -> None:
    log_event("SIGNAL_CHANGE", direction_pair, vehicles=served,
              detail=f"green {green_time}s via {method}")


def log_accident(direction: str, conf: float, snapshot: str = "") -> None:
    log_event("ACCIDENT", direction, vehicles=0,
              detail=f"conf={conf:.2f} snapshot={os.path.basename(snapshot)}",
              extra={"confidence": round(conf, 3), "snapshot": snapshot})


def log_emergency(direction: str) -> None:
    log_event("EMERGENCY", direction, detail="green corridor activated")


def log_savings(fuel_l: float, co2_kg: float) -> None:
    log_event("SAVINGS", "ALL", detail=f"fuel_saved={fuel_l:.3f}L "
              f"co2_saved={co2_kg:.3f}kg",
              extra={"fuel_saved_l": round(fuel_l, 4),
                     "co2_saved_kg": round(co2_kg, 4)})


def log_violation(direction: str, vehicle_id: int, speed_kmh: float,
                  snapshot: str = "") -> None:
    log_event("SPEED_VIOLATION", direction, vehicles=1,
              detail=f"vehicle #{vehicle_id} {speed_kmh:.0f} km/h "
                     f"snapshot={os.path.basename(snapshot)}",
              extra={"vehicle_id": vehicle_id,
                     "speed_kmh": round(speed_kmh, 1),
                     "snapshot": snapshot})


def paths() -> Dict[str, str]:
    return {"json": JSON_PATH, "csv": CSV_PATH}
