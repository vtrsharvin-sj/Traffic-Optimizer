"""
live_app.py
-----------
REAL-TIME dashboard: four traffic videos -> YOLO vehicle detection -> live
counts -> Quantum (QAOA) adaptive signal control + emergency green corridor +
classical-vs-quantum comparison.

Run with:
    streamlit run live_app.py

No simulation, no Arduino, no Twilio. Everything on screen is driven by the
actual vehicles detected in your four videos.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import logger
import alerts
import signal_controller as sc
from video_detector import MultiVideoDetector, DIRECTIONS
from signal_controller import AdaptiveSignalController
from savings_predictor import SavingsPredictor

LOG_INTERVAL = 5.0  # seconds between logged snapshots

st.set_page_config(page_title="Quantum Live Traffic",
                   page_icon="🚦", layout="wide")

REFRESH_MS = 800  # UI refresh cadence


# --------------------------------------------------------------------------- #
# Session state / detector lifecycle
# --------------------------------------------------------------------------- #
if "detector" not in st.session_state:
    st.session_state.detector = None
    st.session_state.video_paths = dict(config.VIDEO_PATHS)
    st.session_state.history = {d: [] for d in DIRECTIONS}
    st.session_state.emergency_dir = "None"


def start_detector(paths):
    if st.session_state.detector is not None:
        st.session_state.detector.stop()
    det = MultiVideoDetector(
        paths, model_path=config.MODEL_PATH, conf=config.CONF_THRESHOLD,
        process_width=config.PROCESS_WIDTH, max_fps=config.MAX_DETECT_FPS,
    )
    det.start()
    st.session_state.detector = det


# --------------------------------------------------------------------------- #
# Sidebar - video sources & controls
# --------------------------------------------------------------------------- #
st.sidebar.title("🎥 Video sources")
st.sidebar.caption("Point each approach at one of your four traffic videos.")

paths = {}
for d in DIRECTIONS:
    paths[d] = st.sidebar.text_input(f"{d} video",
                                     value=st.session_state.video_paths[d])

missing = [d for d, p in paths.items() if not os.path.exists(p)]

col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶ Start", width="stretch"):
    if missing:
        st.sidebar.error("Missing videos: " + ", ".join(missing))
    elif not os.path.exists(config.MODEL_PATH):
        st.sidebar.error(f"YOLO model not found: {config.MODEL_PATH}")
    else:
        st.session_state.video_paths = paths
        start_detector(paths)
        st.sidebar.success("Detection started.")

if col_stop.button("⏹ Stop", width="stretch"):
    if st.session_state.detector:
        st.session_state.detector.stop()
        st.session_state.detector = None
        st.sidebar.info("Detection stopped.")

st.sidebar.subheader("🚑 Emergency vehicle")
st.session_state.emergency_dir = st.sidebar.selectbox(
    "Ambulance approaching from", ["None"] + DIRECTIONS,
    index=(["None"] + DIRECTIONS).index(st.session_state.emergency_dir))


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🚦 Quantum-Enhanced Live Traffic Optimization")
st.caption("Real YOLO vehicle detection on four traffic videos driving a "
           "hybrid quantum-classical adaptive signal controller.")

if st.session_state.detector is not None:
    st.caption(f"⚡ Detection backend: **{st.session_state.detector.get_backend()}**")

if missing and st.session_state.detector is None:
    st.info("Set your four video paths in the sidebar (or drop north/south/"
            "east/west.mp4 into a `videos/` folder) and click **Start**. "
            "You can also test with the same video in all four boxes.")

detector = st.session_state.detector


# --------------------------------------------------------------------------- #
# Live view
# --------------------------------------------------------------------------- #
if detector is not None:
    counts = detector.get_counts()

    # ---- adaptive signal controller (one direction green at a time) --- #
    if "signal_ctrl" not in st.session_state:
        st.session_state.signal_ctrl = AdaptiveSignalController()
    if "savings" not in st.session_state:
        st.session_state.savings = SavingsPredictor()
    ctrl = st.session_state.signal_ctrl

    em_dir = None if st.session_state.emergency_dir == "None" \
        else st.session_state.emergency_dir
    sig = ctrl.update(counts, emergency_dir=em_dir)
    lights = sig.lights                 # dir -> RED / GREEN / YELLOW
    green_dir = sig.green_dir

    # ---- predict fuel + CO2 saved vs fixed timing --------------------- #
    savings = st.session_state.savings.predict(
        counts, sig.plan, sig.order, sc.YELLOW, sc.ALL_RED)

    # ---- heavy-congestion SMS alert (recommend switching signal) ------ #
    if config.CONGESTION_ALERT_ENABLED:
        for d in DIRECTIONS:
            v = counts.get(d, 0)
            # alert if an approach is heavily congested and is NOT the one
            # currently getting the green (i.e. it should be switched to).
            if v >= config.CONGESTION_THRESHOLD and d != green_dir:
                if alerts.send_congestion_sms(d, v, green_dir):
                    logger.log_event("CONGESTION_ALERT", d, vehicles=v,
                                     detail=f"heavy traffic, suggest switching "
                                            f"green from {green_dir} to {d}")

    congested = [d for d in DIRECTIONS
                 if counts.get(d, 0) >= config.CONGESTION_THRESHOLD]
    if congested:
        cg = ", ".join(f"{d} ({counts[d]})" for d in congested)
        st.warning(f"🚦 Heavy traffic on **{cg}** (>{config.CONGESTION_THRESHOLD} "
                   f"vehicles). SMS sent recommending a switch to give it "
                   f"priority.")

    # ---- log to JSON + CSV (throttled) -------------------------------- #
    now_t = time.time()
    densities = {d: detector.get_state(d).density for d in DIRECTIONS}
    last_log = st.session_state.get("_last_log", 0.0)
    last_green = st.session_state.get("_last_green_dir", None)
    if now_t - last_log >= LOG_INTERVAL:
        logger.log_counts(counts, densities)
        logger.log_savings(savings.total_fuel_saved_l,
                           savings.total_co2_saved_kg)
        st.session_state._last_log = now_t
    if last_green != green_dir and sig.phase == "GREEN":
        logger.log_signal_change(green_dir, sig.green_time,
                                 counts.get(green_dir, 0), sig.reason)
        st.session_state._last_green_dir = green_dir
        if em_dir:
            logger.log_emergency(em_dir)

    # ---- top metrics --------------------------------------------------- #
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("North", counts.get("North", 0))
    m2.metric("South", counts.get("South", 0))
    m3.metric("East", counts.get("East", 0))
    m4.metric("West", counts.get("West", 0))
    m5.metric("Total vehicles", sum(counts.values()))

    # ---- signal decision banner --------------------------------------- #
    light_row = "  ".join(
        f"{'🟢' if lights[d]=='GREEN' else '🟡' if lights[d]=='YELLOW' else '🔴'} {d}"
        for d in DIRECTIONS)
    if sig.phase == "GREEN":
        st.success(f"🟢 **{green_dir} GREEN** — {sig.green_time}s "
                   f"({sig.time_left:.0f}s left)   —   {light_row}")
    elif sig.phase == "YELLOW":
        st.warning(f"🟡 **{green_dir} YELLOW** ({sig.time_left:.0f}s) — "
                   f"clearing   —   {light_row}")
    else:
        st.info(f"🔴 ALL-RED ({sig.time_left:.0f}s) — switching   —   {light_row}")

    # this cycle's service order (busiest first) + planned green times
    order_str = "  ➜  ".join(
        f"**{d}** ({sig.plan.get(d, 0)}s)" for d in sig.order)
    st.caption(f"Cycle order (busiest → least): {order_str}   ·   {sig.reason}")

    if em_dir:
        st.error(f"🚑 EMERGENCY: ambulance from **{em_dir}** — served first.")

    # ---- live video grid + accident banner ---------------------------- #
    states = {d: detector.get_state(d) for d in DIRECTIONS}
    accident_dirs = [d for d in DIRECTIONS if states[d].accident]
    if accident_dirs:
        details = ", ".join(f"{d} ({states[d].accident_conf:.0%})"
                            for d in accident_dirs)
        st.error(f"🚨 ACCIDENT DETECTED at: **{details}**. "
                 f"SMS alert sent to {config.TWILIO_TO} "
                 f"(cooldown {config.SMS_COOLDOWN}s).")
        if st.button("✅ Acknowledge / clear accidents"):
            detector.clear_all_accidents()
            st.rerun()

    st.subheader("📹 Live detection feeds")
    row1 = st.columns(2)
    row2 = st.columns(2)
    slots = {"North": row1[0], "South": row1[1],
             "East": row2[0], "West": row2[1]}

    for d in DIRECTIONS:
        state = states[d]
        with slots[d]:
            l = lights[d]
            if l == "GREEN":
                light = f"🟢 GREEN ({sig.time_left:.0f}s left / {sig.full_time.get(d, 0)}s)"
            elif l == "YELLOW":
                light = f"🟡 YELLOW ({sig.time_left:.0f}s left)"
            else:
                light = (f"🔴 RED ({sig.red_left.get(d, 0):.0f}s left / "
                         f"green {sig.full_time.get(d, 0)}s)")
            acc_tag = "  🚨 ACCIDENT" if state.accident else ""
            spd_tag = (f"  ⚡{state.violations} viol" if state.violations
                       else "")
            st.markdown(f"**{d}**  —  {light}  |  {state.smoothed_count} vehicles  "
                        f"|  {state.density}{acc_tag}{spd_tag}")
            frame = state.frame
            if frame is not None:
                # BGR -> RGB for Streamlit
                st.image(frame[:, :, ::-1], width="stretch")
            else:
                st.write("_waiting for frames..._")

    # ---- vehicle summary + speed-violation summary (no images) -------- #
    st.subheader("📋 Vehicle summary")
    total_now = sum(counts.values())
    total_viol = sum(states[d].violations for d in DIRECTIONS)
    fastest = max((states[d].max_speed for d in DIRECTIONS), default=0.0)
    busiest = max(DIRECTIONS, key=lambda d: states[d].smoothed_count)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Vehicles now", total_now)
    s2.metric("Busiest approach", f"{busiest} ({states[busiest].smoothed_count})")
    s3.metric("Speed violations", total_viol)
    s4.metric("Fastest seen", f"{fastest:.0f} km/h")

    # ---- predicted environmental savings ------------------------------ #
    st.subheader("🌱 Predicted savings")

    # totals are small per cycle, so show mL / g and scale up for readability
    fuel_ml = savings.total_fuel_saved_l * 1000.0      # litres -> mL
    co2_g = savings.total_co2_saved_kg * 1000.0        # kg -> g
    fuel_total_str = (f"{savings.total_fuel_saved_l:.2f} L"
                      if fuel_ml >= 1000 else f"{fuel_ml:.0f} mL")
    co2_total_str = (f"{savings.total_co2_saved_kg:.2f} kg"
                     if co2_g >= 1000 else f"{co2_g:.0f} g")

    e1, e2, e3 = st.columns(3)
    e1.metric("⛽ Fuel saved", fuel_total_str,
              f"+{savings.fuel_saved_l * 1000:.0f} mL / cycle")
    e2.metric("🌍 CO₂ saved", co2_total_str,
              f"+{savings.co2_saved_kg * 1000:.0f} g / cycle")
    e3.metric("⏱️ Idle time avoided", f"{savings.idle_saved_s:.0f} veh·s",
              f"{savings.cycles} cycles run")
    st.caption("Predicted from live vehicle counts: idle fuel burn avoided by "
               "giving busy approaches demand-proportional green time instead "
               "of a fixed equal split. Accumulated over the session.")

    summary_rows = []
    for d in DIRECTIONS:
        s = states[d]
        summary_rows.append({
            "Approach": d,
            "Vehicles": s.smoothed_count,
            "Density": s.density,
            "Max speed (km/h)": round(s.max_speed, 1),
            "Speed violations": s.violations,
            "Accident": "🚨 yes" if s.accident else "no",
        })
    st.dataframe(pd.DataFrame(summary_rows), width="stretch",
                 hide_index=True)

    st.caption(
        f"⚡ {total_viol} speed violation(s) over {config.SPEED_LIMIT_KMH} km/h. "
        f"Violation images are saved to the `speed_violations/` folder; "
        f"accident images to `accident_snapshots/`. "
        f"All events are logged in traffic_log.json / traffic_logs.csv.")

    # ---- update history for charts ------------------------------------ #
    for d in DIRECTIONS:
        h = st.session_state.history[d]
        h.append(counts.get(d, 0))
        if len(h) > 60:
            h.pop(0)

    # ---- classical vs quantum comparison ------------------------------ #
    # ---- vehicle count trend ------------------------------------------ #
    st.subheader("📈 Vehicle count over time")
    fig = go.Figure()
    for d in DIRECTIONS:
        fig.add_trace(go.Scatter(y=st.session_state.history[d],
                                 mode="lines", name=d))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    # auto-refresh the live view
    time.sleep(REFRESH_MS / 1000.0)
    st.rerun()
else:
    st.subheader("How it works")
    st.markdown(
        "1. **Detection** — YOLO (`yolo11s.pt`) detects and tracks vehicles "
        "in each of the four videos (cars, motorcycles, buses, trucks).\n"
        "2. **Live counts** — the number of vehicles per approach is measured "
        "in real time.\n"
        "3. **Adaptive signals** — each cycle serves the approaches "
        "busiest-first, one green at a time, with green times proportional to "
        "demand (max-pressure + Webster-style timing).\n"
        "4. **Emergency preemption** — pick an approaching ambulance direction "
        "to give it priority green.\n"
        "5. **Accident detection + speed violations** — logged and saved to "
        "disk; alerts via SMS."
    )
