"""
app.py
------
Quantum-Enhanced Adaptive Urban Traffic Optimization  -  Streamlit dashboard.

Run with:
    streamlit run app.py

This dashboard ties together the whole platform:
  * builds a multi-intersection network (4-8 junctions)
  * shows the road network on an OpenStreetMap map (Folium)
  * runs the hybrid quantum (QAOA) optimizer and classical baselines
  * handles dynamic events (congestion / accident / closure / emergency)
  * routes an emergency green corridor
  * reports waiting time, throughput, fuel and CO2, and compares
    classical vs quantum results.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import folium
from streamlit_folium import st_folium

from traffic_network import TrafficNetwork
from quantum_optimizer import (optimize_signals, apply_assignment,
                               QISKIT_AVAILABLE)
from classical_baseline import solve_fixed, solve_rule_based
from environmental import evaluate_plan, improvement
from emergency import (compute_emergency_route, route_travel_time,
                       mark_corridor, clear_corridor)
import events as ev


st.set_page_config(page_title="Quantum Traffic Optimizer",
                   page_icon="🚦", layout="wide")

PHASE_NAME = {0: "North-South", 1: "East-West"}
PHASE_COLOR = {0: "#2ecc71", 1: "#e67e22"}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state(n: int, seed: int, level: float):
    net = TrafficNetwork(n_intersections=n, seed=seed)
    net.randomize_traffic(level=level)
    st.session_state.net = net
    st.session_state.event_log = []
    st.session_state.emergency_path = []
    st.session_state.last_result = None


if "net" not in st.session_state:
    init_state(6, 7, 0.5)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")

st.sidebar.subheader("Network")
n_int = st.sidebar.slider("Number of intersections", 4, 8, 6)
seed = st.sidebar.number_input("Random seed", value=7, step=1)
level = st.sidebar.slider("Traffic level", 0.1, 1.0, 0.5, 0.05)

if st.sidebar.button("🔄 Rebuild network"):
    init_state(n_int, int(seed), level)
    st.rerun()

if st.sidebar.button("🎲 Randomize traffic"):
    st.session_state.net.randomize_traffic(level=level)
    st.session_state.last_result = None
    st.rerun()

net: TrafficNetwork = st.session_state.net
node_ids = list(net.intersections.keys())

st.sidebar.subheader("Quantum engine")
prefer_quantum = st.sidebar.checkbox("Use quantum (QAOA) when available", True)
if QISKIT_AVAILABLE:
    st.sidebar.success("Qiskit detected - QAOA on Aer simulator.")
else:
    st.sidebar.warning("Qiskit not installed - classical annealing on the "
                       "same QUBO is used as the quantum stand-in.")

st.sidebar.subheader("Dynamic event")
event = st.sidebar.selectbox("Trigger an event", ev.EVENT_TYPES)

st.sidebar.subheader("Emergency corridor")
em_start = st.sidebar.selectbox("Ambulance start", node_ids, index=0)
em_dest = st.sidebar.selectbox("Ambulance destination",
                               node_ids, index=len(node_ids) - 1)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🚦 Quantum-Enhanced Adaptive Urban Traffic Optimization")
st.caption("Hybrid Quantum-Classical Traffic Optimization Platform  -  "
           "QUBO / QAOA signal-phase optimization with emergency green corridor.")

col_a, col_b, col_c, col_d = st.columns(4)
summ = net.summary()
col_a.metric("Intersections", int(summ["intersections"]))
col_b.metric("Roads", int(summ["roads"]))
col_c.metric("Total queue (veh)", int(summ["total_queue"]))
col_d.metric("Avg saturation", f"{summ['avg_saturation'] * 100:.0f}%")


# --------------------------------------------------------------------------- #
# Action buttons
# --------------------------------------------------------------------------- #
c1, c2, c3 = st.columns(3)
run_opt = c1.button("🧠 Optimize signals", type="primary", use_container_width=True)
apply_event = c2.button("⚠️ Apply selected event", use_container_width=True)
route_emerg = c3.button("🚑 Route emergency corridor", use_container_width=True)


# ---- event handling -------------------------------------------------------- #
if apply_event:
    if event == "congestion":
        st.session_state.event_log.append(ev.trigger_congestion(net))
    elif event == "accident":
        msg, _ = ev.trigger_accident(net)
        st.session_state.event_log.append(msg)
    elif event == "road_closure":
        msg, _ = ev.trigger_road_closure(net)
        st.session_state.event_log.append(msg)
    elif event == "emergency":
        path = compute_emergency_route(net, em_start, em_dest)
        st.session_state.emergency_path = path
        mark_corridor(net, path)
        st.session_state.event_log.append(
            f"Emergency vehicle dispatched {em_start} -> {em_dest}.")
    else:
        st.session_state.event_log.append("No event selected.")
    st.session_state.last_result = None
    st.rerun()


# ---- emergency routing ----------------------------------------------------- #
if route_emerg:
    path = compute_emergency_route(net, em_start, em_dest)
    st.session_state.emergency_path = path
    mark_corridor(net, path)
    if path:
        st.session_state.event_log.append(
            f"Green corridor set along {' -> '.join(path)}.")
    else:
        st.session_state.event_log.append(
            f"No available route from {em_start} to {em_dest} (closures?).")
    st.session_state.last_result = None
    st.rerun()


# ---- optimization ---------------------------------------------------------- #
if run_opt:
    em_path = st.session_state.emergency_path or None

    # Baselines evaluated on current state
    fixed_res = solve_fixed(net)
    fixed_metrics = evaluate_plan(net, fixed_res)

    rule_res = solve_rule_based(net)
    rule_metrics = evaluate_plan(net, rule_res)

    # Quantum / hybrid optimization
    q_res = optimize_signals(net, emergency_path=em_path,
                             prefer_quantum=prefer_quantum)
    apply_assignment(net, q_res)
    q_metrics = evaluate_plan(net, q_res)

    st.session_state.last_result = {
        "fixed": (fixed_res, fixed_metrics),
        "rule": (rule_res, rule_metrics),
        "quantum": (q_res, q_metrics),
    }


# --------------------------------------------------------------------------- #
# Map + details
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.subheader("🗺️ Road network")
    center = [net.BASE_LAT + 0.004, net.BASE_LON + 0.005]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    emergency_path = st.session_state.emergency_path

    # roads
    for (u, v), road in net.roads.items():
        a = net.intersections[u]
        b = net.intersections[v]
        if road.closed:
            color, weight, dash = "#7f8c8d", 3, "8"
        else:
            sat = min(road.saturation, 1.0)
            # green -> red by saturation
            r = int(255 * sat)
            g = int(200 * (1 - sat))
            color = f"#{r:02x}{g:02x}30"
            weight = 3 + sat * 6
            dash = None
        folium.PolyLine(
            [(a.lat, a.lon), (b.lat, b.lon)],
            color=color, weight=weight, opacity=0.8, dash_array=dash,
            tooltip=(f"{u}->{v} | dens {road.density}/{road.capacity} "
                     f"| queue {road.queue}"
                     + (" | CLOSED" if road.closed else "")),
        ).add_to(fmap)

    # emergency corridor overlay
    if emergency_path and len(emergency_path) > 1:
        coords = [(net.intersections[n].lat, net.intersections[n].lon)
                  for n in emergency_path]
        folium.PolyLine(coords, color="#2980ff", weight=6, opacity=0.9,
                        tooltip="Emergency green corridor").add_to(fmap)

    # intersections
    for node_id, node in net.intersections.items():
        ns, ew = net.phase_load(node_id)
        phase_col = PHASE_COLOR[node.green_phase]
        icon_col = "blue" if node.is_emergency else "gray"
        popup = (f"<b>{node.name}</b><br>"
                 f"Green: {PHASE_NAME[node.green_phase]}<br>"
                 f"Green time: {node.green_time}s<br>"
                 f"NS load: {ns:.0f} | EW load: {ew:.0f}")
        folium.CircleMarker(
            [node.lat, node.lon], radius=11, color=phase_col,
            fill=True, fill_color=phase_col, fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=250),
            tooltip=f"{node_id} ({PHASE_NAME[node.green_phase]} green)",
        ).add_to(fmap)
        folium.map.Marker(
            [node.lat, node.lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;color:#111;'
                     f'font-weight:bold;">{node_id}</div>'),
        ).add_to(fmap)

    st_folium(fmap, height=460, use_container_width=True)

    st.caption("Node color = current green phase (green NS / orange EW). "
               "Road color = saturation (green low, red high). "
               "Blue line = emergency corridor.")

with right:
    st.subheader("🚑 Emergency corridor")
    if emergency_path:
        tt = route_travel_time(net, emergency_path)
        st.write(f"**Route:** {' → '.join(emergency_path)}")
        st.metric("Estimated travel time", f"{tt:.0f} s")
        if st.button("Clear corridor"):
            clear_corridor(net)
            st.session_state.emergency_path = []
            st.rerun()
    else:
        st.info("No active corridor. Pick start/destination and click "
                "'Route emergency corridor'.")

    st.subheader("📋 Event log")
    if st.session_state.event_log:
        for line in reversed(st.session_state.event_log[-8:]):
            st.write("• " + line)
        if st.button("Clear events / reopen roads"):
            ev.reopen_all(net)
            st.session_state.event_log = []
            st.rerun()
    else:
        st.write("No events yet.")


# --------------------------------------------------------------------------- #
# Signal table + optimization results
# --------------------------------------------------------------------------- #
st.subheader("🚦 Signal state per intersection")
rows = []
for node_id, node in net.intersections.items():
    ns, ew = net.phase_load(node_id)
    rows.append({
        "Intersection": node_id,
        "Green phase": PHASE_NAME[node.green_phase],
        "Green time (s)": node.green_time,
        "NS load": round(ns, 1),
        "EW load": round(ew, 1),
        "Corridor": "🚑" if node.is_emergency else "",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


result = st.session_state.last_result
if result:
    st.header("📊 Classical vs Quantum comparison")

    fixed_res, fixed_m = result["fixed"]
    rule_res, rule_m = result["rule"]
    q_res, q_m = result["quantum"]

    st.info(f"**Quantum method used:** {q_res.method}. {q_res.detail}")

    # metric comparison table
    comp = pd.DataFrame({
        "Metric": ["Avg waiting time (s)", "Total waiting (s)",
                   "Throughput (veh/cycle)", "Fuel (L)", "CO₂ (kg)",
                   "QUBO cost (energy)"],
        "Fixed timing": [fixed_m.avg_wait_s, fixed_m.total_wait_s,
                         fixed_m.throughput, fixed_m.fuel_litres,
                         fixed_m.co2_kg, round(fixed_res.energy, 1)],
        "Rule-based": [rule_m.avg_wait_s, rule_m.total_wait_s,
                       rule_m.throughput, rule_m.fuel_litres,
                       rule_m.co2_kg, round(rule_res.energy, 1)],
        "Quantum (QAOA)": [q_m.avg_wait_s, q_m.total_wait_s,
                           q_m.throughput, q_m.fuel_litres,
                           q_m.co2_kg, round(q_res.energy, 1)],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # improvement over fixed baseline
    imp = improvement(fixed_m, q_m)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Waiting time", f"{q_m.avg_wait_s:.0f} s",
              f"{imp['waiting_time_pct']:+.0f}% vs fixed")
    i2.metric("Throughput", f"{q_m.throughput:.0f}",
              f"{imp['throughput_pct']:+.0f}% vs fixed")
    i3.metric("Fuel", f"{q_m.fuel_litres:.2f} L",
              f"{imp['fuel_pct']:+.0f}% vs fixed")
    i4.metric("CO₂", f"{q_m.co2_kg:.2f} kg",
              f"{imp['co2_pct']:+.0f}% vs fixed")

    # bar chart
    metrics_names = ["Avg wait (s)", "Fuel (L)", "CO₂ (kg)"]
    fig = go.Figure()
    fig.add_bar(name="Fixed", x=metrics_names,
                y=[fixed_m.avg_wait_s, fixed_m.fuel_litres, fixed_m.co2_kg])
    fig.add_bar(name="Rule-based", x=metrics_names,
                y=[rule_m.avg_wait_s, rule_m.fuel_litres, rule_m.co2_kg])
    fig.add_bar(name="Quantum", x=metrics_names,
                y=[q_m.avg_wait_s, q_m.fuel_litres, q_m.co2_kg])
    fig.update_layout(barmode="group", height=380,
                      title="Lower is better (waiting, fuel, CO₂)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔬 Where is the quantum component?"):
        st.markdown(
            "- Each intersection contributes **one binary variable** "
            "`x_i` (0 = North-South green, 1 = East-West green).\n"
            "- The total unserved queue + coordination + emergency terms are "
            "encoded as a **QUBO / Ising Hamiltonian** in "
            "`quantum_optimizer.build_qubo`.\n"
            "- **QAOA** (Qiskit Aer simulator) minimises that Hamiltonian in "
            "`solve_qaoa`. If Qiskit is unavailable, the *same* QUBO is solved "
            "by classical simulated annealing so results stay comparable.\n"
            "- Classical baselines (fixed timing, rule-based greedy) are scored "
            "on the identical cost function for a fair comparison."
        )
else:
    st.info("Click **Optimize signals** to run the quantum optimizer and "
            "compare it against the classical baselines.")
