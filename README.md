# Quantum-Enhanced Adaptive Urban Traffic Optimization

A **hybrid quantum-classical traffic optimization platform** that dynamically
manages traffic signals. Built for **Windows laptops** with a **Streamlit**
dashboard — no Arduino / Jetson / UNO hardware required.

This is a software-only build of the traffic system based on the
*Quantum-Enhanced Adaptive Urban Traffic Optimization* problem statement.

There are **two apps**:

| App | Command | What it does |
|-----|---------|--------------|
| **Network simulator** | `streamlit run app.py` | Multi-intersection map view with the QUBO/QAOA engine on a synthetic NetworkX network (4–8 intersection scenario, emergency corridor map, classical-vs-quantum comparison). |
| **Live detection** | `streamlit run live_app.py` | Runs **real YOLO vehicle detection** on four traffic videos (North/South/East/West), counts vehicles live, and drives the adaptive signal controller from those real counts. |

## Features

- **Multi-intersection network** (`traffic_network.py`) — models 4–8 connected
  intersections (NetworkX) with per-road density, queue length, capacity and
  signal status.
- **Quantum optimization engine** (`quantum_optimizer.py`) — encodes signal-phase
  selection as a **QUBO / Ising** problem and solves it with **QAOA** on the
  Qiskit Aer simulator. If the QAOA stack isn't available, the *same* QUBO is
  solved with classical **simulated annealing** so the app still runs (see
  [Quantum / QAOA note](#quantum--qaoa-note)).
- **Adaptive signal controller** (`signal_controller.py`,
  `AdaptiveSignalController`) — green-light duration adapts to live load instead
  of a fixed timer.
- **Emergency green corridor** (`emergency.py`) — routes an ambulance on the
  fastest path and forces green along the corridor, then restores normal control.
- **Dynamic events** (`events.py`) — sudden congestion, accident, road closure,
  emergency vehicle arrival.
- **Environmental analysis** (`environmental.py`) — waiting time, throughput,
  fuel, CO₂.
- **Classical vs quantum comparison** (`classical_baseline.py`) — fixed-timing
  and rule-based greedy baselines scored on the same cost function.
- **Live YOLO detection** (`video_detector.py`, `MultiVideoDetector`) — detects
  and tracks cars, motorcycles, buses and trucks across the four feeds using
  Ultralytics (`yolo11s.pt`).
- **Accident detection + optional SMS** (`accident_ncnn_model/`, `alerts.py`) —
  the NCNN accident model flags frames; on a hit a snapshot is saved and an
  optional **Twilio SMS** is sent (with a per-direction cooldown).
- **Speed violations** (`speed_tracker.py`, `simple_tracker.py`) — estimates
  per-vehicle speed from tracked box movement and flags vehicles over the limit.
- **Savings predictor** (`savings_predictor.py`) — estimates time/fuel savings
  from the adaptive control.
- **Logging** (`logger.py`) — writes structured JSON (`traffic_log.json`) and a
  flat CSV (`traffic_logs.csv`) of counts, signal changes, accidents and
  emergencies.

## Network simulator (`app.py`)

An interactive OpenStreetMap/Folium dashboard over a synthetic network.

### How to use

1. Set the number of intersections, seed and traffic level in the sidebar.
2. Click **Optimize signals** to run the optimizer and see the
   classical-vs-quantum comparison.
3. Pick an **event** (congestion / accident / road closure / emergency) and
   click **Apply selected event**, then re-optimize to watch signals adapt.
4. Choose an ambulance **start** and **destination**, click **Route emergency
   corridor**, then **Optimize signals** to open the green corridor.

## Live detection mode (`live_app.py`)

`live_app.py` maps four videos to the four approaches of one real junction:

```
        North
          |
  West -- + -- East
          |
        South
```

- YOLO (`yolo11s.pt`) detects + tracks vehicles in each video
  (`video_detector.py`).
- The four live counts become the North/South/East/West loads that drive the
  adaptive signal controller.
- An **emergency vehicle** direction adds a strong bias so that corridor stays
  green.
- Fixed-timing and rule-based controllers are shown side-by-side, live.

### Point it at your videos

By default it looks for `car video 0..3.mp4` in your Downloads folder, then
falls back to `videos/north.mp4`, `videos/south.mp4`, etc. You can also type any
path into the sidebar boxes. Testing tip: put the same video in all four boxes
to see it working.

### GPU acceleration (make detection fast)

Detection runs on the **NVIDIA GPU** automatically when a CUDA build of PyTorch
is installed. On an RTX 5080 this takes YOLO from ~19 FPS (CPU) to **100+ FPS**.

Install the CUDA PyTorch build (RTX 50-series / Blackwell needs CUDA 12.8):

```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

If that download is slow or keeps timing out, grab the wheels with curl (more
reliable resume) and install locally:

```powershell
curl.exe -L --retry 100 --retry-all-errors -C - -o torch.whl "https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl"
curl.exe -L --retry 100 --retry-all-errors -C - -o torchvision.whl "https://download.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp313-cp313-win_amd64.whl"
pip install --force-reinstall --no-deps torch.whl torchvision.whl
```

Check it worked:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The dashboard shows the active backend (e.g. "PyTorch on CUDA (FP16) — NVIDIA
GeForce RTX 5080") at the top.

**TensorRT (optional, even faster):** set `USE_TENSORRT = True` in `config.py`
and `pip install tensorrt`. The app auto-exports a local `yolo11s.engine` from
`yolo11s.pt` on first run. PyTorch-CUDA is already fast enough for four feeds,
so this is off by default.

### Accuracy

Frames are **letterboxed** (aspect-preserving resize + grey pad) before
inference and inference runs at `IMG_SIZE = 960`. Tune `CONF_THRESHOLD` /
`IOU_THRESHOLD` / `IMG_SIZE` in `config.py`.

### Accident detection + SMS alerts

- The trained **accident model** (`accident_ncnn_model/`, NCNN, single
  "Accident" class) runs on the four feeds. If an accident is detected, that
  approach is flagged, a red banner + border is drawn, a snapshot is saved to
  `accident_snapshots/`, and an optional **Twilio SMS** is sent (with a
  per-direction cooldown).
- Requires `pip install ncnn` and `pip install twilio`.
- Configure SMS in `config.py` or via environment variables (see
  [Secrets](#secrets)). If Twilio credentials are missing or inactive, accidents
  are still detected and logged; the SMS is simply skipped.
- Tune sensitivity with `ACCIDENT_CONF` and cadence with `ACCIDENT_EVERY_N`.

### Speed violations

Vehicles are tracked and their speed is estimated from how far the box centre
moves between frames (`speed_tracker.py`, `simple_tracker.py`). Vehicles over
`SPEED_LIMIT_KMH` get a cropped snapshot in `speed_violations/`, are logged as
`SPEED_VIOLATION` events, and appear in the dashboard gallery. Tune in
`config.py`: `SPEED_LIMIT_KMH`, `PIXELS_PER_METER` (camera scale — the main knob
for absolute accuracy), `SPEED_MAX_PLAUSIBLE_KMH`, and `SPEED_MIN_OBS`. Speed
from monocular video is approximate; calibrate `PIXELS_PER_METER` per camera.

### Logging (JSON + CSV)

Every run logs to two files in the project folder (`logger.py`):

- **`traffic_log.json`** — a list of structured events:
  `{ "timestamp", "type", "direction", "data": {...} }`
- **`traffic_logs.csv`** — one row per event:
  `Timestamp, Event_Type, Direction, Vehicles, Density, Detail`

Event types include `COUNTS`, `SIGNAL_CHANGE`, `ACCIDENT`, `SPEED_VIOLATION`,
and `EMERGENCY`.

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Streamlit network simulator (entry point) |
| `live_app.py` | Streamlit live YOLO detection dashboard (entry point) |
| `traffic_network.py` | NetworkX intersection/road model |
| `quantum_optimizer.py` | QUBO builder + QAOA + annealing fallback |
| `classical_baseline.py` | Fixed-timing and rule-based baselines |
| `emergency.py` | Emergency green corridor routing |
| `events.py` | Dynamic event triggers |
| `environmental.py` | Waiting time / fuel / CO₂ metrics |
| `config.py` | Live-mode configuration (video paths, thresholds, SMS, speed) |
| `logger.py` | JSON + CSV event logging |
| `alerts.py` | Twilio SMS alerts for accidents |
| `model_loader.py` | Loads the YOLO detection model (CPU/CUDA/TensorRT) |
| `video_detector.py` | `MultiVideoDetector` — per-feed YOLO detection + tracking |
| `signal_controller.py` | `AdaptiveSignalController` — adaptive green timing |
| `simple_tracker.py` | Lightweight centroid tracker |
| `speed_tracker.py` | Per-vehicle speed estimation + violation flagging |
| `savings_predictor.py` | Time/fuel savings estimate from adaptive control |
| `live_intersection.py` | Single-junction QUBO helper for live mode |
| `accident_ncnn_model/` | NCNN accident-detection model weights + loader |
| `requirements.txt` | Python dependencies |

## Setup (Windows / PowerShell)

```powershell
cd "quantum_traffic_optimizer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The heavy / optional dependencies (PyTorch + Ultralytics, OpenCV, NCNN, Twilio,
Qiskit) are grouped and labelled in `requirements.txt`. The core simulator
(`app.py`) runs without them; live detection and the quantum QAOA path need the
corresponding optional groups installed.

## Run

```powershell
streamlit run app.py        # network simulator
streamlit run live_app.py   # live YOLO detection
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Quantum / QAOA note

Each intersection is one binary variable `x_i` (0 = North-South green,
1 = East-West green). The total unserved queue plus coordination and emergency
terms form a QUBO / Ising Hamiltonian (`quantum_optimizer.build_qubo`), which
**QAOA** minimises (`quantum_optimizer.solve_qaoa`).

The QAOA path requires the **qiskit 1.x stack**: `qiskit`, `qiskit-aer`,
`qiskit-optimization`, and `qiskit-algorithms`. On **qiskit 2.x** the QAOA path
is unavailable because `qiskit.primitives.Sampler` was removed, so the app
transparently falls back to classical **simulated annealing** on the identical
QUBO. The classical baselines are scored on the same cost function, so the
comparison stays fair either way.

## Secrets

Twilio credentials and any other secrets live in a `.env` file, which is
**gitignored and never committed**.

1. Copy the template: `copy .env.example .env`
2. Fill in your Twilio credentials in `.env`:

   ```
   TWILIO_SID=ACxxxxxxxx
   TWILIO_AUTH=your_auth_token
   TWILIO_FROM=+1xxxxxxxxxx
   TWILIO_TO=+91xxxxxxxxxx
   ```

3. Never commit `.env`. Commit `.env.example` (placeholders only).

`config.py` reads these via `python-dotenv`.
