# Detailed setup and reference

**Explore adaptive traffic signals with a network simulator and a four-feed vehicle detection dashboard.**

This Python project brings together traffic modeling, YOLO detection, emergency routing, and a QUBO optimization experiment. It runs locally with Streamlit; no traffic-signal hardware is required.

| Dashboard | Start command | Control method |
| --- | --- | --- |
| Network simulator | `python -m streamlit run app.py` | QAOA when the optional compatible stack is available; classical simulated annealing otherwise |
| Video detection | `python -m streamlit run live_app.py` | Rule-based adaptive timing driven by detected vehicle counts |

## Quick start

Run these commands from the repository root. Python 3.11 is the recommended environment for the optional legacy quantum dependencies.

```powershell
git clone https://github.com/vtrsharvin-sj/Traffic-Optimizer.git
cd Traffic-Optimizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. Open the local URL printed by Streamlit. Map tiles need an internet connection.

The default installation includes only the simulator. Choose additional features explicitly:

| Feature | Install command |
| --- | --- |
| Vehicle detection | `python -m pip install -r requirements/live.txt` |
| NCNN accident detection | `python -m pip install ncnn` |
| SMS alerts | `python -m pip install -r requirements/alerts.txt` |
| Legacy QAOA experiment | `python -m pip install -r requirements/quantum.txt` |

Use a separate Python 3.11 virtual environment for QAOA. That requirements file constrains the V1 sampler stack used by this code; it does not support Qiskit 2.x. If imports or solving fail, the simulator falls back to classical annealing and reports the backend. A quantum dependency installation failure must be resolved before launch; runtime fallback cannot fix pip errors.

## Network simulator

1. Choose 4–8 intersections, a random seed, and a traffic level.
2. Optimize signals and compare the result with fixed and greedy baselines.
3. Apply congestion, an accident, or a road closure, then optimize again.
4. Select an ambulance start and destination to explore emergency routing.

The network uses synthetic queues on a small grid displayed near Bengaluru. Each intersection has one binary choice: north–south or east–west green. The objective combines unserved traffic, neighboring phase coordination, and an emergency-direction bias.

QAOA runs on a simulator, not a quantum device. Annealing is a classical solver. The project does not demonstrate a quantum speedup. Emergency phase preferences are soft penalties, and a turning route may request conflicting phases at a junction.

## Video detection

```powershell
python -m pip install -r requirements/live.txt
python -m streamlit run live_app.py
```

Place your own videos in `videos/`:

```text
videos/
  north.mp4
  south.mp4
  east.mp4
  west.mp4
  accident.mp4     # optional: replaces the West default for the accident demo
```

You can change every source path in the sidebar. Videos are local assets and are excluded from Git. Click **Start** to load the model and begin processing, and **Stop** to release the detector.

- Vehicle weights: `models/yolo11s.pt`.
- Accident model: `models/accident_ncnn_model/`; install NCNN to enable inference.
- Accident inference is configured for **West only** by default.
- Counts represent vehicles visible in a feed, not measured road throughput.
- The dashboard uses `AdaptiveSignalController`; the single-junction QUBO helper is separate experimental code.
- Speed estimates use image motion and a configurable pixel-to-meter scale. They need camera calibration and are not certified speed measurements.

CPU inference is supported. CUDA is selected when the installed PyTorch build supports the local GPU. TensorRT export is disabled by default; enable it in configuration only with an appropriate local installation. Engines are specific to the GPU/software stack and should be built on the target machine. No particular FPS is guaranteed.

## Configuration and output

Edit `traffic_optimizer/core/config.py` for confidence thresholds, inference size, accident cadence, camera calibration, and alert cooldowns. Paths are anchored to the repository, so launching from another working directory does not relocate models or logs.

Optional SMS configuration:

```powershell
Copy-Item .env.example .env
```

Fill in `TWILIO_SID`, `TWILIO_AUTH`, `TWILIO_FROM`, and `TWILIO_TO`. The alert module also reads `TWILIO_TRIAL_TEMPLATES` and `TWILIO_TEMPLATE`; configure them for your account. Messages are sent only when the app is running and the alert conditions and credentials permit it. Keep real credentials in the ignored `.env` file.

Generated files stay under the ignored `runtime/` directory:

| Location | Contents |
| --- | --- |
| `runtime/logs/traffic_log.json` | Latest 5,000 structured events |
| `runtime/logs/traffic_logs.csv` | Appended event history |
| `runtime/accident_snapshots/` | Accident images |
| `runtime/speed_violations/` | Flagged vehicle crops |

JSON updates use atomic replacement and event memory is bounded. Logging is thread-safe within one Python process; use separate output directories for independent concurrent processes. CSV and snapshots are not automatically rotated.

## Repository layout

```text
Traffic-Optimizer/
  app.py                       # network dashboard launcher
  live_app.py                  # detection dashboard launcher
  traffic_optimizer/
    apps/                      # Streamlit dashboards
    core/                      # configuration, logging, SMS
    detection/                 # models, tracking, video inference
    quantum/                   # QUBO, solvers, baselines, estimates
    traffic/                   # network, events, signals, routing
  models/                      # existing model assets and metadata
  videos/                      # local input clips (not committed)
  requirements/                # base and optional dependency groups
  tests/                       # regression and dashboard smoke tests
  docs/                        # maintenance notes
  .github/workflows/           # automated core checks
  runtime/                     # generated locally, not committed
```

## Development checks

```powershell
python -m unittest discover -s tests -v
python -m compileall -q traffic_optimizer app.py live_app.py
```

Core tests cover emergency phase geometry, optimization, log retention, and dashboard reruns. Live dashboard smoke tests skip when OpenCV is absent. These checks do not send SMS or prove detection accuracy. See [maintenance notes](MAINTENANCE.md) for migration details and scope.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing video | Set an existing local file in each sidebar input; clones do not include clips. |
| Missing model | Check `models/yolo11s.pt` and the NCNN model directory. |
| Accident detection unavailable | Install NCNN and inspect the detector's startup output. |
| QAOA unavailable | Use the isolated legacy requirements, or continue with classical annealing. |
| Slow inference | Check the reported backend; reduce `IMG_SIZE` or frame rate as needed. |
| SMS not received | Check account configuration, recipient, cooldown, and console errors. |

## Scope and license

This is a research/demo application, not a deployed traffic controller. Environmental and savings metrics are model estimates, not field measurements. The original Jetson backup is preserved separately in the local workspace archive.

Project source is covered by the existing [MIT license](../LICENSE). Model weights and third-party dependencies may have separate terms; the project license does not replace those terms. The provenance and training evaluation of the inherited accident weights are not documented here.
