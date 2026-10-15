# 🚦 Traffic Optimizer

**Watch traffic. Count vehicles. Explore smarter signal timing.**

Traffic Optimizer lets you simulate a road network or use four traffic videos to detect vehicles and adjust signal timing. It runs on your computer and opens in your browser. No Arduino, Jetson, or physical traffic lights are needed.

[Get started](#get-started) · [Use your videos](#use-your-own-videos) · [Optional features](#optional-features) · [Detailed guide](docs/SETUP.md)

## What can it do?

- **Count vehicles** — detect cars, motorcycles, buses, and trucks.
- **Adjust signals** — give busier approaches more green time.
- **Explore emergency priority** — test ambulance routes and priority signals.
- **Flag possible accidents** — use the included accident model.
- **Save results** — record events and save accident or speed-alert images.
- **Compare signal plans** — experiment with different optimization methods.

## Choose your dashboard

| Dashboard | What you see | What you need |
| --- | --- | --- |
| **Network simulator** | A map with 4–8 intersections, simulated traffic, and signal comparisons | Basic installation |
| **Video dashboard** | Four traffic feeds, vehicle counts, and adaptive signals | Video dependencies and your own clips |

**Start with the network simulator.** You do not need videos or a GPU for it.

## Get started

### 1. Download the project

Install Python and Git first. Python 3.11 is a good starting point, especially for the optional quantum feature.

Open **PowerShell** and run:

```powershell
git clone https://github.com/vtrsharvin-sj/Traffic-Optimizer.git
cd Traffic-Optimizer
```

Already downloaded the project? Open a terminal inside the folder containing `app.py` and skip this step.

### 2. Install the basics

Create a separate Python environment for this project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, replace the activation command with `source .venv/bin/activate`.

### 3. Start the simulator

```powershell
python -m streamlit run app.py
```

Open the local link printed in the terminal, usually `http://localhost:8501`.

Choose a traffic level and click **Optimize signals**. Then try congestion, a road closure, or an emergency route and compare the results. The map needs internet access to load its tiles.

To stop the app, press **Ctrl+C** in the terminal.

## Use your own videos

Use the same project folder with your Python environment activated.

**1. Install video support.**

```powershell
python -m pip install -r requirements/live.txt
```

**2. Add four clips to `videos/`.**

```text
videos/
├── north.mp4
├── south.mp4
├── east.mp4
└── west.mp4
```

Videos are not included in the GitHub download. You can also enter file paths in the dashboard, or use the same clip for all four directions to try it out.

**3. Open the video dashboard.**

```powershell
python -m streamlit run live_app.py
```

Check the four paths in the sidebar and click **Start**. Click **Stop** when you finish.

The vehicle model is stored at `models/yolo11s.pt`. CPU processing works; a compatible NVIDIA GPU can speed it up.

> **Accident demo:** Adding `videos/accident.mp4` makes it the default West feed. Accident detection checks West only by default and requires NCNN, listed below.

## Optional features

Install only the extras you want, in your activated environment.

| Feature | Install command | Next step |
| --- | --- | --- |
| Accident detection | `python -m pip install ncnn` | Use the West feed and the included accident model. |
| SMS alerts | `python -m pip install -r requirements/alerts.txt` | Copy `.env.example` to `.env` and enter your Twilio details. |
| Quantum experiment | `python -m pip install -r requirements/quantum.txt` | Use a separate Python 3.11 environment; see the [detailed guide](docs/SETUP.md). |

Keep API credentials in `.env`. This file is excluded from Git.

**How signal control works:** The video dashboard uses rules based on vehicle counts. The network simulator can optionally use QAOA, a quantum optimization algorithm running in simulation. Without a compatible quantum setup, it uses a classical solver called simulated annealing. The optional quantum setup uses Qiskit 1.x, not 2.x.

## Where is everything?

| File or folder | Purpose |
| --- | --- |
| `app.py` | Start the network simulator |
| `live_app.py` | Start the video dashboard |
| `traffic_optimizer/` | Application code, grouped by feature |
| `traffic_optimizer/core/config.py` | Detection, video, speed, and alert settings |
| `models/` | Vehicle and accident models |
| `videos/` | Your local input clips |
| `runtime/` | Generated logs and saved images |
| `requirements/` | Basic and optional dependencies |
| `tests/` | Automated checks |
| `docs/` | Detailed setup and maintenance notes |

Logs go to `runtime/logs/`, accident images to `runtime/accident_snapshots/`, and speed-alert images to `runtime/speed_violations/`. Videos and generated results stay local and are not committed to Git.

## For contributors

Run the checks from the project folder:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q traffic_optimizer app.py live_app.py
```

See the [detailed guide](docs/SETUP.md) for configuration and the [maintenance notes](docs/MAINTENANCE.md) for code organization and known limits.

## Project scope and license

This is a research and demonstration project. Speed, fuel savings, and emissions are estimates; accident predictions can be wrong. It has not been validated to control real traffic signals, and it does not claim a quantum speedup.

Source code uses the [MIT license](LICENSE). Model weights and third-party packages may have their own terms.
