"""Streamlit launcher; executes the dashboard on every rerun."""
import runpy
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)
runpy.run_module("traffic_optimizer.apps.live_app", run_name="__main__")
