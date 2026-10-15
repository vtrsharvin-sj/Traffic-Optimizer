"""traffic_optimizer package root."""
import os

# Repo root = one level ABOVE this package directory. Everything on disk
# (model weights, videos, snapshots, logs) is anchored here, NOT to the
# location of any individual module. See core/config.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
