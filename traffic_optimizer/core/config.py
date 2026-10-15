"""Repository-relative paths and live detection settings."""

import os

from traffic_optimizer import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")

# Load secrets from a local .env file (Twilio credentials, etc.) if present.
# Falls back silently to real environment variables if python-dotenv isn't
# installed. Never commit .env to source control.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass

# Local defaults; override any feed in the dashboard sidebar.
VIDEO_PATHS = {
    direction: os.path.join(VIDEO_DIR, direction.lower() + ".mp4")
    for direction in ("North", "South", "East", "West")
}
# Preserve the accident demonstration on West when its optional clip exists.
_accident_video = os.path.join(VIDEO_DIR, "accident.mp4")
if os.path.isfile(_accident_video):
    VIDEO_PATHS["West"] = _accident_video

# Only these approaches run the accident model. Per request: West only.
ACCIDENT_DIRECTIONS = ["West"]

# Base PyTorch weights. On first run with a CUDA GPU these are exported to a
# local TensorRT .engine (see ENGINE_PATH) for maximum speed.
MODEL_PATH = os.path.join(MODEL_DIR, "yolo11s.pt")

# Locally-built TensorRT engine. NOTE: a .engine is tied to the exact GPU +
# TensorRT version it was built on, so the Jetson-built engine is NOT reused
# here -- we export a fresh one from yolo11s.pt on THIS machine.
ENGINE_PATH = os.path.join(MODEL_DIR, "yolo11s.engine")

# Device selection: "auto" uses CUDA when available, else CPU.
DEVICE = "auto"

# Optional export on this machine; engines depend on GPU and TensorRT version.
USE_TENSORRT = False
HALF_PRECISION = True     # FP16 - big speedup on RTX GPUs

# Detection tuning -- balanced for SPEED + accuracy.
CONF_THRESHOLD = 0.25     # vehicle confidence (lower = detect more)
IOU_THRESHOLD = 0.55      # NMS IoU
IMG_SIZE = 640            # inference resolution (fast on GPU with FP16)
PROCESS_WIDTH = 640       # letterbox target size before inference
MAX_DETECT_FPS = 60.0     # cap on the 4-feed cycle rate
MAX_DET = 300             # allow many detections per frame (dense traffic)

# ------------------------------------------------------------------ #
# Accident detection (uses the trained NCNN accident model, 1 class)
# ------------------------------------------------------------------ #
ACCIDENT_MODEL_PATH = os.path.join(MODEL_DIR, "accident_ncnn_model")
ACCIDENT_ENABLED = True
ACCIDENT_CONF = 0.45           # confidence to call it an accident (raise to reduce false positives)
ACCIDENT_EVERY_N = 2           # run the accident model every N cycles (lower = detect sooner)
ACCIDENT_SNAPSHOT_DIR = os.path.join(RUNTIME_DIR, "accident_snapshots")

# ------------------------------------------------------------------ #
# Twilio SMS alerts (set your real credentials, or via env vars)
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Speed violation detection
# ------------------------------------------------------------------ #
SPEED_ENABLED = True
SPEED_LIMIT_KMH = 60           # flag vehicles faster than this
PIXELS_PER_METER = 12.0        # rough CCTV scale (tune per camera view)
SPEED_SMOOTHING = 0.7          # EMA factor for stable speed readings
SPEED_MAX_PLAUSIBLE_KMH = 150  # reject bigger jumps as tracker-ID-swap noise
SPEED_MIN_OBS = 3              # need this many consistent frames before trusting
VIOLATION_SNAPSHOT_DIR = os.path.join(RUNTIME_DIR, "speed_violations")
VIOLATION_COOLDOWN = 5.0       # seconds before the same vehicle can re-trigger

# ------------------------------------------------------------------ #
# Twilio SMS alerts (set your real credentials, or via env vars)
# ------------------------------------------------------------------ #
TWILIO_ENABLED = True
# All Twilio credentials come from the .env file (or real env vars) — never
# hard-code them here. See .env.example for the required keys.
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH", "")

# Trial accounts can ONLY send predefined template names (not free text).
# While on trial, keep this True so alerts still deliver. After UPGRADING the
# Twilio account, set it False to send the full descriptive alert messages.
TWILIO_TRIAL_TEMPLATES = os.environ.get("TWILIO_TRIAL_TEMPLATES", "1") == "1"
TWILIO_TEMPLATE = os.environ.get("TWILIO_TEMPLATE", "sms_appointment_reminders")
# Leave FROM blank -> alerts.py auto-detects a Twilio number on the account.
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")
TWILIO_TO = os.environ.get("TWILIO_TO", "")
SMS_COOLDOWN = 60              # seconds between accident alerts per direction

# ------------------------------------------------------------------ #
# Heavy-congestion SMS alert (recommend switching the signal)
# ------------------------------------------------------------------ #
CONGESTION_ALERT_ENABLED = True
CONGESTION_THRESHOLD = 12      # vehicles on one approach that counts as "heavy"
CONGESTION_COOLDOWN = 90       # seconds between congestion alerts per direction
# Only alert if that approach is NOT already the one getting green.
