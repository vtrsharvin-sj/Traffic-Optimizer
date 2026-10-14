"""
video_detector.py
------------------
Real-time vehicle detection from four traffic videos (one per approach:
North / South / East / West) using YOLO (Ultralytics, yolo11s.pt).

Each video is processed in its own background thread. For every processed frame
we:
  * run YOLO detection (with the built-in ByteTrack tracker),
  * keep only vehicle classes (car, motorcycle, bus, truck),
  * count the vehicles currently visible,
  * store the annotated frame (BGR ndarray) so the UI can display it live.

The rest of the platform reads the live per-direction vehicle counts and feeds
them straight into the QUBO / QAOA optimizer -- no simulated traffic.

This module has NO Arduino / Twilio / serial dependencies.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import os
from datetime import datetime

import cv2
import numpy as np

import config
from model_loader import load_model, describe, LoadedModel
from alerts import send_accident_sms
from speed_tracker import SpeedTracker
from simple_tracker import CentroidTracker
import logger

# COCO vehicle class ids: 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASSES = [2, 3, 5, 7]
DIRECTIONS = ["North", "South", "East", "West"]


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio and pad to a square `size` x `size`.

    This is what YOLO does internally and it preserves object shapes, which is
    important for accuracy. Squashing to a square (the old behaviour) distorted
    vehicles and hurt detection.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # grey pad
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def unletterbox_box(box, orig_w: int, orig_h: int, size: int):
    """Map an (x1,y1,x2,y2) box from a `size`x`size` letterbox back to the
    original image's pixel coordinates."""
    scale = size / max(orig_w, orig_h)
    nw, nh = orig_w * scale, orig_h * scale
    left = (size - nw) / 2.0
    top = (size - nh) / 2.0
    x1 = (box[0] - left) / scale
    y1 = (box[1] - top) / scale
    x2 = (box[2] - left) / scale
    y2 = (box[3] - top) / scale
    x1 = max(0.0, min(orig_w, x1)); x2 = max(0.0, min(orig_w, x2))
    y1 = max(0.0, min(orig_h, y1)); y2 = max(0.0, min(orig_h, y2))
    return (x1, y1, x2, y2)


def density_level(count: int) -> str:
    if count >= 15:
        return "CRITICAL"
    if count >= 10:
        return "HIGH"
    if count >= 5:
        return "MEDIUM"
    return "LOW"


@dataclass
class LaneState:
    """Live state for one approach/direction."""
    direction: str
    count: int = 0
    density: str = "LOW"
    frame: Optional[np.ndarray] = None   # latest annotated BGR frame
    accident: bool = False
    accident_conf: float = 0.0
    accident_snapshot: Optional[str] = None  # path to saved snapshot
    smoothed_count: int = 0
    violations: int = 0            # cumulative speed violations on this lane
    max_speed: float = 0.0         # fastest vehicle seen (km/h)
    fps: float = 0.0
    total_frames: int = 0
    finished: bool = False
    recent: deque = field(default_factory=lambda: deque(maxlen=6))


class MultiVideoDetector:
    """
    Runs YOLO on up to four videos concurrently and exposes live counts and
    annotated frames per direction. Thread-safe.
    """

    def __init__(self, video_paths: Dict[str, str], model_path: str = "yolo11s.pt",
                 conf: float = 0.35, process_width: int = 640,
                 max_fps: float = 30.0):
        self.video_paths = video_paths
        self.model_path = model_path
        self.conf = conf
        self.process_width = process_width
        self.min_dt = 1.0 / max_fps
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.lanes: Dict[str, LaneState] = {
            d: LaneState(direction=d) for d in video_paths
        }
        self.error: Optional[str] = None
        self.loaded: Optional[LoadedModel] = None
        self.backend: str = "loading..."
        self._accident_model = None
        self.accident_ready = False
        self._speed = SpeedTracker()
        self._tracker = CentroidTracker()
        self._violation_log: List[dict] = []   # recent violations (for UI)
        os.makedirs(config.ACCIDENT_SNAPSHOT_DIR, exist_ok=True)
        os.makedirs(config.VIOLATION_SNAPSHOT_DIR, exist_ok=True)

    # ---- lifecycle ------------------------------------------------------- #
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # ONE shared GPU model processing all four feeds in a single thread.
        # A TensorRT/CUDA context is not safe to call from many threads at once,
        # and one RTX-class GPU is fast enough to round-robin all four videos.
        self._thread = threading.Thread(target=self._run_all, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    # ---- reads ----------------------------------------------------------- #
    def get_counts(self) -> Dict[str, int]:
        # Return the smoothed (peak-over-window) count -- this is the stable
        # "vehicles present" figure that feeds the optimizer and the UI.
        with self._lock:
            return {d: s.smoothed_count for d, s in self.lanes.items()}

    def get_state(self, direction: str) -> LaneState:
        with self._lock:
            s = self.lanes[direction]
            # shallow copy of the scalar fields; frame is shared read-only
            return LaneState(direction=s.direction, count=s.count,
                             smoothed_count=s.smoothed_count,
                             density=s.density, frame=s.frame,
                             accident=s.accident, accident_conf=s.accident_conf,
                             accident_snapshot=s.accident_snapshot,
                             violations=s.violations, max_speed=s.max_speed,
                             fps=s.fps, total_frames=s.total_frames,
                             finished=s.finished)

    def get_frame_jpeg(self, direction: str):
        with self._lock:
            frame = self.lanes[direction].frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    # ---- single-thread batched runner ---------------------------------- #
    def _run_all(self) -> None:
        try:
            self.loaded = load_model()
            self.backend = describe(self.loaded)
            print(f"[detector] backend: {self.backend}")
        except Exception as exc:
            self.error = f"YOLO load failed: {exc}"
            for d in self.lanes:
                self.lanes[d].finished = True
            return

        model = self.loaded.model
        device = self.loaded.device

        # Load the accident model (NCNN, CPU) once.
        if config.ACCIDENT_ENABLED:
            try:
                from ultralytics import YOLO
                self._accident_model = YOLO(config.ACCIDENT_MODEL_PATH,
                                            task="detect")
                self.accident_ready = True
                print("[detector] accident model loaded")
            except Exception as exc:
                print(f"[detector] accident model load failed: {exc}")
                self._accident_model = None

        dirs = list(self.video_paths.keys())
        caps = {}
        for d in dirs:
            cap = cv2.VideoCapture(self.video_paths[d])
            if not cap.isOpened():
                self.error = f"Could not open video for {d}: {self.video_paths[d]}"
                for x in self.lanes:
                    self.lanes[x].finished = True
                return
            caps[d] = cap

        prev_time = time.time()
        cycle = 0
        while self._running:
            cycle += 1
            frames = []          # letterboxed frames for the vehicle model
            raw_frames = []      # original frames (for accident snapshots)
            active = []
            for d in dirs:
                cap = caps[d]
                ret, frame = cap.read()
                if not ret:                       # loop videos endlessly
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                # Aspect-preserving letterbox -> uniform size, no distortion.
                lb = letterbox(frame, self.process_width)
                frames.append(lb)
                raw_frames.append(frame)
                active.append(d)

            if not frames:
                time.sleep(0.05)
                continue

            # ONE batched inference for all four feeds -> GPU efficient.
            # FAST path: predict (FP16) is ~2x faster than track (ByteTrack).
            # We assign IDs with our own lightweight centroid tracker instead.
            try:
                results = model.predict(
                    frames, verbose=False, conf=self.conf,
                    iou=config.IOU_THRESHOLD, classes=VEHICLE_CLASSES,
                    device=device, imgsz=config.IMG_SIZE,
                    max_det=config.MAX_DET,
                    half=(device != "cpu"),
                )
            except Exception as exc:
                print(f"[detector] inference error: {exc}")
                time.sleep(0.1)
                continue

            # ---- accident detection (every N cycles) -------------------- #
            # Only the approaches listed in config.ACCIDENT_DIRECTIONS are
            # checked (per requirement: West only).
            accident_flags = {d: (False, 0.0) for d in active}
            if (self._accident_model is not None
                    and cycle % config.ACCIDENT_EVERY_N == 0):
                acc_dirs = [d for d in active
                            if d in config.ACCIDENT_DIRECTIONS]
                acc_frames = [raw_frames[active.index(d)] for d in acc_dirs]
                if acc_frames:
                    try:
                        a_results = self._accident_model.predict(
                            acc_frames, verbose=False,
                            conf=config.ACCIDENT_CONF, imgsz=640,
                        )
                        for d, ar in zip(acc_dirs, a_results):
                            if ar.boxes is not None and len(ar.boxes) > 0:
                                best = float(ar.boxes.conf.max())
                                accident_flags[d] = (True, best)
                    except Exception as exc:
                        print(f"[detector] accident inference error: {exc}")

            now = time.time()
            fps = 1.0 / max(1e-3, now - prev_time)
            prev_time = now

            for d, res, raw in zip(active, results, raw_frames):
                count = 0 if res.boxes is None else len(res.boxes)
                dens = density_level(count)
                annotated = res.plot()
                acc, acc_conf = accident_flags[d]

                # ---- speed estimation + violations --------------------- #
                oh, ow = raw.shape[:2]
                dets = []
                id_by_lbindex = {}
                if (config.SPEED_ENABLED and res.boxes is not None
                        and len(res.boxes) > 0):
                    lb_xyxy = res.boxes.xyxy.tolist()   # letterbox coords
                    orig_boxes = [unletterbox_box(b, ow, oh, self.process_width)
                                  for b in lb_xyxy]
                    ids = self._tracker.update(d, orig_boxes)
                    for k, (tid, ob) in enumerate(zip(ids, orig_boxes)):
                        dets.append((tid, ob))
                        id_by_lbindex[k] = tid
                speeds, violations = self._speed.update(d, dets)
                _draw_speeds(annotated, res, speeds, id_by_lbindex)

                self._handle_lane(d, count, dens, annotated, raw, acc,
                                  acc_conf, fps, speeds, violations)

            spare = self.min_dt - (time.time() - now)
            if spare > 0:
                time.sleep(spare)

        for cap in caps.values():
            cap.release()
        for d in self.lanes:
            self.lanes[d].finished = True

    # ---- per-lane bookkeeping (counts, accident, speed, snapshot, SMS) -- #
    def _handle_lane(self, d, count, dens, annotated, raw, acc, acc_conf, fps,
                     speeds=None, violations=None):
        newly_detected = False
        with self._lock:
            st = self.lanes[d]
            was_accident = st.accident
            # Smooth out frame-to-frame noise: report the peak vehicle count
            # over the last few frames so a single frame that misses cars
            # doesn't make the number collapse. This gives a stable, inclusive
            # "vehicles present on this approach" reading.
            st.recent.append(count)
            smoothed = max(st.recent)
            st.count = count            # raw current-frame count (header)
            st.smoothed_count = smoothed
            st.density = density_level(smoothed)
            st.fps = fps
            st.total_frames += 1
            if speeds:
                fastest = max(speeds.values()) if speeds else 0.0
                st.max_speed = round(max(st.max_speed, fastest), 1)
            if acc:
                if not was_accident:
                    newly_detected = True
                st.accident = True
                st.accident_conf = acc_conf
            # note: accident latches on; cleared only when user resets
            snapshot_path = st.accident_snapshot

        # ---- speed violations: crop + save + log ----------------------- #
        if violations:
            for v in violations:
                self._save_violation(v, raw)

        # draw header + accident banner on the annotated frame
        _draw_header(annotated, d, count, dens, fps * 4,
                     accident=(acc or self.lanes[d].accident))

        # save a snapshot + fire SMS the first time an accident is seen here
        if newly_detected:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(config.ACCIDENT_SNAPSHOT_DIR,
                                f"accident_{d}_{ts}.jpg")
            try:
                cv2.imwrite(path, raw)
                with self._lock:
                    self.lanes[d].accident_snapshot = path
                print(f"[detector] ACCIDENT in {d} (conf {acc_conf:.2f}) "
                      f"snapshot -> {path}")
            except Exception as exc:
                print(f"[detector] snapshot save failed: {exc}")
            # log to JSON + CSV
            try:
                logger.log_accident(d, acc_conf, path)
            except Exception as exc:
                print(f"[detector] accident log failed: {exc}")
            # SMS (non-blocking, has its own cooldown)
            send_accident_sms(d, acc_conf)

        with self._lock:
            self.lanes[d].frame = annotated

    def get_backend(self) -> str:
        return self.backend

    def clear_accident(self, direction: str) -> None:
        with self._lock:
            st = self.lanes[direction]
            st.accident = False
            st.accident_conf = 0.0
            st.accident_snapshot = None

    def clear_all_accidents(self) -> None:
        for d in list(self.lanes):
            self.clear_accident(d)

    # ---- speed violation handling -------------------------------------- #
    def _save_violation(self, v, raw) -> None:
        x1, y1, x2, y2 = v.box
        h, w = raw.shape[:2]
        pad = 15
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        crop = raw[y1p:y2p, x1p:x2p]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (f"violation_{v.direction}_id{v.vehicle_id}_"
                 f"{int(v.speed_kmh)}kmh_{ts}.jpg")
        path = os.path.join(config.VIOLATION_SNAPSHOT_DIR, fname)
        try:
            if crop.size > 0:
                cv2.imwrite(path, crop)
            else:
                cv2.imwrite(path, raw)
            print(f"[detector] SPEED VIOLATION {v.direction} id{v.vehicle_id} "
                  f"{v.speed_kmh:.0f} km/h -> {path}")
        except Exception as exc:
            print(f"[detector] violation snapshot failed: {exc}")
            path = ""
        with self._lock:
            self.lanes[v.direction].violations += 1
            self._violation_log.append({
                "direction": v.direction, "vehicle_id": v.vehicle_id,
                "speed": round(v.speed_kmh, 1), "snapshot": path,
                "time": ts})
            self._violation_log = self._violation_log[-50:]
        try:
            logger.log_violation(v.direction, v.vehicle_id, v.speed_kmh, path)
        except Exception as exc:
            print(f"[detector] violation log failed: {exc}")

    def get_violations(self) -> List[dict]:
        with self._lock:
            return list(self._violation_log)


def _draw_speeds(annotated, res, speeds, id_by_index) -> None:
    """Overlay speed (km/h) labels on the annotated (letterbox) frame."""
    if res.boxes is None or not speeds:
        return
    xyxy = res.boxes.xyxy.tolist()
    for k, b in enumerate(xyxy):
        tid = id_by_index.get(k)
        if tid is None:
            continue
        spd = speeds.get(tid)
        if spd is None or spd < 1:
            continue
        x1, y1 = int(b[0]), int(b[1])
        over = spd > config.SPEED_LIMIT_KMH
        color = (0, 0, 255) if over else (0, 220, 0)
        label = f"{spd:.0f}km/h" + (" !" if over else "")
        cv2.putText(annotated, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _draw_header(img: np.ndarray, direction: str, count: int,
                 density: str, fps: float, accident: bool = False) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    color = {
        "CRITICAL": (0, 0, 255), "HIGH": (0, 140, 255),
        "MEDIUM": (0, 220, 255), "LOW": (0, 220, 0),
    }[density]
    cv2.putText(img, f"{direction}  Vehicles: {count}  [{density}]",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, f"{fps:4.1f} FPS", (w - 90, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    if accident:
        # flashing red border + warning text
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        txt = "!!! ACCIDENT DETECTED !!!"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (0, h - 40), (tw + 20, h), (0, 0, 255), -1)
        cv2.putText(img, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
