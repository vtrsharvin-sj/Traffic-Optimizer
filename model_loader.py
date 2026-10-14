"""
model_loader.py
---------------
Loads the fastest available YOLO model for THIS machine.

Priority:
  1. TensorRT engine (yolo11s.engine) on CUDA  -> fastest
  2. PyTorch weights (yolo11s.pt) on CUDA (FP16)
  3. PyTorch weights on CPU                     -> slow fallback

IMPORTANT about .engine files
-----------------------------
A TensorRT .engine is compiled for one specific GPU + TensorRT + CUDA version.
The engine that came from the Jetson Orin will NOT load on a Windows RTX GPU.
So we EXPORT a fresh engine from yolo11s.pt on this machine the first time
(cached to disk), then reuse it on subsequent runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import config


@dataclass
class LoadedModel:
    model: object
    device: str          # "cuda:0" or "cpu"
    kind: str            # "tensorrt" | "pytorch-cuda" | "pytorch-cpu"
    half: bool


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def export_engine(pt_path: str, engine_path: str, imgsz: int,
                  half: bool) -> bool:
    """
    Build a TensorRT engine from the .pt weights on the current GPU.
    Returns True on success. Requires the `tensorrt` python package.
    """
    from ultralytics import YOLO
    try:
        m = YOLO(pt_path)
        # Ultralytics writes "<stem>.engine" next to the .pt file.
        out = m.export(format="engine", imgsz=imgsz, half=half,
                       device=0, verbose=False)
        produced = str(out)
        if produced and os.path.exists(produced) and produced != engine_path:
            # normalise location/name to ENGINE_PATH
            try:
                if os.path.exists(engine_path):
                    os.remove(engine_path)
                os.replace(produced, engine_path)
            except Exception:
                # if rename fails, just use whatever path export produced
                config.ENGINE_PATH = produced  # type: ignore
        return os.path.exists(config.ENGINE_PATH)
    except Exception as exc:
        print(f"[model_loader] TensorRT export failed: {exc}")
        return False


def load_model() -> LoadedModel:
    """Return the best model we can build on this machine."""
    from ultralytics import YOLO

    want_cuda = (config.DEVICE in ("auto", "cuda", "0", 0)) and _cuda_available()

    # ---- CPU path -------------------------------------------------------- #
    if not want_cuda:
        model = YOLO(config.MODEL_PATH)
        return LoadedModel(model=model, device="cpu",
                           kind="pytorch-cpu", half=False)

    print(f"[model_loader] CUDA GPU detected: {_gpu_name()}")

    # ---- TensorRT engine path ------------------------------------------- #
    if config.USE_TENSORRT:
        engine = config.ENGINE_PATH
        if not os.path.exists(engine):
            print("[model_loader] Building TensorRT engine (first run only)...")
            export_engine(config.MODEL_PATH, engine, config.IMG_SIZE,
                          config.HALF_PRECISION)
            engine = config.ENGINE_PATH
        if os.path.exists(engine):
            try:
                model = YOLO(engine, task="detect")
                return LoadedModel(model=model, device="cuda:0",
                                   kind="tensorrt", half=config.HALF_PRECISION)
            except Exception as exc:
                print(f"[model_loader] Engine load failed ({exc}); "
                      f"falling back to PyTorch CUDA.")

    # ---- PyTorch CUDA path ---------------------------------------------- #
    model = YOLO(config.MODEL_PATH)
    try:
        model.to("cuda:0")
    except Exception:
        pass
    return LoadedModel(model=model, device="cuda:0",
                       kind="pytorch-cuda", half=config.HALF_PRECISION)


def describe(lm: LoadedModel) -> str:
    label = {
        "tensorrt": "TensorRT engine (FP16) on GPU",
        "pytorch-cuda": "PyTorch on CUDA (FP16)",
        "pytorch-cpu": "PyTorch on CPU (slow)",
    }[lm.kind]
    if lm.device.startswith("cuda"):
        return f"{label} — {_gpu_name()}"
    return label
