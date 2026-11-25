#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetsonDualCam (Python / Qt) — Hybrid Vision (YOLO + Math + OpenCV)

Detection pipeline:
  Layer 1 : YOLO (.pt/.onnx via ultralytics, or legacy .engine via TensorRT)
  Layer 2 : Math  — pin count + alignment std-dev          (Phase 3)
  Layer 3 : OpenCV — surface contamination threshold        (Phase 4)

Legacy 4-class .engine mode preserved for backward compatibility.
"""

import os
import sys
import re
import glob
import time
import json
import csv
import subprocess
import traceback
import asyncio
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Prevent ultralytics from auto-installing packages (blocks UI on first run)
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "False")

import numpy as np
import cv2
from local_retention import LocalRetentionStore

# Qt compat — PySide6 (x86/pip) or PyQt5 (Jetson system packages)
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal
    _QT_EXEC = staticmethod(lambda app: app.exec())
    _QT_BACKEND = "PySide6"
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import pyqtSignal as Signal
    _QT_EXEC = staticmethod(lambda app: app.exec_())
    _QT_BACKEND = "PyQt5"

# Jetson detection — True when running on Jetson hardware
_IS_JETSON = os.path.exists("/etc/nv_tegra_release")

_HOME = os.path.expanduser("~")
DEFAULT_LOCAL_SAVE_BASE = os.path.join(_HOME, "Local_Backup", "PeaPLED_SideCAM")
LOCAL_STORAGE_ROOT = os.environ.get("LOCAL_STORAGE_ROOT", DEFAULT_LOCAL_SAVE_BASE)
LOCAL_STORAGE_MAX_GB = float(os.environ.get("LOCAL_STORAGE_MAX_GB", "60"))
LOCAL_STORAGE_DELETE_GB = float(os.environ.get("LOCAL_STORAGE_DELETE_GB", "30"))

# TensorRT / CUDA — optional (legacy .engine only)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    _TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except Exception:
    trt = None
    cuda = None
    _TRT_AVAILABLE = False
    TRT_LOGGER = None

try:
    import gpiod
    # GPIO code below uses the libgpiod v1 Python API.  Do not expose a
    # non-working GPIO option when only the incompatible v2 API is installed.
    if not all(hasattr(gpiod, name) for name in ("Chip", "LINE_REQ_DIR_IN")):
        gpiod = None
except Exception:
    gpiod = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except Exception:
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    uvicorn = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
APP_NAME = "JetsonDualCamPy"
PIN_CONFIG_FILENAME = "PeaPLED_Config.csv"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(APP_DIR, "models"),
    os.path.abspath(os.path.join(APP_DIR, os.pardir, "train-7", "weights")),
]
WEIGHTS_DIR = MODEL_DIRS[0]

# Hybrid model (train-7: 2 classes)
HYBRID_CLASSES        = ["CONNECTOR", "PIN"]
HYBRID_CLS_CONNECTOR  = 0
HYBRID_CLS_PIN        = 1

# Legacy model (old 4-class .engine)
LEGACY_CLASSES = ["BROKEN_CONNECTOR", "CONNECTOR", "PIN_MISSING", "SCRATCH"]
THRESH_OVERLAY        = 0.10
THRESH_PASS_CONNECTOR = 0.50
THRESH_PASS_OTHERS    = 0.10

# Phase 3 defaults
DEFAULT_EXPECTED_PINS     = 4
DEFAULT_PIN_STD_THRESHOLD = 15.0   # pixels

# Phase 4 defaults  (HSV yellow/orange hue filter — see phase4_compute)
DEFAULT_SURFACE_BRIGHT_THR = 80    # HSV V-channel minimum (0-255); only pixels brighter than this count
DEFAULT_SURFACE_AREA_THR   = 500   # max allowed anomaly pixel count before FAIL
DEFAULT_SURFACE_SAT_THR    = 80    # HSV S-channel minimum (0-255); yellow/orange above this threshold
DEFAULT_SURFACE_DIFF_THR   = 100   # Golden template: per-pixel diff threshold — 100 ignores natural
                                   # surface variation (<100 diff) but catches high-contrast contamination

# Golden template dimensions (connector ROI is normalized to this size before saving/comparing)
TEMPLATE_W, TEMPLATE_H = 640, 320


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def app_config_path() -> str:
    # Keep this copy independent from older JetsonDualCam installations that
    # use the same Qt organization/application name in ~/.config.
    base = os.path.join(APP_DIR, ".appdata")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "config.json")


def default_config() -> dict:
    return {
        "lh_cam_index": 0,
        "rh_cam_index": 1,
        "lh_cam_device": "",
        "rh_cam_device": "",
        "lh_res_w": 1920, "lh_res_h": 1080,
        "rh_res_w": 1920, "rh_res_h": 1080,
        "capture_settle_ms": 80,
        "capture_discard_frames": 3,
        "force_pass": False,
        "conf_pct": 50,
        "cam_ctrl_profile": {
            "enabled": True,
            "order": [
                "auto_exposure", "exposure_time_absolute",
                "white_balance_automatic", "white_balance_temperature",
                "focus_automatic_continuous",
                "brightness", "contrast", "saturation", "sharpness",
            ],
            "ctrls": {
                "auto_exposure": 1,
                "exposure_time_absolute": 10,
                "white_balance_automatic": 0,
                "white_balance_temperature": 3,
                "focus_automatic_continuous": 0,
                "brightness": 128,
                "contrast": 128,
                "saturation": 64,
                "sharpness": 128,
            },
        },
        "lh_cam_ctrl": {},
        "rh_cam_ctrl": {},
        # Hybrid Vision
        "expected_pins":      DEFAULT_EXPECTED_PINS,
        "pin_std_threshold":  DEFAULT_PIN_STD_THRESHOLD,
        "surface_bright_thr": DEFAULT_SURFACE_BRIGHT_THR,
        "surface_area_thr":   DEFAULT_SURFACE_AREA_THR,
        "surface_sat_thr":    DEFAULT_SURFACE_SAT_THR,
        "surface_diff_thr":   DEFAULT_SURFACE_DIFF_THR,
        # Input source
        "input_source": "live",
    }


def _deep_merge_dict(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config() -> dict:
    cfg = default_config()
    p = app_config_path()
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                _deep_merge_dict(cfg, user_cfg)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    p = app_config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # Atomic replace keeps the last valid config if the disk fills mid-write.
    tmp = f"{p}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Model auto-discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_model_file(weights_dir: str = "") -> str:
    """Prefer Hybrid models used by Phase 3/4 and Dashboard over legacy engines."""
    search_dirs = [weights_dir] if weights_dir else MODEL_DIRS
    for name in ("best.onnx", "best.pt", "best.engine", "best_fp16.engine"):
        for model_dir in search_dirs:
            p = os.path.join(model_dir, name)
            if os.path.isfile(p):
                return p
    for ext in (".onnx", ".pt", ".engine"):
        for model_dir in search_dirs:
            found = sorted(glob.glob(os.path.join(model_dir, f"*{ext}")))
            if found:
                return found[0]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    in_w: int
    in_h: int
    orig_w: int
    orig_h: int


def letterbox_bgr(
    img_bgr: np.ndarray,
    new_shape: Tuple[int, int],
    color=(114, 114, 114),
) -> Tuple[np.ndarray, LetterboxMeta]:
    h0, w0 = img_bgr.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])
    r = min(new_w / w0, new_h / h0)
    unpad_w, unpad_h = int(round(w0 * r)), int(round(h0 * r))
    pad_w, pad_h = new_w - unpad_w, new_h - unpad_h
    pad_x, pad_y = pad_w / 2, pad_h / 2
    resized = cv2.resize(img_bgr, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    x1, y1 = int(round(pad_x)), int(round(pad_y))
    out[y1 : y1 + unpad_h, x1 : x1 + unpad_w] = resized
    meta = LetterboxMeta(
        scale=r, pad_x=pad_x, pad_y=pad_y,
        in_w=new_w, in_h=new_h, orig_w=w0, orig_h=h0,
    )
    return out, meta


def bgr_to_chw_float32(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    img = img[:, :, ::-1]
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)


def qimage_from_bgr(mat_bgr: np.ndarray) -> QtGui.QImage:
    if mat_bgr is None or mat_bgr.size == 0:
        return QtGui.QImage()
    rgb = cv2.cvtColor(mat_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(
        rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888
    ).copy()


def scale_back_det(det: dict, meta: LetterboxMeta) -> dict:
    x = max(0.0, min(float(meta.orig_w),
                     (float(det["cx"]) - meta.pad_x) / meta.scale))
    y = max(0.0, min(float(meta.orig_h),
                     (float(det["cy"]) - meta.pad_y) / meta.scale))
    w = float(det["w"]) / meta.scale
    h = float(det["h"]) / meta.scale
    return {**det, "cx": x, "cy": y, "w": w, "h": h}


def rotated_box_points(cx, cy, w, h, angle_rad) -> np.ndarray:
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T
    pts[:, 0] += float(cx)
    pts[:, 1] += float(cy)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# TrtRunner — legacy .engine (TRT 10 / PyCUDA)
# ─────────────────────────────────────────────────────────────────────────────
if _TRT_AVAILABLE:
    class TrtRunner:
        def __init__(self, engine_path: str):
            self.engine_path = engine_path
            self.runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
            self.ctx = self.engine.create_execution_context()
            if self.ctx is None:
                raise RuntimeError("Failed to create execution context")
            self.in_name, self.out_names = self._find_io()
            self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
            self.out_shapes = [
                tuple(self.engine.get_tensor_shape(n)) for n in self.out_names
            ]
            self.stream = cuda.Stream()
            self.d_in = cuda.mem_alloc(
                int(np.prod(self.in_shape)) * np.dtype(np.float32).itemsize
            )
            self.d_out = [
                cuda.mem_alloc(int(np.prod(sh)) * np.dtype(np.float32).itemsize)
                for sh in self.out_shapes
            ]
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            for n, ptr in zip(self.out_names, self.d_out):
                self.ctx.set_tensor_address(n, int(ptr))

        def _find_io(self):
            in_name, outs = None, []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    in_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    outs.append(name)
            if in_name is None or not outs:
                raise RuntimeError("Could not find IO tensors")
            return in_name, outs

        def infer(self, inp_chw: np.ndarray) -> List[np.ndarray]:
            inp_chw = np.ascontiguousarray(inp_chw.astype(np.float32))
            cuda.memcpy_htod_async(self.d_in, inp_chw, self.stream)
            if not self.ctx.execute_async_v3(stream_handle=self.stream.handle):
                raise RuntimeError("execute_async_v3 returned False")
            outs = []
            for sh, ptr in zip(self.out_shapes, self.d_out):
                arr = np.empty(sh, dtype=np.float32)
                cuda.memcpy_dtoh_async(arr, ptr, self.stream)
                outs.append(arr)
            self.stream.synchronize()
            return outs
else:
    class TrtRunner:  # type: ignore
        def __init__(self, engine_path: str):
            raise RuntimeError("TensorRT not available on this system")


# ─────────────────────────────────────────────────────────────────────────────
# YoloRunner — ultralytics .pt / .onnx (Hybrid Vision)
# ─────────────────────────────────────────────────────────────────────────────
class YoloRunner:
    """Ultralytics YOLO runner for .pt and .onnx models."""

    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise ImportError(
                "ultralytics package required. Run: pip install ultralytics"
            )
        # Pre-check onnxruntime before ultralytics tries to auto-install it
        if model_path.lower().endswith(".onnx"):
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                raise ImportError(
                    "onnxruntime is required for .onnx models.\n"
                    "Run: pip install onnxruntime"
                )
        self.model = YOLO(model_path, task="detect")
        self.model_path = model_path
        task = getattr(self.model, "task", "") or ""
        self.is_obb = "obb" in str(task).lower()

    def infer(self, bgr: np.ndarray, conf: float = 0.25) -> List[dict]:
        # Force CPU for .onnx models — onnxruntime CUDA provider may not load
        # (e.g. libcublasLt missing) causing a DeviceType:1→0 tensor copy error
        infer_device = "cpu" if self.model_path.lower().endswith(".onnx") else None
        call_kwargs: dict = dict(conf=conf, verbose=False)
        if infer_device is not None:
            call_kwargs["device"] = infer_device
        results = self.model(bgr, **call_kwargs)
        if not results:
            return []
        r = results[0]
        dets: List[dict] = []

        if self.is_obb and r.obb is not None and len(r.obb):
            for i in range(len(r.obb)):
                xywhr = r.obb.xywhr[i].cpu().numpy()
                dets.append(dict(
                    cx=float(xywhr[0]), cy=float(xywhr[1]),
                    w=float(xywhr[2]),  h=float(xywhr[3]),
                    ang=float(xywhr[4]),
                    cls_id=int(r.obb.cls[i].cpu().item()),
                    score=float(r.obb.conf[i].cpu().item()),
                ))
        elif r.boxes is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                xyxy = r.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                dets.append(dict(
                    cx=float((x1 + x2) / 2), cy=float((y1 + y2) / 2),
                    w=float(x2 - x1),        h=float(y2 - y1),
                    ang=0.0,
                    cls_id=int(r.boxes.cls[i].cpu().item()),
                    score=float(r.boxes.conf[i].cpu().item()),
                ))
        return dets


class OnnxRunner:
    """Lightweight Ultralytics detect-model runner without PyTorch."""

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError("onnxruntime is required for .onnx models") from exc
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])
        self.model_path = model_path

    def infer(self, bgr: np.ndarray, conf: float = 0.25) -> List[dict]:
        lb, meta = letterbox_bgr(bgr, (self.in_h, self.in_w))
        inp = bgr_to_chw_float32(lb)[None, ...]
        pred = np.asarray(self.session.run(None, {self.input_name: inp})[0])
        if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[1] < 6:
            return []

        raw = pred[0].T  # N x (cx, cy, w, h, class scores...)
        scores_all = raw[:, 4:]
        cls_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(len(raw)), cls_ids]
        keep = scores >= float(conf)
        raw, cls_ids, scores = raw[keep], cls_ids[keep], scores[keep]
        if not len(raw):
            return []

        boxes_xywh = []
        for row in raw:
            cx, cy, w, h = [float(v) for v in row[:4]]
            boxes_xywh.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

        # Suppress boxes per class so pins never suppress the connector box.
        selected: List[int] = []
        for cls_id in np.unique(cls_ids):
            idxs = np.flatnonzero(cls_ids == cls_id)
            nms = cv2.dnn.NMSBoxes(
                [boxes_xywh[i] for i in idxs],
                [float(scores[i]) for i in idxs],
                float(conf), 0.45,
            )
            if len(nms):
                selected.extend(int(idxs[int(i)]) for i in np.asarray(nms).reshape(-1))

        dets = []
        for i in selected:
            det = dict(
                cx=float(raw[i, 0]), cy=float(raw[i, 1]),
                w=float(raw[i, 2]), h=float(raw[i, 3]),
                ang=0.0, cls_id=int(cls_ids[i]), score=float(scores[i]),
            )
            dets.append(scale_back_det(det, meta))
        return dets


# ─────────────────────────────────────────────────────────────────────────────
# Legacy decode helpers (4-class .engine)
# ─────────────────────────────────────────────────────────────────────────────
def best_per_class_raw_head(
    out: np.ndarray,
) -> Tuple[List[dict], np.ndarray]:
    if out.ndim != 3 or out.shape[0] != 1:
        return [], np.zeros(4, dtype=np.float32)
    C, N = out.shape[1], out.shape[2]
    if C < 9:
        return [], np.zeros(4, dtype=np.float32)
    o = out[0]
    cx, cy, w, h, scores, ang = o[0], o[1], o[2], o[3], o[4:8], o[8]
    dets, best_scores = [], np.zeros(4, dtype=np.float32)
    for cls_id in range(4):
        idx = int(np.argmax(scores[cls_id]))
        sc = float(scores[cls_id, idx])
        best_scores[cls_id] = sc
        dets.append(dict(
            cx=float(cx[idx]), cy=float(cy[idx]),
            w=float(w[idx]),   h=float(h[idx]),
            ang=float(ang[idx]), cls_id=cls_id, score=sc,
        ))
    return dets, best_scores


def verdict_pass_fail_from_scores(best_scores: np.ndarray) -> str:
    if best_scores is None or best_scores.size < 4:
        return "FAIL"
    sc0, sc1, sc2, sc3 = [float(x) for x in best_scores[:4]]
    if (sc1 >= THRESH_PASS_CONNECTOR and sc0 < THRESH_PASS_OTHERS
            and sc2 < THRESH_PASS_OTHERS and sc3 < THRESH_PASS_OTHERS):
        return "PASS"
    return "FAIL"


def draw_overlay_legacy(
    img_bgr: np.ndarray, dets: List[dict], best_scores: np.ndarray
) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        if float(d.get("score", 0.0)) < THRESH_OVERLAY:
            continue
        cls_id = int(d.get("cls_id", -1))
        name = (LEGACY_CLASSES[cls_id]
                if 0 <= cls_id < len(LEGACY_CLASSES) else f"cls{cls_id}")
        pts = rotated_box_points(d["cx"], d["cy"], d["w"], d["h"], d["ang"])
        cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)], True,
                      (0, 255, 255), 2)
        cv2.putText(out, f"{name} {d['score']:.2f}",
                    (int(d["cx"]), int(d["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if best_scores is not None and best_scores.size >= 4:
        sc0, sc1, sc2, sc3 = [float(x) for x in best_scores[:4]]
        cv2.putText(out,
                    f"BC:{sc0:.2f} C:{sc1:.2f} PM:{sc2:.2f} S:{sc3:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Pin Count & Alignment Math
# ─────────────────────────────────────────────────────────────────────────────
def phase3_compute(
    dets: List[dict],
    expected_pins: int,
    std_threshold: float,
) -> Tuple[str, str, dict]:
    """Returns (verdict, reason_code, detail_dict)."""
    connectors = [d for d in dets if d["cls_id"] == HYBRID_CLS_CONNECTOR]
    pins_all   = [d for d in dets if d["cls_id"] == HYBRID_CLS_PIN]

    _empty = {"pin_count": 0, "exp_pins": expected_pins,
               "std_y": 0.0, "pins_sorted": [], "connector": None}
    if not connectors:
        return "FAIL", "no_connector", _empty

    conn = max(connectors, key=lambda d: d["w"] * d["h"])
    cx1 = conn["cx"] - conn["w"] / 2
    cx2 = conn["cx"] + conn["w"] / 2
    cy1 = conn["cy"] - conn["h"] / 2
    cy2 = conn["cy"] + conn["h"] / 2

    pins_in = [
        p for p in pins_all
        if cx1 <= p["cx"] <= cx2 and cy1 <= p["cy"] <= cy2
    ]
    pins_sorted = sorted(pins_in, key=lambda p: p["cx"])

    std_y = 0.0
    if len(pins_in) >= 2:
        std_y = float(np.std([p["cy"] for p in pins_in]))

    detail = {
        "pin_count": len(pins_in),
        "exp_pins":  expected_pins,
        "std_y":     std_y,
        "pins_sorted": pins_sorted,
        "connector": conn,
    }

    if len(pins_in) != expected_pins:
        return "FAIL", f"pin_count:{len(pins_in)}_exp:{expected_pins}", detail

    if len(pins_in) >= 2 and std_y > std_threshold:
        return "FAIL", f"bent_pin_std:{std_y:.1f}px", detail

    return "PASS", "OK", detail


def phase3_pin_check(
    dets: List[dict],
    expected_pins: int,
    std_threshold: float,
) -> Tuple[str, str]:
    """Returns (verdict, reason_code)."""
    v, r, _ = phase3_compute(dets, expected_pins, std_threshold)
    return v, r


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Surface Contamination Check
# ─────────────────────────────────────────────────────────────────────────────
def phase4_compute(
    bgr: np.ndarray,
    dets: List[dict],
    bright_threshold: int,
    area_threshold: int,
    sat_threshold: int = DEFAULT_SURFACE_SAT_THR,
) -> Tuple[str, str, dict]:
    """Returns (verdict, reason_code, detail_dict).

    Detects contamination on the connector housing via two channels:
      1. Yellow/orange (flux, tape, debris): H=15-45, S>sat_thr, V>bright_thr — full image width.
      2. White/cream (fibers, plastic debris): V>190, S<60 — inside connector cavity at pin-row level.

    ROI strategy:
      - Yellow ch: full image width × cy1 to max_pin_y+25 (catches outer-wall contamination).
      - White  ch: connector bbox X × min_pin_y-10 to max_pin_y+25 (cavity interior only —
           avoids gray outer housing which has V≈100-160, well below the V>190 threshold).
    Pin areas masked +8px. Morphological opening 3×3 removes isolated noise.
    """
    connectors = [d for d in dets if d["cls_id"] == HYBRID_CLS_CONNECTOR]
    if not connectors:
        return "PASS", "OK", {"surface_area": 0}

    conn  = max(connectors, key=lambda d: d["w"] * d["h"])
    pins  = [d for d in dets if d["cls_id"] == HYBRID_CLS_PIN]
    h_img, w_img = bgr.shape[:2]

    cy1 = max(0, int(conn["cy"] - conn["h"] / 2))
    min_pin_y = None
    max_pin_y = None
    if pins:
        min_pin_y = min(int(p["cy"] - p["h"] / 2) for p in pins)
        max_pin_y = max(int(p["cy"] + p["h"] / 2) for p in pins)
        roi_y2 = min(h_img, max_pin_y + 100)
    else:
        roi_y2 = min(h_img, int(conn["cy"] + conn["h"] / 2))

    if roi_y2 <= cy1:
        return "PASS", "OK", {"surface_area": 0}

    # Refine YOLO bbox to precise connector housing boundary (excludes PCB foam)
    conn_cx1 = max(0, int(conn["cx"] - conn["w"] / 2))
    conn_cx2 = min(w_img, int(conn["cx"] + conn["w"] / 2))
    rx1, ry1_ref, rx2, ry2 = _refine_conn_bbox(bgr, conn_cx1, cy1, conn_cx2, roi_y2)
    roi_y2 = ry2      # refined bottom — excludes PCB foam
    roi_x1 = rx1      # refined left edge
    roi_x2 = rx2      # refined right edge

    # ── Channel 1: Yellow/orange — connector-width ROI ───────────────────────
    roi = bgr[cy1:roi_y2, roi_x1:roi_x2].copy()

    pin_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for p in pins:
        exp = 8
        px1 = max(0, int(p["cx"] - p["w"] / 2) - roi_x1 - exp)
        py1 = max(0, int(p["cy"] - p["h"] / 2) - cy1 - exp)
        px2 = min(roi_x2 - roi_x1, int(p["cx"] + p["w"] / 2) - roi_x1 + exp)
        py2 = min(roi.shape[0], int(p["cy"] + p["h"] / 2) - cy1 + exp)
        if px2 > px1 and py2 > py1:
            pin_mask[py1:py2, px1:px2] = 255

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = (
        (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) &
        (hsv[:, :, 1] > int(sat_threshold)) &
        (hsv[:, :, 2] > int(bright_threshold))
    ).astype(np.uint8)
    yellow_mask[pin_mask == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    yellow_clean = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_area  = int(np.count_nonzero(yellow_clean))
    n_lbl_y, _, cc_stats_y, _ = cv2.connectedComponentsWithStats(yellow_clean, connectivity=8)
    yellow_max_blob = int(cc_stats_y[1:, cv2.CC_STAT_AREA].max()) if n_lbl_y > 1 else 0

    # ── Channel 2: White/cream debris — connector cavity interior ────────────
    # Skip the first/last 50px of the connector bbox (housing left/right walls
    # have V>190 naturally; real debris sits in the interior cavity).
    white_area = 0
    white_max_blob = 0
    if pins and min_pin_y is not None:
        wb_y1 = max(0, min_pin_y - 10)
        wb_y2 = roi_y2
        wb_x1 = roi_x1 + 20    # use refined x-bounds (smaller trim — already clipped to housing)
        wb_x2 = roi_x2 - 20

        if wb_x2 > wb_x1 and wb_y2 > wb_y1:
            wb_roi = bgr[wb_y1:wb_y2, wb_x1:wb_x2].copy()
            wb_hsv = cv2.cvtColor(wb_roi, cv2.COLOR_BGR2HSV)

            wb_pin_mask = np.zeros(wb_roi.shape[:2], dtype=np.uint8)
            for p in pins:
                exp = 8
                bpx1 = max(0, int(p["cx"] - p["w"] / 2) - wb_x1 - exp)
                bpy1 = max(0, int(p["cy"] - p["h"] / 2) - wb_y1 - exp)
                bpx2 = min(wb_roi.shape[1], int(p["cx"] + p["w"] / 2) - wb_x1 + exp)
                bpy2 = min(wb_roi.shape[0], int(p["cy"] + p["h"] / 2) - wb_y1 + exp)
                if bpx2 > bpx1 and bpy2 > bpy1:
                    wb_pin_mask[bpy1:bpy2, bpx1:bpx2] = 255

            # V>190 catches white/cream (cavity bg is dark V≈20-60, housing walls V≈100-160)
            # S<60 excludes yellow/orange that might share high V
            white_mask = (
                (wb_hsv[:, :, 2] > 190) & (wb_hsv[:, :, 1] < 60)
            ).astype(np.uint8)
            white_mask[wb_pin_mask == 255] = 0
            white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_area  = int(np.count_nonzero(white_clean))
            n_lbl_w, _, cc_stats_w, _ = cv2.connectedComponentsWithStats(white_clean, connectivity=8)
            white_max_blob = int(cc_stats_w[1:, cv2.CC_STAT_AREA].max()) if n_lbl_w > 1 else 0

    total_area = yellow_area + white_area
    max_blob   = max(yellow_max_blob, white_max_blob)
    detail = {
        "surface_area":    total_area,
        "yellow_area":     yellow_area,
        "white_area":      white_area,
        "yellow_max_blob": yellow_max_blob,
        "white_max_blob":  white_max_blob,
    }

    # High-confidence cream/white contamination: Good connectors have white_area 0-8px only.
    # Yellow: require both enough total pixels AND a concentrated blob (not scattered housing colour).
    should_fail = (
        (white_area > 80 and white_max_blob >= 50) or
        (total_area > area_threshold and max_blob >= 180)
    )
    if should_fail:
        parts = []
        if yellow_area > 0:
            parts.append(f"yellow={yellow_area}px")
        if white_area > 0:
            parts.append(f"white={white_area}px")
        # Include masks so draw_hybrid_overlay can highlight defect locations
        detail["yellow_mask"] = yellow_clean   # shape (roi_y2-cy1, roi_x2-roi_x1)
        detail["roi_y1"]      = cy1
        detail["roi_x1"]      = roi_x1        # x-offset for overlay projection
        if white_area > 0 and pins and min_pin_y is not None:
            detail["white_mask"] = white_clean
            detail["wb_y1"]      = wb_y1
            detail["wb_x1"]      = wb_x1
        return (
            "FAIL",
            f"surface_defect:area={total_area}px(blob={max_blob}px,{','.join(parts) or 'debris'})",
            detail,
        )
    return "PASS", "OK", detail


def phase4_surface_check(
    bgr: np.ndarray,
    dets: List[dict],
    bright_threshold: int,
    area_threshold: int,
) -> Tuple[str, str]:
    """Returns (verdict, reason_code)."""
    v, r, _ = phase4_compute(bgr, dets, bright_threshold, area_threshold)
    return v, r


# ─────────────────────────────────────────────────────────────────────────────
# VBAI-style Golden Template helpers
# ─────────────────────────────────────────────────────────────────────────────

def _refine_conn_bbox(
    bgr: np.ndarray,
    cx1: int, cy1: int, cx2: int, cy2: int,
) -> Tuple[int, int, int, int]:
    """Refine YOLO connector bbox to the precise housing rectangle.

    Uses Canny edge detection within the YOLO crop to locate the connector
    housing outline, excluding PCB substrate below and background at sides.
    Returns (rx1, ry1, rx2, ry2) in full-image coordinates.
    Falls back to the input bbox when no suitable contour is found.
    """
    roi = bgr[cy1:cy2, cx1:cx2]
    h, w = roi.shape[:2]
    if h < 20 or w < 20:
        return cx1, cy1, cx2, cy2

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 1.5)
    med   = float(np.median(blur))
    lo    = max(10,  int(med * 0.40))
    hi    = min(240, int(med * 1.30))
    edges = cv2.Canny(blur, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cx1, cy1, cx2, cy2

    min_area = h * w * 0.15
    valid = sorted([c for c in cnts if cv2.contourArea(c) >= min_area],
                   key=cv2.contourArea, reverse=True)
    if not valid:
        return cx1, cy1, cx2, cy2

    rect = cv2.minAreaRect(valid[0])
    box  = cv2.boxPoints(rect)
    bx1 = int(max(0,   np.floor(box[:, 0].min())))
    by1 = int(max(0,   np.floor(box[:, 1].min())))
    bx2 = int(min(w-1, np.ceil( box[:, 0].max())))
    by2 = int(min(h-1, np.ceil( box[:, 1].max())))

    if (bx2 - bx1) * (by2 - by1) < h * w * 0.20:
        return cx1, cy1, cx2, cy2      # refined too small — keep YOLO bbox

    return cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2


def _extract_conn_roi_norm(
    bgr: np.ndarray,
    dets: List[dict],
) -> Tuple[Optional[np.ndarray], dict]:
    """Extract and normalize the connector ROI to (TEMPLATE_W × TEMPLATE_H) using YOLO bbox.

    Returns (roi_normalized_bgr, meta_dict).
    meta_dict keys:
      n_pins          – number of pins detected
      pin_rects_norm  – list of [x1,y1,x2,y2] in 0-1 fraction of template dims (+8px margin)
    Returns (None, {}) when no connector detected.
    """
    connectors = [d for d in dets if d["cls_id"] == HYBRID_CLS_CONNECTOR]
    pins       = [d for d in dets if d["cls_id"] == HYBRID_CLS_PIN]
    if not connectors:
        return None, {}

    conn    = max(connectors, key=lambda d: d["w"] * d["h"])
    h_img, w_img = bgr.shape[:2]

    cx1 = max(0, int(conn["cx"] - conn["w"] / 2))
    cy1 = max(0, int(conn["cy"] - conn["h"] / 2))
    cx2 = min(w_img, int(conn["cx"] + conn["w"] / 2))

    if pins:
        max_pin_y = max(int(p["cy"] + p["h"] / 2) for p in pins)
        cy2 = min(h_img, max_pin_y + 100)
    else:
        cy2 = min(h_img, int(conn["cy"] + conn["h"] / 2))

    # Refine YOLO bbox to precise connector housing boundary (excludes PCB foam)
    cx1, cy1, cx2, cy2 = _refine_conn_bbox(bgr, cx1, cy1, cx2, cy2)

    roi_w, roi_h = cx2 - cx1, cy2 - cy1
    if roi_w <= 0 or roi_h <= 0:
        return None, {}

    roi_bgr  = bgr[cy1:cy2, cx1:cx2].copy()
    roi_norm = cv2.resize(roi_bgr, (TEMPLATE_W, TEMPLATE_H))

    # Store pin positions as normalized 0-1 fractions of the ROI (with +8px margin)
    pin_rects_norm = []
    for p in pins:
        exp   = 8
        px1_n = (int(p["cx"] - p["w"] / 2) - cx1 - exp) / roi_w
        py1_n = (int(p["cy"] - p["h"] / 2) - cy1 - exp) / roi_h
        px2_n = (int(p["cx"] + p["w"] / 2) - cx1 + exp) / roi_w
        py2_n = (int(p["cy"] + p["h"] / 2) - cy1 + exp) / roi_h
        pin_rects_norm.append([
            max(0.0, px1_n), max(0.0, py1_n),
            min(1.0, px2_n), min(1.0, py2_n),
        ])

    meta = {
        "n_pins":         len(pins),
        "pin_rects_norm": pin_rects_norm,
        "roi_bounds":     (cx1, cy1, cx2, cy2),  # full-image crop coords for overlay mapping
    }
    return roi_norm, meta


def phase4_golden_compare(
    bgr:           np.ndarray,
    dets:          List[dict],
    template_bgr:  np.ndarray,
    template_meta: dict,
    diff_thr:      int = DEFAULT_SURFACE_DIFF_THR,
    area_thr:      int = DEFAULT_SURFACE_AREA_THR,
) -> Tuple[str, str, dict]:
    """VBAI-style Golden Template Comparison using YOLO-normalised connector ROI.

    Algorithm (mirrors NI VBAI "Detect Defects — Golden Template" step):
      1. Crop connector ROI with the same YOLO-bbox boundaries used when saving the template.
      2. Resize to (TEMPLATE_W × TEMPLATE_H) — registration via consistent YOLO crop.
      3. CLAHE on L-channel to normalise lighting variation before comparing.
      4. Grayscale absolute difference → binary threshold at diff_thr.
      5. Morphological opening 3×3 removes isolated noise pixels.
      6. Mask out pin bboxes (golden pins vary naturally; stored in template_meta).
      7. FAIL if diff_area > area_thr.
    """
    roi_norm, roi_meta = _extract_conn_roi_norm(bgr, dets)
    if roi_norm is None:
        return "PASS", "OK", {"diff_area": 0, "surface_area": 0}

    def _clahe(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    insp_lab  = cv2.cvtColor(_clahe(roi_norm),     cv2.COLOR_BGR2LAB)
    tmpl_lab  = cv2.cvtColor(_clahe(template_bgr), cv2.COLOR_BGR2LAB)
    l_diff    = cv2.absdiff(insp_lab[:, :, 0], tmpl_lab[:, :, 0])
    a_diff    = cv2.absdiff(insp_lab[:, :, 1], tmpl_lab[:, :, 1])
    b_diff    = cv2.absdiff(insp_lab[:, :, 2], tmpl_lab[:, :, 2])
    # Amplify A/B ×2 so yellow contamination (high B-diff ~70-90 → 140-180) exceeds diff_thr=100
    color_diff = np.clip(np.maximum(a_diff, b_diff).astype(np.uint16) * 2, 0, 255).astype(np.uint8)
    diff       = np.maximum(l_diff, color_diff)

    # ── Texture channel: local std-dev comparison (NI Vision "Detect Texture Defects") ──
    # Foam/debris disrupts local surface roughness even when colour shift is small.
    def _local_stddev(gray_f: np.ndarray, k: int = 11) -> np.ndarray:
        kbox = np.ones((k, k), dtype=np.float32) / (k * k)
        mu   = cv2.filter2D(gray_f, -1, kbox)
        mu2  = cv2.filter2D(gray_f ** 2, -1, kbox)
        return np.sqrt(np.maximum(0.0, mu2 - mu ** 2))

    gray_i          = cv2.cvtColor(_clahe(roi_norm),     cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_t          = cv2.cvtColor(_clahe(template_bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex_diff        = cv2.absdiff(_local_stddev(gray_i).astype(np.uint8),
                                  _local_stddev(gray_t).astype(np.uint8))
    tex_diff_scaled = np.clip(tex_diff.astype(np.uint16) * 3, 0, 255).astype(np.uint8)
    diff            = np.maximum(diff, tex_diff_scaled)
    # ── end texture channel ──────────────────────────────────────────────────

    # Build pin mask from normalized template coordinates
    pin_mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=np.uint8)
    for (x1n, y1n, x2n, y2n) in template_meta.get("pin_rects_norm", []):
        px1 = max(0, int(x1n * TEMPLATE_W))
        py1 = max(0, int(y1n * TEMPLATE_H))
        px2 = min(TEMPLATE_W, int(x2n * TEMPLATE_W))
        py2 = min(TEMPLATE_H, int(y2n * TEMPLATE_H))
        if px2 > px1 and py2 > py1:
            pin_mask[py1:py2, px1:px2] = 255
    diff[pin_mask == 255] = 0

    _, diff_bin  = cv2.threshold(diff, int(diff_thr), 255, cv2.THRESH_BINARY)
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_clean   = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel)
    diff_area    = int(np.count_nonzero(diff_clean))

    detail = {"diff_area": diff_area, "surface_area": diff_area}
    if diff_area > area_thr:
        # Include diff mask + ROI bounds for defect location overlay
        detail["diff_mask"] = diff_clean                        # shape (TEMPLATE_H, TEMPLATE_W)
        detail["conn_roi"]  = roi_meta.get("roi_bounds")        # (cx1, cy1, cx2, cy2) full-image
        return "FAIL", f"surface_defect:diff={diff_area}px(template)", detail
    return "PASS", "OK", detail


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid overlay drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_hybrid_overlay(
    bgr: np.ndarray,
    dets: List[dict],
    verdict: str,
    v3: str, r3: str,
    v4: str, r4: str,
    detail3: Optional[dict] = None,
    detail4: Optional[dict] = None,
) -> np.ndarray:
    out = bgr.copy()
    colors = {
        HYBRID_CLS_CONNECTOR: (0, 200, 255),
        HYBRID_CLS_PIN:        (0, 255, 100),
    }

    # Sort pins left→right for consistent P1, P2, P3… numbering
    pins_ordered = sorted(
        [d for d in dets if d["cls_id"] == HYBRID_CLS_PIN],
        key=lambda p: p["cx"],
    )
    pin_number = {id(p): i + 1 for i, p in enumerate(pins_ordered)}

    for d in dets:
        cls_id = d["cls_id"]
        color  = colors.get(cls_id, (200, 200, 200))

        if abs(d.get("ang", 0.0)) > 1e-4:
            pts = rotated_box_points(d["cx"], d["cy"], d["w"], d["h"], d["ang"])
            cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)],
                          True, color, 2)
            lx, ly = int(d["cx"]), max(14, int(d["cy"] - d["h"] / 2) - 4)
        else:
            bx1 = int(d["cx"] - d["w"] / 2)
            by1 = int(d["cy"] - d["h"] / 2)
            bx2 = int(d["cx"] + d["w"] / 2)
            by2 = int(d["cy"] + d["h"] / 2)
            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 2)
            lx, ly = bx1 + 4, max(14, by1 - 4)

        if cls_id == HYBRID_CLS_PIN:
            pnum = pin_number.get(id(d), "?")
            cv2.putText(out, f"P{pnum} {d['score']:.2f}",
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)
        elif cls_id == HYBRID_CLS_CONNECTOR:
            # "Pins: found/expected" inside connector box top-left
            if detail3 is not None:
                cnt = detail3.get("pin_count", "?")
                exp = detail3.get("exp_pins", "?")
                cv2.putText(out, f"Pins: {cnt}/{exp}",
                            (lx, ly + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                name = (HYBRID_CLASSES[cls_id]
                        if 0 <= cls_id < len(HYBRID_CLASSES) else f"cls{cls_id}")
                cv2.putText(out, f"{name} {d['score']:.2f}",
                            (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1, cv2.LINE_AA)

    # Alignment reference line through mean Y of detected pins
    if detail3 is not None:
        ps = detail3.get("pins_sorted", [])
        if len(ps) >= 2:
            mean_y  = int(np.mean([p["cy"] for p in ps]))
            x_left  = int(ps[0]["cx"]  - ps[0]["w"]  / 2)
            x_right = int(ps[-1]["cx"] + ps[-1]["w"] / 2)
            cv2.line(out, (x_left, mean_y), (x_right, mean_y), (255, 255, 0), 1)
            std_y = detail3.get("std_y", 0.0)
            cv2.putText(out, f"std: {std_y:.1f}px",
                        (x_left, mean_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # ── Phase 4 defect location overlay ──────────────────────────────────────
    if v4 == "FAIL" and detail4:
        h_out, w_out = out.shape[:2]
        defect_layer = np.zeros_like(out)

        # Colour mode — yellow channel (connector-width ROI)
        if "yellow_mask" in detail4:
            ry1     = detail4["roi_y1"]
            rx1_off = detail4.get("roi_x1", 0)
            msk     = detail4["yellow_mask"]             # shape (roi_h, roi_w)  values 0/1
            h_m     = min(msk.shape[0], h_out - ry1)
            roi_w_m = msk.shape[1]
            x_end   = min(w_out, rx1_off + roi_w_m)
            if h_m > 0 and x_end > rx1_off:
                roi_view = defect_layer[ry1:ry1 + h_m, rx1_off:x_end]
                roi_view[msk[:h_m, :x_end - rx1_off] == 1] = [0, 0, 255]   # red

        # Colour mode — white/cream channel (cavity ROI window)
        if "white_mask" in detail4:
            wy1 = detail4["wb_y1"]; wx1 = detail4["wb_x1"]
            msk = detail4["white_mask"]
            hm  = min(msk.shape[0], h_out - wy1)
            wm  = min(msk.shape[1], w_out - wx1)
            if hm > 0 and wm > 0:
                roi_view = defect_layer[wy1:wy1 + hm, wx1:wx1 + wm]
                roi_view[msk[:hm, :wm] == 1] = [0, 165, 255]        # orange

        # Golden template mode — resize diff_mask from 640×320 → connector ROI → full image
        if "diff_mask" in detail4 and detail4.get("conn_roi"):
            cx1_r, cy1_r, cx2_r, cy2_r = detail4["conn_roi"]
            rw = max(1, cx2_r - cx1_r); rh = max(1, cy2_r - cy1_r)
            dil_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            dmsk_d = cv2.dilate(detail4["diff_mask"], dil_k)   # grow sparse clusters → visible blobs
            dmsk = cv2.resize(dmsk_d, (rw, rh),
                              interpolation=cv2.INTER_NEAREST)
            rh2 = min(rh, h_out - cy1_r); rw2 = min(rw, w_out - cx1_r)
            if rh2 > 0 and rw2 > 0:
                roi_view = defect_layer[cy1_r:cy1_r + rh2, cx1_r:cx1_r + rw2]
                roi_view[dmsk[:rh2, :rw2] > 0] = [0, 0, 255]        # red

        # Semi-transparent fill: blend defect highlights over the overlay
        mask_any = np.any(defect_layer > 0, axis=2)
        if mask_any.any():
            out = cv2.addWeighted(out, 0.60, defect_layer, 0.40, 0)

            # Draw labelled bounding boxes around each defect cluster
            # Use the blue channel (red=[0,0,255]) and orange channel ([0,165,255]) combined
            combined = ((defect_layer[:, :, 2] > 0)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw * ch > 150:       # ignore tiny noise clusters (< ~12×12 px)
                    cv2.rectangle(out, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
                    cv2.putText(out, "DEFECT", (x, max(14, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Status bar at bottom (taller + bigger font)
    h, w_img = out.shape[:2]
    bar_h = 96
    cv2.rectangle(out, (0, h - bar_h), (w_img, h), (20, 20, 20), -1)
    p3_col = (0, 220, 0) if v3 == "PASS" else (50, 50, 230)
    p4_col = (0, 220, 0) if v4 == "PASS" else (50, 50, 230)
    cv2.putText(out, f"[Ph3-Pin]  {v3}: {r3}", (10, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, p3_col, 2, cv2.LINE_AA)
    cv2.putText(out, f"[Ph4-Surf] {v4}: {r4}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, p4_col, 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# USB Camera wrapper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class UsbCamera:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.index = 0
        self.device_path = ""

    def _ctrl_device(self) -> str:
        return self.device_path if self.device_path else f"/dev/video{int(self.index)}"

    def apply_v4l2_ctrls(self, ctrls: dict, order: list = None,
                          silent: bool = True) -> bool:
        if not ctrls:
            return False
        dev  = self._ctrl_device()
        keys = order[:] if isinstance(order, list) and order else list(ctrls.keys())
        applied = False
        for k in keys:
            if k not in ctrls:
                continue
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", dev, f"--set-ctrl={k}={ctrls[k]}"],
                    stdout=subprocess.DEVNULL if silent else None,
                    stderr=subprocess.DEVNULL if silent else None,
                    check=False,
                )
                applied = True
            except Exception:
                pass
        return applied

    def apply_v4l2_ctl(self, ctrls: dict, order: list = None,
                        silent: bool = True) -> bool:
        return self.apply_v4l2_ctrls(ctrls, order=order, silent=silent)

    def apply_cv2_props_fallback(self, cfg: dict) -> None:
        if self.cap is None or not self.cap.isOpened() or not cfg:
            return
        try:
            if "exposure" in cfg:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, float(cfg["exposure"]))
            if "exposure_time_absolute" in cfg:
                self.cap.set(cv2.CAP_PROP_EXPOSURE,
                             float(cfg["exposure_time_absolute"]))
            if "autofocus" in cfg:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cfg["autofocus"] else 0)
            if "focus" in cfg:
                self.cap.set(cv2.CAP_PROP_FOCUS, float(cfg["focus"]))
            if "auto_exposure" in cfg:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(cfg["auto_exposure"]))
        except Exception:
            pass

    def open(self, index: int, device_path: str = "") -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True
        self.index = int(index)
        self.device_path = device_path or ""
        if self.device_path:
            for api in (cv2.CAP_V4L2, 0):
                cap = cv2.VideoCapture(self.device_path, api)
                if cap.isOpened():
                    self.cap = cap; return True
        for api in (cv2.CAP_V4L2, 0):
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                self.cap = cap; return True
        dev = f"/dev/video{self.index}"
        for api in (cv2.CAP_V4L2, 0):
            cap = cv2.VideoCapture(dev, api)
            if cap.isOpened():
                self.cap = cap; return True
        return False

    def grab(self) -> bool:
        if self.cap is None or not self.cap.isOpened():
            return False
        try:
            return self.cap.grab()
        except Exception:
            return False

    def retrieve(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        try:
            ok, frame = self.cap.retrieve()
            return frame if ok else None
        except Exception:
            return None

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def set_resolution(self, w: int, h: int):
        if self.cap is None or not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    capture_request_signal  = Signal(dict, object)
    # (is_lh, overlay_bgr, verdict, error_str)
    _test_infer_done_signal = Signal(bool, object, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("JetsonDualCam — Hybrid Vision (YOLO + Math + OpenCV)")

        self.cfg = load_config()

        # Local-only storage. Retention runs in its own thread and never uses
        # Network_Backup or changes application configuration.
        self.local_store = LocalRetentionStore(
            root=LOCAL_STORAGE_ROOT,
            max_gb=LOCAL_STORAGE_MAX_GB,
            delete_gb=LOCAL_STORAGE_DELETE_GB,
        )
        self.local_store.start()

        self.lh_cam = UsbCamera()
        self.rh_cam = UsbCamera()

        # Runners (lazy-loaded)
        self.lh_runner = None
        self.rh_runner = None
        self.lh_loaded_path = ""
        self.rh_loaded_path = ""

        # Test image state (Image File Mode)
        self._lh_test_img: Optional[np.ndarray] = None
        self._rh_test_img: Optional[np.ndarray] = None

        # Last inference detail dicts (populated by _infer_hybrid)
        self._lh_detail: dict = {}
        self._rh_detail: dict = {}

        # Last raw (rotated) BGR frame + YOLO dets — used for template saving
        self._lh_last_raw_bgr: Optional[np.ndarray] = None
        self._rh_last_raw_bgr: Optional[np.ndarray] = None
        self._lh_last_dets:    List[dict] = []
        self._rh_last_dets:    List[dict] = []

        # Golden template cache  {key: (template_bgr, meta)}
        # key = "p{n_pins}", e.g. "p12" for a 12-pin connector
        self._templates: dict = {}
        self._template_dir: str = os.path.join(
            os.path.dirname(app_config_path()), "templates"
        )
        self._load_all_templates()
        self._load_pin_config()

        # Preview / capture labels
        self.lh_preview_label = QtWidgets.QLabel("LH Preview")
        self.rh_preview_label = QtWidgets.QLabel("RH Preview")
        self.lh_capture_label = QtWidgets.QLabel("LH Captured")
        self.rh_capture_label = QtWidgets.QLabel("RH Captured")
        for lab in (self.lh_preview_label, self.rh_preview_label,
                    self.lh_capture_label, self.rh_capture_label):
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setMinimumSize(640, 480)
            lab.setStyleSheet("background:#111; color:#ccc;")

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._make_preview_tab(True),  "LH Preview")
        self.tabs.addTab(self._make_preview_tab(False), "RH Preview")
        self.tabs.addTab(self._make_capture_tab(True),  "LH Capture")
        self.tabs.addTab(self._make_capture_tab(False), "RH Capture")
        self.tabs.addTab(self._make_dashboard_tab(),    "Dashboard")
        self.tabs.addTab(self._make_settings_tab(),     "Settings")

        # ── Bottom bar ──────────────────────────────────────────────────────
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom)
        bl.setContentsMargins(8, 5, 8, 5)
        bl.setSpacing(6)

        self.btn_start  = QtWidgets.QPushButton("Start")
        self.btn_stop   = QtWidgets.QPushButton("Stop")
        self.btn_manual = QtWidgets.QPushButton("Manual Trigger")
        self.btn_stop.setEnabled(False)

        # Input source selector (always visible in bottom bar)
        src_frame = QtWidgets.QFrame()
        src_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        src_lay = QtWidgets.QHBoxLayout(src_frame)
        src_lay.setContentsMargins(6, 2, 6, 2)
        src_lay.setSpacing(4)
        src_lay.addWidget(QtWidgets.QLabel("Source:"))
        self.rb_live    = QtWidgets.QRadioButton("Live Camera")
        self.rb_testimg = QtWidgets.QRadioButton("Test Image")
        self.rb_live.setChecked(True)
        src_lay.addWidget(self.rb_live)
        src_lay.addWidget(self.rb_testimg)

        self.btn_load_lh = QtWidgets.QPushButton("LH Image…")
        self.btn_load_rh = QtWidgets.QPushButton("RH Image…")
        self.lbl_lh_img  = QtWidgets.QLabel("(none)")
        self.lbl_rh_img  = QtWidgets.QLabel("(none)")
        self.lbl_lh_img.setMaximumWidth(160)
        self.lbl_rh_img.setMaximumWidth(160)
        for w in (self.btn_load_lh, self.btn_load_rh,
                  self.lbl_lh_img, self.lbl_rh_img):
            w.setVisible(False)
            src_lay.addWidget(w)

        self.lh_pf = QtWidgets.QLabel("LH: -")
        self.rh_pf = QtWidgets.QLabel("RH: -")
        _pf_style = ("font-weight:700; padding:4px 8px; border-radius:6px;"
                     " background:#2b2b2b; color:#e0e0e0;")
        self.lh_pf.setStyleSheet(_pf_style)
        self.rh_pf.setStyleSheet(_pf_style)

        bl.addWidget(self.btn_start)
        bl.addWidget(self.btn_stop)
        bl.addWidget(self.btn_manual)
        bl.addWidget(src_frame)
        bl.addStretch(1)
        bl.addWidget(self.lh_pf)
        bl.addWidget(self.rh_pf)

        dock = QtWidgets.QDockWidget()
        dock.setTitleBarWidget(QtWidgets.QWidget())
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(bottom)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

        # Connections
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_manual.clicked.connect(self.on_manual_trigger)
        self.rb_testimg.toggled.connect(self._on_source_toggled)
        self.btn_load_lh.clicked.connect(lambda: self._load_test_image(True))
        self.btn_load_rh.clicked.connect(lambda: self._load_test_image(False))
        self.capture_request_signal.connect(self._on_web_capture_request)
        self._test_infer_done_signal.connect(self._on_test_infer_done)

        # Timers
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self.update_previews)
        self.gpio_timer = QtCore.QTimer(self)
        self.gpio_timer.timeout.connect(self.poll_gpio)

        self._last_gpio  = None
        self._gpio_chip  = None
        self._gpio_line  = None
        self._web_thread = None
        self._web_should_stop = False
        self._web_running = False

        QtCore.QTimer.singleShot(0, self._apply_cfg_to_widgets)
        # Warmup after config widgets are populated (1 s is enough headroom)
        QtCore.QTimer.singleShot(1000, self._warmup_model)

    # ── UI builders ────────────────────────────────────────────────────────

    def _make_preview_tab(self, is_lh: bool) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        cfg = QtWidgets.QGroupBox("Camera Settings")
        gl  = QtWidgets.QGridLayout(cfg)

        idx   = QtWidgets.QSpinBox(); idx.setRange(0, 20)
        dev   = QtWidgets.QComboBox(); self._populate_video_devices(dev)
        res_w = QtWidgets.QSpinBox(); res_w.setRange(160, 4096); res_w.setValue(1920)
        res_h = QtWidgets.QSpinBox(); res_h.setRange(120, 2160); res_h.setValue(1080)
        btn_open  = QtWidgets.QPushButton("Open Camera")
        btn_close = QtWidgets.QPushButton("Close")
        rot180    = QtWidgets.QCheckBox("Rotate 180°")

        gl.addWidget(QtWidgets.QLabel("Index:"),      0, 0); gl.addWidget(idx, 0, 1)
        gl.addWidget(QtWidgets.QLabel("Device:"),     1, 0); gl.addWidget(dev, 1, 1)
        gl.addWidget(QtWidgets.QLabel("Resolution:"), 2, 0)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(res_w); hl.addWidget(QtWidgets.QLabel("x")); hl.addWidget(res_h)
        gl.addLayout(hl, 2, 1)
        gl.addWidget(rot180,    3, 1)
        gl.addWidget(btn_open,  4, 0); gl.addWidget(btn_close, 4, 1)

        preview_label = self.lh_preview_label if is_lh else self.rh_preview_label
        v.addWidget(cfg)
        v.addWidget(preview_label, 1)

        if is_lh:
            self.lh_idx = idx; self.lh_dev = dev
            self.lh_res_w = res_w; self.lh_res_h = res_h; self.lh_rot = rot180
            btn_open.clicked.connect(lambda: self.open_cam(True))
            btn_close.clicked.connect(lambda: self.close_cam(True))
        else:
            self.rh_idx = idx; self.rh_dev = dev
            self.rh_res_w = res_w; self.rh_res_h = res_h; self.rh_rot = rot180
            btn_open.clicked.connect(lambda: self.open_cam(False))
            btn_close.clicked.connect(lambda: self.close_cam(False))
        return w

    def _make_capture_tab(self, is_lh: bool) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(self.lh_capture_label if is_lh else self.rh_capture_label, 1)
        return w

    def _make_dashboard_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        _vd_style_idle = (
            "font-size:28px; font-weight:bold; border-radius:8px;"
            " background:#2b2b2b; color:#e0e0e0; padding:8px;"
        )
        _det_style = "font-size:13px; padding:4px 0;"

        for side in ("LH", "RH"):
            grp = QtWidgets.QGroupBox(f"{side} Camera")
            col = QtWidgets.QVBoxLayout(grp)
            col.setSpacing(4)

            verdict_lbl = QtWidgets.QLabel("—")
            verdict_lbl.setAlignment(QtCore.Qt.AlignCenter)
            verdict_lbl.setMinimumHeight(64)
            verdict_lbl.setStyleSheet(_vd_style_idle)

            yolo_lbl = QtWidgets.QLabel("YOLO: —")
            p3_lbl   = QtWidgets.QLabel("Phase 3 (Pin Math): —")
            p4_lbl   = QtWidgets.QLabel("Phase 4 (Surface): —")
            for lbl in (yolo_lbl, p3_lbl, p4_lbl):
                lbl.setWordWrap(True)
                lbl.setTextFormat(QtCore.Qt.RichText)
                lbl.setStyleSheet(_det_style)

            thumb = QtWidgets.QLabel("(no capture yet)")
            thumb.setAlignment(QtCore.Qt.AlignCenter)
            thumb.setMinimumSize(320, 240)
            thumb.setStyleSheet("background:#111; color:#666;")

            col.addWidget(verdict_lbl)
            col.addWidget(yolo_lbl)
            col.addWidget(p3_lbl)
            col.addWidget(p4_lbl)
            col.addWidget(thumb, 1)
            root.addWidget(grp, 1)

            if side == "LH":
                self.dash_lh_verdict = verdict_lbl
                self.dash_lh_yolo    = yolo_lbl
                self.dash_lh_p3      = p3_lbl
                self.dash_lh_p4      = p4_lbl
                self.dash_lh_thumb   = thumb
            else:
                self.dash_rh_verdict = verdict_lbl
                self.dash_rh_yolo    = yolo_lbl
                self.dash_rh_p3      = p3_lbl
                self.dash_rh_p4      = p4_lbl
                self.dash_rh_thumb   = thumb

        return w

    def _make_settings_tab(self) -> QtWidgets.QWidget:
        # Wrap everything in a QScrollArea so all settings remain reachable
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)

        # ── Save path ────────────────────────────────────────────────────
        gb_path = QtWidgets.QGroupBox("Save Path")
        hl_path = QtWidgets.QHBoxLayout(gb_path)
        self.save_path = QtWidgets.QLineEdit()
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self.browse_save_path)
        hl_path.addWidget(QtWidgets.QLabel("Folder:")); hl_path.addWidget(self.save_path, 1)
        hl_path.addWidget(btn_browse)

        # ── YOLO model paths ─────────────────────────────────────────────
        gb_yolo = QtWidgets.QGroupBox(
            "YOLO Model  (Hybrid .onnx/.pt preferred; .engine is legacy mode)"
        )
        gl_yolo = QtWidgets.QGridLayout(gb_yolo)
        self.lh_engine = QtWidgets.QLineEdit()
        self.rh_engine = QtWidgets.QLineEdit()
        btn_lh   = QtWidgets.QPushButton("Browse LH…")
        btn_rh   = QtWidgets.QPushButton("Browse RH…")
        btn_auto = QtWidgets.QPushButton("Auto-detect Hybrid Model")
        btn_lh.clicked.connect(lambda: self.browse_engine(True))
        btn_rh.clicked.connect(lambda: self.browse_engine(False))
        btn_auto.clicked.connect(self._auto_detect_model)
        self.conf_pct = QtWidgets.QSpinBox()
        self.conf_pct.setRange(0, 100); self.conf_pct.setSuffix(" %")
        gl_yolo.addWidget(QtWidgets.QLabel("LH model:"), 0, 0)
        gl_yolo.addWidget(self.lh_engine, 0, 1); gl_yolo.addWidget(btn_lh, 0, 2)
        gl_yolo.addWidget(QtWidgets.QLabel("RH model:"), 1, 0)
        gl_yolo.addWidget(self.rh_engine, 1, 1); gl_yolo.addWidget(btn_rh, 1, 2)
        gl_yolo.addWidget(btn_auto, 2, 0, 1, 3)
        gl_yolo.addWidget(QtWidgets.QLabel("Conf threshold:"), 3, 0)
        gl_yolo.addWidget(self.conf_pct, 3, 1)
        gl_yolo.addWidget(
            QtWidgets.QLabel("Hybrid classes: CONNECTOR / PIN  |  Legacy .engine: 4-class"),
            3, 2,
        )

        # ── Hybrid Vision thresholds ─────────────────────────────────────
        gb_hybrid = QtWidgets.QGroupBox(
            "Hybrid Vision — Phase 3 (Pin Math) & Phase 4 (Surface Check)"
        )
        gh = QtWidgets.QGridLayout(gb_hybrid)
        self.expected_pins      = QtWidgets.QSpinBox()
        self.expected_pins.setRange(1, 64)
        self.pin_std_thr        = QtWidgets.QDoubleSpinBox()
        self.pin_std_thr.setRange(0.0, 999.0)
        self.pin_std_thr.setDecimals(1); self.pin_std_thr.setSuffix(" px")
        self.surface_bright_thr = QtWidgets.QSpinBox()
        self.surface_bright_thr.setRange(0, 255)
        self.surface_area_thr   = QtWidgets.QSpinBox()
        self.surface_area_thr.setRange(0, 9999999); self.surface_area_thr.setSuffix(" px")
        self.surface_sat_thr    = QtWidgets.QSpinBox()
        self.surface_sat_thr.setRange(0, 255)

        gh.addWidget(QtWidgets.QLabel("Expected pin count (Phase 3):"),        0, 0)
        gh.addWidget(self.expected_pins,      0, 1)
        gh.addWidget(QtWidgets.QLabel("→ FAIL if detected ≠ expected"),        0, 2)
        gh.addWidget(QtWidgets.QLabel("Pin alignment std-dev limit (Phase 3):"), 1, 0)
        gh.addWidget(self.pin_std_thr,        1, 1)
        gh.addWidget(QtWidgets.QLabel("→ FAIL if std-dev of pin Y > limit"),   1, 2)
        gh.addWidget(QtWidgets.QLabel("Surface min brightness (Phase 4):"),     2, 0)
        gh.addWidget(self.surface_bright_thr, 2, 1)
        gh.addWidget(QtWidgets.QLabel("→ HSV V 0-255; yellow pixel must be brighter than this"), 2, 2)
        gh.addWidget(QtWidgets.QLabel("Surface defect area limit (Phase 4):"), 3, 0)
        gh.addWidget(self.surface_area_thr,   3, 1)
        gh.addWidget(QtWidgets.QLabel("→ FAIL if yellow/orange anomaly pixels > limit"), 3, 2)
        gh.addWidget(QtWidgets.QLabel("Surface min saturation (Phase 4):"),    4, 0)
        gh.addWidget(self.surface_sat_thr,    4, 1)
        gh.addWidget(QtWidgets.QLabel("→ HSV S 0-255; yellow/orange pixel must exceed this"), 4, 2)

        self.skip_phase4 = QtWidgets.QCheckBox("Skip Phase 4 — ไม่ตรวจ Defect (ผ่านอัตโนมัติ)")
        gh.addWidget(self.skip_phase4, 5, 0, 1, 3)

        # ── Golden Template (VBAI-style Phase 4) ─────────────────────────
        gb_tmpl = QtWidgets.QGroupBox(
            "Golden Template — VBAI-style Phase 4  (overrides color detection when template is loaded)"
        )
        gl_tmpl = QtWidgets.QGridLayout(gb_tmpl)

        self._template_status_lbl = QtWidgets.QLabel("…")
        self.surface_diff_thr = QtWidgets.QSpinBox()
        self.surface_diff_thr.setRange(5, 200)
        self.surface_diff_thr.setSuffix("  (pixel diff 0-255)")

        btn_save_lh_tmpl = QtWidgets.QPushButton("Save LH as Template")
        btn_save_rh_tmpl = QtWidgets.QPushButton("Save RH as Template")
        btn_del_tmpl     = QtWidgets.QPushButton("Delete Template…")
        btn_clear_tmpl   = QtWidgets.QPushButton("Clear All Templates")
        btn_save_lh_tmpl.clicked.connect(lambda: self._save_as_template(True))
        btn_save_rh_tmpl.clicked.connect(lambda: self._save_as_template(False))
        btn_del_tmpl.clicked.connect(self._delete_one_template)
        btn_clear_tmpl.clicked.connect(self._clear_templates)

        gl_tmpl.addWidget(QtWidgets.QLabel("Status:"),          0, 0)
        gl_tmpl.addWidget(self._template_status_lbl,            0, 1, 1, 3)
        gl_tmpl.addWidget(QtWidgets.QLabel("Diff threshold:"),  1, 0)
        gl_tmpl.addWidget(self.surface_diff_thr,                1, 1)
        gl_tmpl.addWidget(
            QtWidgets.QLabel("→ lower = more sensitive; raise if too many false alarms"), 1, 2
        )
        gl_tmpl.addWidget(btn_save_lh_tmpl, 2, 0)
        gl_tmpl.addWidget(btn_save_rh_tmpl, 2, 1)
        gl_tmpl.addWidget(btn_del_tmpl,     2, 2)
        gl_tmpl.addWidget(btn_clear_tmpl,   2, 3)

        self._refresh_template_status()  # show templates already loaded at startup

        _how = QtWidgets.QLabel(
            "How to use:  1) Load a Good connector image → analyze it → come here → Save as Template.\n"
            "             2) Template is keyed by pin count (e.g. p12 for 12-pin connectors).\n"
            "             3) When a template is found for the connector type, it REPLACES color detection."
        )
        _how.setStyleSheet("color:#aaa; font-size:10px;")
        gl_tmpl.addWidget(_how, 3, 0, 1, 3)

        # ── Trigger mode ─────────────────────────────────────────────────
        gb_trig = QtWidgets.QGroupBox("Trigger Mode")
        hl_trig = QtWidgets.QHBoxLayout(gb_trig)
        self.rb_gpio   = QtWidgets.QRadioButton("GPIO I/O")
        self.rb_manual = QtWidgets.QRadioButton("Manual button")
        self.rb_web    = QtWidgets.QRadioButton("Web service")
        hl_trig.addWidget(self.rb_gpio)
        hl_trig.addWidget(self.rb_manual)
        hl_trig.addWidget(self.rb_web)
        hl_trig.addStretch(1)

        # ── GPIO ─────────────────────────────────────────────────────────
        gb_gpio = QtWidgets.QGroupBox("GPIO Trigger (Jetson / libgpiod)")
        gl2 = QtWidgets.QGridLayout(gb_gpio)
        self.gpio_chip_name   = QtWidgets.QLineEdit()
        self.gpio_chip_name.setPlaceholderText("e.g. gpiochip0")
        self.gpio_line_offset = QtWidgets.QSpinBox(); self.gpio_line_offset.setRange(0, 512)
        self.gpio_poll_ms     = QtWidgets.QSpinBox()
        self.gpio_poll_ms.setRange(1, 1000); self.gpio_poll_ms.setSuffix(" ms")
        self.edge_rising      = QtWidgets.QRadioButton("Rising edge")
        self.edge_falling     = QtWidgets.QRadioButton("Falling edge")
        self.edge_rising.setChecked(True)
        self.trigger_delay_ms = QtWidgets.QSpinBox()
        self.trigger_delay_ms.setRange(0, 10000); self.trigger_delay_ms.setSuffix(" ms")
        self.gpio_status      = QtWidgets.QLabel("Unknown")
        gl2.addWidget(QtWidgets.QLabel("Chip:"),          0, 0); gl2.addWidget(self.gpio_chip_name,   0, 1)
        gl2.addWidget(QtWidgets.QLabel("Line offset:"),   1, 0); gl2.addWidget(self.gpio_line_offset, 1, 1)
        gl2.addWidget(QtWidgets.QLabel("Poll interval:"), 2, 0); gl2.addWidget(self.gpio_poll_ms,     2, 1)
        hle = QtWidgets.QHBoxLayout()
        hle.addWidget(self.edge_rising); hle.addWidget(self.edge_falling); hle.addStretch(1)
        gl2.addWidget(QtWidgets.QLabel("Edge:"),          3, 0); gl2.addLayout(hle,                   3, 1)
        gl2.addWidget(QtWidgets.QLabel("Trigger delay:"), 4, 0); gl2.addWidget(self.trigger_delay_ms, 4, 1)
        gl2.addWidget(QtWidgets.QLabel("Status:"),        5, 0); gl2.addWidget(self.gpio_status,      5, 1)

        # ── Web service ───────────────────────────────────────────────────
        gb_web = QtWidgets.QGroupBox("Web Service Settings")
        glw = QtWidgets.QGridLayout(gb_web)
        self.web_host   = QtWidgets.QLineEdit(); self.web_host.setPlaceholderText("0.0.0.0")
        self.web_port   = QtWidgets.QSpinBox()
        self.web_port.setRange(1, 65535); self.web_port.setValue(9000)
        self.web_status = QtWidgets.QLabel("Stopped")
        glw.addWidget(QtWidgets.QLabel("Host:"),   0, 0); glw.addWidget(self.web_host,   0, 1)
        glw.addWidget(QtWidgets.QLabel("Port:"),   1, 0); glw.addWidget(self.web_port,   1, 1)
        glw.addWidget(QtWidgets.QLabel("Status:"), 2, 0); glw.addWidget(self.web_status, 2, 1)

        # ── Test / Mechanics ─────────────────────────────────────────────
        gb_test = QtWidgets.QGroupBox("Test / Mechanics")
        vl_test = QtWidgets.QVBoxLayout(gb_test)
        self.force_pass = QtWidgets.QCheckBox(
            "Force PASS (bypass Fail) — keep analysis/overlay, just report as Pass"
        )
        vl_test.addWidget(self.force_pass)
        gl_test = QtWidgets.QGridLayout()
        self.capture_settle_ms      = QtWidgets.QSpinBox()
        self.capture_settle_ms.setRange(0, 5000); self.capture_settle_ms.setSuffix(" ms")
        self.capture_discard_frames = QtWidgets.QSpinBox()
        self.capture_discard_frames.setRange(0, 10)
        gl_test.addWidget(QtWidgets.QLabel("Capture settle delay:"), 0, 0)
        gl_test.addWidget(self.capture_settle_ms,      0, 1)
        gl_test.addWidget(QtWidgets.QLabel("Discard frames:"),        1, 0)
        gl_test.addWidget(self.capture_discard_frames, 1, 1)
        gl_test.addWidget(
            QtWidgets.QLabel("(Tip: settle 150–400 ms, discard 1–3 frames)"), 2, 0, 1, 2
        )
        vl_test.addLayout(gl_test)

        btn_save = QtWidgets.QPushButton("Save Config")
        btn_save.clicked.connect(self.on_save_config)

        v.addWidget(gb_path)
        v.addWidget(gb_yolo)
        v.addWidget(gb_hybrid)
        v.addWidget(gb_tmpl)
        v.addWidget(gb_trig)
        v.addWidget(gb_gpio)
        v.addWidget(gb_web)
        v.addWidget(gb_test)
        v.addWidget(btn_save)
        v.addStretch(1)

        if gpiod is None:
            gb_gpio.setEnabled(False); self.rb_gpio.setEnabled(False)
        if FastAPI is None or uvicorn is None:
            self.rb_web.setEnabled(False)
            self.web_status.setText("FastAPI/uvicorn not installed")

        self.force_pass.setChecked(bool(self.cfg.get("force_pass", False)))
        mode0 = self.cfg.get("trigger_mode", "manual")
        if mode0 == "gpio" and gpiod is not None:
            self.rb_gpio.setChecked(True)
        elif mode0 == "web" and (FastAPI is not None and uvicorn is not None):
            self.rb_web.setChecked(True)
        else:
            self.rb_manual.setChecked(True)

        scroll.setWidget(inner)
        return scroll

    # ── Helpers ────────────────────────────────────────────────────────────

    def _populate_video_devices(self, combo: QtWidgets.QComboBox):
        combo.clear()
        combo.addItem("(auto)", "")
        for i in range(20):
            p = f"/dev/video{i}"
            if os.path.exists(p):
                combo.addItem(p, p)

    def _apply_cfg_to_widgets(self):
        self.save_path.setText(
            self.cfg.get("save_path", os.path.join(os.path.expanduser("~"), "captures"))
        )
        # Auto-fill model paths from train-7 if not saved yet
        auto = find_model_file()
        self.lh_engine.setText(self.cfg.get("lh_engine", "") or auto)
        self.rh_engine.setText(self.cfg.get("rh_engine", "") or auto)

        self.conf_pct.setValue(int(self.cfg.get("conf_pct", 50)))
        self.lh_idx.setValue(int(self.cfg.get("lh_cam_index", 0)))
        self.rh_idx.setValue(int(self.cfg.get("rh_cam_index", 1)))
        self._set_combo_by_data(self.lh_dev, self.cfg.get("lh_cam_device", ""))
        self._set_combo_by_data(self.rh_dev, self.cfg.get("rh_cam_device", ""))
        self.lh_res_w.setValue(int(self.cfg.get("lh_res_w", 1920)))
        self.lh_res_h.setValue(int(self.cfg.get("lh_res_h", 1080)))
        self.rh_res_w.setValue(int(self.cfg.get("rh_res_w", 1920)))
        self.rh_res_h.setValue(int(self.cfg.get("rh_res_h", 1080)))
        self.lh_rot.setChecked(bool(self.cfg.get("lh_rot180", False)))
        self.rh_rot.setChecked(bool(self.cfg.get("rh_rot180", False)))
        self.gpio_chip_name.setText(self.cfg.get("gpio_chip", "gpiochip0"))
        self.web_host.setText(self.cfg.get("web_host", "0.0.0.0"))
        self.web_port.setValue(int(self.cfg.get("web_port", 9000)))
        self.gpio_line_offset.setValue(int(self.cfg.get("gpio_line", 40)))
        self.gpio_poll_ms.setValue(int(self.cfg.get("gpio_poll_ms", 10)))
        self.trigger_delay_ms.setValue(int(self.cfg.get("trigger_delay_ms", 0)))
        self.capture_settle_ms.setValue(int(self.cfg.get("capture_settle_ms", 0)))
        self.capture_discard_frames.setValue(
            int(self.cfg.get("capture_discard_frames", 0))
        )
        edge = self.cfg.get("gpio_edge", "rising")
        self.edge_rising.setChecked(edge == "rising")
        self.edge_falling.setChecked(edge == "falling")
        # Hybrid params
        self.expected_pins.setValue(
            int(self.cfg.get("expected_pins", DEFAULT_EXPECTED_PINS))
        )
        self.pin_std_thr.setValue(
            float(self.cfg.get("pin_std_threshold", DEFAULT_PIN_STD_THRESHOLD))
        )
        self.surface_bright_thr.setValue(
            int(self.cfg.get("surface_bright_thr", DEFAULT_SURFACE_BRIGHT_THR))
        )
        self.surface_area_thr.setValue(
            int(self.cfg.get("surface_area_thr", DEFAULT_SURFACE_AREA_THR))
        )
        self.surface_sat_thr.setValue(
            int(self.cfg.get("surface_sat_thr", DEFAULT_SURFACE_SAT_THR))
        )
        self.surface_diff_thr.setValue(
            int(self.cfg.get("surface_diff_thr", DEFAULT_SURFACE_DIFF_THR))
        )
        self.skip_phase4.setChecked(bool(self.cfg.get("skip_phase4", False)))
        self._refresh_template_status()
        # Input source
        src = self.cfg.get("input_source", "live")
        if src == "image_file":
            self.rb_testimg.setChecked(True)
        else:
            self.rb_live.setChecked(True)

    @staticmethod
    def _set_combo_by_data(combo: QtWidgets.QComboBox, data_value: str):
        for i in range(combo.count()):
            if combo.itemData(i) == data_value:
                combo.setCurrentIndex(i); return
        combo.setCurrentIndex(0)

    # ── Input source ───────────────────────────────────────────────────────

    def _on_source_toggled(self, checked: bool):
        test = self.rb_testimg.isChecked()
        for w in (self.btn_load_lh, self.btn_load_rh,
                  self.lbl_lh_img, self.lbl_rh_img):
            w.setVisible(test)

    def _load_test_image(self, is_lh: bool):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Test Image", os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if not f:
            return
        img = cv2.imread(f)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Load Image", f"Cannot read:\n{f}")
            return
        name = os.path.basename(f)
        if is_lh:
            self._lh_test_img = img
            self.lbl_lh_img.setText(name)
            self._show_on_label(self.lh_preview_label, img)
        else:
            self._rh_test_img = img
            self.lbl_rh_img.setText(name)
            self._show_on_label(self.rh_preview_label, img)

        # Auto-run inference in a background thread so the UI stays responsive
        model_path = (self.lh_engine if is_lh else self.rh_engine).text().strip()
        if not model_path:
            return  # no model configured yet — just show preview

        # Disable both load buttons while running to prevent concurrent calls
        self.btn_load_lh.setEnabled(False)
        self.btn_load_rh.setEnabled(False)
        lbl = self.lbl_lh_img if is_lh else self.lbl_rh_img
        lbl.setText(f"{name}  [analyzing…]")

        bgr = img.copy()
        rot = self.lh_rot if is_lh else self.rh_rot
        if rot.isChecked():
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)

        # Read ALL widget values here on the main thread — safe to do now
        conf = max(0.0, min(1.0, float(self.conf_pct.value()) / 100.0))
        fp   = bool(self.force_pass.isChecked())
        hp   = {
            "exp_pins":    self._pins_from_sn(name) or int(self.expected_pins.value()),
            "std_thr":     float(self.pin_std_thr.value()),
            "br_thr":      int(self.surface_bright_thr.value()),
            "area_thr":    int(self.surface_area_thr.value()),
            "sat_thr":     int(self.surface_sat_thr.value()),
            "diff_thr":    int(self.surface_diff_thr.value()),
            "skip_phase4": bool(self.skip_phase4.isChecked()),
            "templates":   dict(self._templates),  # snapshot — thread-safe read-only copy
        }

        def _worker():
            try:
                overlay, verdict, _ = self._infer_one(
                    bgr, is_lh, model_path, conf, fp, hybrid_params=hp
                )
                self._test_infer_done_signal.emit(is_lh, overlay, verdict, "")
            except Exception as exc:
                import traceback as _tb
                print(f"[_worker] inference error: {exc}")
                _tb.print_exc()
                self._test_infer_done_signal.emit(is_lh, None, "", str(exc))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    @QtCore.Slot(bool, object, str, str)
    def _on_test_infer_done(self, is_lh: bool, overlay, verdict: str, error: str):
        """Called on the main thread when background test-image inference finishes."""
        self.btn_load_lh.setEnabled(True)
        self.btn_load_rh.setEnabled(True)
        name_lbl = self.lbl_lh_img if is_lh else self.lbl_rh_img
        # Restore clean filename (strip the "[analyzing…]" suffix)
        name_lbl.setText(name_lbl.text().replace("  [analyzing…]", ""))

        if error:
            QtWidgets.QMessageBox.warning(
                self, "Inference Error", f"Could not run inference:\n{error}"
            )
            return

        fp          = bool(self.force_pass.isChecked())
        capture_lbl = self.lh_capture_label if is_lh else self.rh_capture_label
        preview_lbl = self.lh_preview_label if is_lh else self.rh_preview_label
        pf_lbl      = self.lh_pf            if is_lh else self.rh_pf
        side        = "LH"                  if is_lh else "RH"
        if overlay is not None:
            self._show_on_label(capture_lbl, overlay)
            self._show_on_label(preview_lbl, overlay)
        # Auto-switch to the Capture tab so user sees the result immediately
        self.tabs.setCurrentIndex(2 if is_lh else 3)
        self._set_pf(pf_lbl, side,
                     "PASS (BYPASS)" if (fp and verdict.upper() != "PASS") else verdict)
        self._refresh_dashboard()

    # ── Model auto-detect button ───────────────────────────────────────────

    def _auto_detect_model(self):
        p = find_model_file()
        if p:
            self.lh_engine.setText(p)
            self.rh_engine.setText(p)
            self.lh_loaded_path = ""; self.lh_runner = None
            self.rh_loaded_path = ""; self.rh_runner = None
            QtWidgets.QMessageBox.information(
                self, "Auto-detect", f"Model found:\n{p}"
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Auto-detect", f"No model found in:\n{WEIGHTS_DIR}"
            )

    # ── Camera control ─────────────────────────────────────────────────────

    def open_cam(self, is_lh: bool):
        cam = self.lh_cam if is_lh else self.rh_cam
        idx = self.lh_idx.value() if is_lh else self.rh_idx.value()
        dev = (self.lh_dev.currentData() if is_lh else self.rh_dev.currentData()) or ""
        if not cam.open(idx, dev):
            QtWidgets.QMessageBox.warning(
                self, "Camera", f"Open camera failed ({'LH' if is_lh else 'RH'})"
            )
            return
        w = self.lh_res_w.value() if is_lh else self.rh_res_w.value()
        h = self.lh_res_h.value() if is_lh else self.rh_res_h.value()
        cam.set_resolution(w, h)
        try:
            prof = self.cfg.get("cam_ctrl_profile", {}) or {}
            if prof.get("enabled", True):
                ctrls = dict(prof.get("ctrls", {}) or {})
                override = self.cfg.get("lh_cam_ctrl" if is_lh else "rh_cam_ctrl", {}) or {}
                if isinstance(override, dict):
                    ctrls.update(override)
                if not cam.apply_v4l2_ctrls(ctrls, order=prof.get("order")):
                    cam.apply_cv2_props_fallback(ctrls)
        except Exception:
            pass
        if not self.preview_timer.isActive():
            self.preview_timer.start(30)

    def close_cam(self, is_lh: bool):
        (self.lh_cam if is_lh else self.rh_cam).close()

    def update_previews(self):
        self._update_one_preview(True)
        self._update_one_preview(False)

    def _update_one_preview(self, is_lh: bool):
        if self.rb_testimg.isChecked():
            return  # test images are shown immediately on load
        cam   = self.lh_cam if is_lh else self.rh_cam
        frame = cam.read()
        if frame is None:
            return
        if (self.lh_rot if is_lh else self.rh_rot).isChecked():
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        lab = self.lh_preview_label if is_lh else self.rh_preview_label
        self._show_on_label(lab, frame)

    # ── Trigger ────────────────────────────────────────────────────────────

    def on_start(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.on_save_config(silent=True)
        if not self.preview_timer.isActive():
            self.preview_timer.start(30)

        mode = "manual"
        if gpiod is not None and self.rb_gpio.isChecked():
            mode = "gpio"
        elif (FastAPI is not None and uvicorn is not None) and self.rb_web.isChecked():
            mode = "web"

        if mode == "gpio":
            if not self._start_gpio():
                QtWidgets.QMessageBox.warning(
                    self, "GPIO", "GPIO init failed. Switching to manual."
                )
                self.rb_manual.setChecked(True)
        elif mode == "web":
            if not self._start_web_service():
                QtWidgets.QMessageBox.warning(
                    self, "Web", "Web service start failed. Switching to manual."
                )
                self.rb_manual.setChecked(True)
        else:
            try:
                self.gpio_status.setText("Manual")
            except Exception:
                pass

    def on_stop(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._stop_gpio()
        self._stop_web_service()

    def on_manual_trigger(self):
        self.perform_capture_and_infer()

    # ── GPIO ───────────────────────────────────────────────────────────────

    def _start_gpio(self) -> bool:
        if gpiod is None:
            return False
        self._stop_gpio(); self._stop_web_service()
        chip_name   = self.gpio_chip_name.text().strip() or "gpiochip0"
        line_offset = int(self.gpio_line_offset.value())
        try:
            self._gpio_chip = gpiod.Chip(chip_name)
            self._gpio_line = self._gpio_chip.get_line(line_offset)
            self._gpio_line.request(consumer=APP_NAME, type=gpiod.LINE_REQ_DIR_IN)
            self._last_gpio = self._gpio_line.get_value()
            self.gpio_status.setText(f"{chip_name}:{line_offset} = {self._last_gpio}")
        except Exception as e:
            self.gpio_status.setText(f"GPIO error: {e}")
            self._stop_gpio(); return False
        self.gpio_timer.start(int(self.gpio_poll_ms.value()))
        return True

    def _stop_gpio(self):
        if self.gpio_timer.isActive():
            self.gpio_timer.stop()
        for obj in (self._gpio_line, self._gpio_chip):
            if obj is not None:
                try:
                    obj.release() if hasattr(obj, "release") else obj.close()
                except Exception:
                    pass
        self._gpio_line = self._gpio_chip = self._last_gpio = None

    def poll_gpio(self):
        if self._gpio_line is None:
            return
        try:
            val = int(self._gpio_line.get_value())
        except Exception:
            return
        self.gpio_status.setText(str(val))
        if self._last_gpio is None:
            self._last_gpio = val; return
        rising  = self.edge_rising.isChecked()  and self._last_gpio == 0 and val == 1
        falling = self.edge_falling.isChecked() and self._last_gpio == 1 and val == 0
        self._last_gpio = val
        if rising or falling:
            delay_ms = int(self.trigger_delay_ms.value())
            if delay_ms > 0:
                QtCore.QTimer.singleShot(delay_ms, self.perform_capture_and_infer)
            else:
                self.perform_capture_and_infer()

    # ── Web service (FastAPI / uvicorn async future) ────────────────────────

    def _start_web_service(self) -> bool:
        if FastAPI is None or uvicorn is None:
            try:
                self.web_status.setText("FastAPI/uvicorn not installed")
            except Exception:
                pass
            return False
        if self._web_running:
            return True

        host = (self.web_host.text().strip() if hasattr(self, "web_host") else "0.0.0.0") or "0.0.0.0"
        port = int(self.web_port.value()) if hasattr(self, "web_port") else 9000
        self._web_should_stop = False

        def run_server():
            try:
                app = FastAPI(title="JetsonDualCamPy Web Trigger", version="1.0")

                @app.get("/health")
                def health():
                    return {"ok": True}

                @app.post("/api/eval")
                async def api_capture(payload: dict):
                    if not isinstance(payload, dict):
                        raise HTTPException(status_code=400,
                                            detail="JSON root must be an object")
                    loop = asyncio.get_running_loop()
                    fut  = loop.create_future()
                    self.capture_request_signal.emit(payload, {"loop": loop, "future": fut})
                    try:
                        resp = await asyncio.wait_for(fut, timeout=15.0)
                    except asyncio.TimeoutError:
                        raise HTTPException(status_code=504,
                                            detail="Timeout waiting for capture")
                    except Exception as exc:
                        raise HTTPException(status_code=500, detail=str(exc))
                    return JSONResponse(content=resp)

                self._web_running = True
                try:
                    self.web_status.setText(f"Running: http://{host}:{port}")
                except Exception:
                    pass
                uvicorn.run(app, host=host, port=port, log_level="warning")
            except Exception as e:
                self._web_running = False
                try:
                    self.web_status.setText(f"Web error: {e}")
                except Exception:
                    pass

        import threading
        self._web_thread = threading.Thread(target=run_server, daemon=True)
        self._web_thread.start()
        return True

    def _stop_web_service(self):
        try:
            if hasattr(self, "web_status") and self._web_running:
                self.web_status.setText("Running (stop requires app restart)")
        except Exception:
            pass

    @QtCore.Slot(dict, object)
    def _on_web_capture_request(self, payload: dict, reply_holder: object):
        loop = fut = None
        if isinstance(reply_holder, dict):
            loop = reply_holder.get("loop")
            fut  = reply_holder.get("future")
        try:
            resp = self.perform_capture_and_infer_payload(payload)
            if loop is not None and fut is not None:
                loop.call_soon_threadsafe(fut.set_result, resp)
            else:
                reply_holder["resp"] = resp; reply_holder["err"] = None
        except Exception as e:
            if loop is not None and fut is not None:
                loop.call_soon_threadsafe(fut.set_exception, e)
            else:
                reply_holder["resp"] = None; reply_holder["err"] = str(e)
            try:
                print("Web capture error:", e); traceback.print_exc()
            except Exception:
                pass
        finally:
            if isinstance(reply_holder, dict):
                reply_holder["done"] = True

    # ── File-path helpers ──────────────────────────────────────────────────

    def _mk_save_paths(
        self, t: time.struct_time, base_path: str = DEFAULT_LOCAL_SAVE_BASE
    ) -> Tuple[str, str]:
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        day_dir = os.path.join(
            base_path,
            f"Year_{t.tm_year}",
            f"{month_names[t.tm_mon-1]}_{t.tm_mon:02d}",
            f"Day_{t.tm_mday}",
        )
        lh_dir = os.path.join(day_dir, "LH_CAM")
        rh_dir = os.path.join(day_dir, "RH_CAM")
        os.makedirs(lh_dir, exist_ok=True)
        os.makedirs(rh_dir, exist_ok=True)
        return lh_dir, rh_dir

    def _buffer_image(self, relative_path: str, img: np.ndarray) -> None:
        if img is None or img.size == 0:
            raise ValueError(f"Cannot save empty image: {relative_path}")
        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            raise IOError(f"Failed to encode image: {relative_path}")
        self.local_store.create_png(relative_path, encoded.tobytes())

    def _make_filename(
        self, verdict: str, sn: str, ts: str, suffix: str = ""
    ) -> str:
        sn      = (sn      or "NOSN").strip()
        verdict = (verdict or "UNK" ).strip()
        suf     = f"_{suffix}" if suffix else ""
        return f"{verdict}_{sn}_{ts}{suf}.png"

    def _save_capture_images(
        self,
        t: time.struct_time,
        sn1: str, sn2: str,
        lh_pf: str, rh_pf: str,
        lh: np.ndarray, rh: np.ndarray,
        lh_overlay: np.ndarray, rh_overlay: np.ndarray,
    ) -> Tuple[str, str]:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", t)

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        day_rel = os.path.join(
            f"Year_{t.tm_year}", f"{month_names[t.tm_mon-1]}_{t.tm_mon:02d}",
            f"Day_{t.tm_mday}",
        )
        lh_rel = os.path.join(day_rel, "LH_CAM")
        rh_rel = os.path.join(day_rel, "RH_CAM")
        self._buffer_image(os.path.join(lh_rel, self._make_filename(lh_pf, sn1, ts, "raw")), lh)
        self._buffer_image(os.path.join(rh_rel, self._make_filename(rh_pf, sn2, ts, "raw")), rh)
        self._buffer_image(os.path.join(lh_rel, self._make_filename(lh_pf, sn1, ts, "overlay")), lh_overlay)
        self._buffer_image(os.path.join(rh_rel, self._make_filename(rh_pf, sn2, ts, "overlay")), rh_overlay)
        return str(self.local_store.root / lh_rel), str(self.local_store.root / rh_rel)

    # ── Capture ────────────────────────────────────────────────────────────

    def _capture_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (lh, rh). Uses test images when in Image File Mode."""
        if self.rb_testimg.isChecked():
            if self._lh_test_img is None or self._rh_test_img is None:
                raise RuntimeError(
                    "Test Image Mode: please load both LH and RH images first."
                )
            return self._lh_test_img.copy(), self._rh_test_img.copy()

        settle_ms = 0
        discard   = 0
        try:
            settle_ms = int(self.capture_settle_ms.value())
        except Exception:
            settle_ms = int(self.cfg.get("capture_settle_ms", 0))
        try:
            discard = int(self.capture_discard_frames.value())
        except Exception:
            discard = int(self.cfg.get("capture_discard_frames", 0))

        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)
        for _ in range(max(0, min(20, discard))):
            self.lh_cam.grab(); self.rh_cam.grab()
        if not self.lh_cam.grab() or not self.rh_cam.grab():
            return None, None
        return self.lh_cam.retrieve(), self.rh_cam.retrieve()

    # ── Runner factory ─────────────────────────────────────────────────────

    def _warmup_model(self):
        """Load model + run one dummy inference in a background thread.

        Called once ~1 s after startup so the first real inference is instant
        (avoids ultralytics init + ONNX session creation on the critical path).
        """
        model_path = self.lh_engine.text().strip() or self.rh_engine.text().strip()
        if not model_path or not os.path.isfile(model_path):
            return

        def _run():
            try:
                print("[warmup] loading model…")
                runner = self._create_runner(model_path)
                dummy  = np.zeros((360, 640, 3), dtype=np.uint8)
                runner.infer(dummy, conf=0.25)
                # Cache so _ensure_runner skips creation on first real call
                if self.lh_runner is None:
                    self.lh_runner      = runner
                    self.lh_loaded_path = model_path
                if self.rh_runner is None:
                    self.rh_runner      = runner
                    self.rh_loaded_path = model_path
                print("[warmup] model ready — first inference will be fast")
            except Exception as exc:
                print(f"[warmup] failed (non-fatal): {exc}")

        import threading
        threading.Thread(target=_run, daemon=True, name="model-warmup").start()

    def _ensure_runner(self, is_lh: bool, model_path: str):
        model_path = model_path.strip()
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path!r}")
        if is_lh:
            if self.lh_runner is None or self.lh_loaded_path != model_path:
                self.lh_runner      = self._create_runner(model_path)
                self.lh_loaded_path = model_path
        else:
            if self.rh_runner is None or self.rh_loaded_path != model_path:
                self.rh_runner      = self._create_runner(model_path)
                self.rh_loaded_path = model_path

    @staticmethod
    def _create_runner(model_path: str):
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".engine":
            return TrtRunner(model_path)
        if ext == ".onnx":
            return OnnxRunner(model_path)
        return YoloRunner(model_path)

    # ── Inference ──────────────────────────────────────────────────────────

    def _infer_one(
        self,
        bgr: np.ndarray,
        is_lh: bool,
        model_path: str,
        conf: float,
        force_pass: bool = False,
        hybrid_params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, str, List[str]]:
        """Returns (overlay_bgr, verdict, symptoms_list).

        hybrid_params: pre-read widget values for thread-safe calls.
            Keys: exp_pins, std_thr, br_thr, area_thr
        """
        self._ensure_runner(is_lh, model_path)
        runner = self.lh_runner if is_lh else self.rh_runner
        ext    = os.path.splitext(model_path)[1].lower()
        if ext == ".engine":
            return self._infer_legacy(bgr, runner, conf, force_pass)
        return self._infer_hybrid(bgr, runner, conf, force_pass,
                                  is_lh=is_lh, hybrid_params=hybrid_params)

    def _infer_legacy(
        self,
        bgr: np.ndarray,
        runner: "TrtRunner",
        conf: float,
        force_pass: bool,
    ) -> Tuple[np.ndarray, str, List[str]]:
        _, _, in_h, in_w = runner.in_shape
        lb, meta   = letterbox_bgr(bgr, (in_h, in_w))
        inp        = bgr_to_chw_float32(lb)
        outs       = runner.infer(inp)
        dets_m, bs = best_per_class_raw_head(outs[0])
        dets       = [scale_back_det(d, meta) for d in dets_m]
        overlay    = draw_overlay_legacy(bgr, dets, bs)
        verdict    = verdict_pass_fail_from_scores(bs)

        syms: List[str] = []
        if bs.size >= 4:
            sc0, sc1, sc2, sc3 = [float(x) for x in bs[:4]]
            if sc1 < THRESH_PASS_CONNECTOR: syms.append("connector_low")
            if sc0 >= THRESH_PASS_OTHERS:   syms.append("broken_connector")
            if sc2 >= THRESH_PASS_OTHERS:   syms.append("pin_missing")
            if sc3 >= THRESH_PASS_OTHERS:   syms.append("scratch")

        self._draw_verdict_banner(overlay, verdict, force_pass)
        return overlay, verdict, syms

    def _infer_hybrid(
        self,
        bgr: np.ndarray,
        runner: "YoloRunner",
        conf: float,
        force_pass: bool,
        is_lh: bool = True,
        hybrid_params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, str, List[str]]:
        # Accept pre-read params (for thread-safe calls) or read from widgets
        if hybrid_params:
            exp_pins    = hybrid_params["exp_pins"]
            std_thr     = hybrid_params["std_thr"]
            br_thr      = hybrid_params["br_thr"]
            area_thr    = hybrid_params["area_thr"]
            sat_thr     = hybrid_params.get("sat_thr",  DEFAULT_SURFACE_SAT_THR)
            diff_thr    = hybrid_params.get("diff_thr", DEFAULT_SURFACE_DIFF_THR)
            skip_phase4 = hybrid_params.get("skip_phase4", False)
            templates   = hybrid_params.get("templates", {})
        else:
            exp_pins    = int(self.expected_pins.value())
            std_thr     = float(self.pin_std_thr.value())
            br_thr      = int(self.surface_bright_thr.value())
            area_thr    = int(self.surface_area_thr.value())
            sat_thr     = int(self.surface_sat_thr.value())
            diff_thr    = int(self.surface_diff_thr.value())
            skip_phase4 = bool(self.skip_phase4.isChecked())
            templates   = self._templates

        dets = runner.infer(bgr, conf=conf)

        # Cache last raw frame + dets for template saving (thread-safe: set before phase compute)
        if is_lh:
            self._lh_last_raw_bgr = bgr
            self._lh_last_dets    = dets
        else:
            self._rh_last_raw_bgr = bgr
            self._rh_last_dets    = dets

        v3, r3, d3 = phase3_compute(dets, exp_pins, std_thr)

        # Phase 4: surface defect check (skipped when skip_phase4 is enabled)
        n_pins   = len([d for d in dets if d["cls_id"] == HYBRID_CLS_PIN])
        tmpl_key = f"p{n_pins}"
        if skip_phase4:
            v4, r4, d4 = "PASS", "SKIP", {"surface_area": 0}
        elif tmpl_key in templates:
            tmpl_bgr, tmpl_meta = templates[tmpl_key]
            v4, r4, d4 = phase4_golden_compare(bgr, dets, tmpl_bgr, tmpl_meta, diff_thr, area_thr)
        else:
            v4, r4, d4 = phase4_compute(bgr, dets, br_thr, area_thr, sat_thr)

        syms: List[str] = []
        if v3 == "FAIL": syms.append(r3)
        if v4 == "FAIL": syms.append(r4)

        verdict = "PASS" if (v3 == "PASS" and v4 == "PASS") else "FAIL"
        overlay = draw_hybrid_overlay(bgr, dets, verdict, v3, r3, v4, r4, detail3=d3, detail4=d4)
        self._draw_verdict_banner(overlay, verdict, force_pass)

        detail = {
            "verdict":         "PASS (BYPASS)" if (force_pass and verdict != "PASS") else verdict,
            "yolo_connectors": len([d for d in dets if d["cls_id"] == HYBRID_CLS_CONNECTOR]),
            "yolo_pins":       len([d for d in dets if d["cls_id"] == HYBRID_CLS_PIN]),
            "v3": v3, "r3": r3,
            "v4": v4, "r4": r4,
            "pin_count":   d3.get("pin_count", 0),
            "exp_pins":    d3.get("exp_pins",  exp_pins),
            "std_y":       d3.get("std_y",     0.0),
            "surface_area": d4.get("surface_area", 0),
            "overlay":     overlay.copy(),
        }
        if is_lh:
            self._lh_detail = detail
        else:
            self._rh_detail = detail

        return overlay, verdict, syms

    @staticmethod
    def _draw_verdict_banner(overlay: np.ndarray, verdict: str, force_pass: bool):
        if force_pass and verdict != "PASS":
            cv2.putText(overlay, "PASS (BYPASS)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, "REAL: FAIL", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            color = (0, 255, 0) if verdict == "PASS" else (0, 0, 255)
            cv2.putText(overlay, verdict, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

    # ── Symptoms helper ────────────────────────────────────────────────────

    def _symptoms_from_scores(self, syms_extra: object, sn: str) -> str:
        """Combine SN-based failure flag with inference symptom list."""
        out: List[str] = []
        sn_s = (sn or "").strip()
        if (not sn_s) or (sn_s.upper() == "NA"):
            out.append("dm_read_fail")
        if isinstance(syms_extra, list):
            out.extend(syms_extra)
        elif isinstance(syms_extra, np.ndarray):
            if syms_extra.size >= 4:
                sc0, sc1, sc2, sc3 = [float(x) for x in syms_extra[:4]]
                if sc1 < THRESH_PASS_CONNECTOR: out.append("connector_low")
                if sc0 >= THRESH_PASS_OTHERS:   out.append("broken_connector_high")
                if sc2 >= THRESH_PASS_OTHERS:   out.append("pin_missing_high")
                if sc3 >= THRESH_PASS_OTHERS:   out.append("scratch_high")
            else:
                out.append("infer_no_scores")
        return ":".join(out) if out else "NA"

    # ── High-level capture+infer entry points ──────────────────────────────

    def perform_capture_and_infer(self):
        try:
            lh, rh = self._capture_pair()
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self, "Capture", str(e)); return
        if lh is None or rh is None:
            QtWidgets.QMessageBox.warning(
                self, "Capture",
                "Capture failed.\n(Live mode: open both cameras first)"
            )
            return

        if self.lh_rot.isChecked(): lh = cv2.rotate(lh, cv2.ROTATE_180)
        if self.rh_rot.isChecked(): rh = cv2.rotate(rh, cv2.ROTATE_180)

        t    = time.localtime()
        conf = max(0.0, min(1.0, float(self.conf_pct.value()) / 100.0))
        fp   = bool(self.force_pass.isChecked())

        lh_overlay, lh_v, _ = self._infer_one(lh, True,  self.lh_engine.text().strip(), conf, fp)
        rh_overlay, rh_v, _ = self._infer_one(rh, False, self.rh_engine.text().strip(), conf, fp)

        lh_pf = "Pass" if (("PASS" if fp else lh_v.upper()) == "PASS") else "Fail"
        rh_pf = "Pass" if (("PASS" if fp else rh_v.upper()) == "PASS") else "Fail"

        self._save_capture_images(t, "NA", "NA", lh_pf, rh_pf,
                                  lh, rh, lh_overlay, rh_overlay)
        self._show_on_label(self.lh_capture_label, lh_overlay)
        self._show_on_label(self.rh_capture_label, rh_overlay)
        self._set_pf(self.lh_pf, "LH",
                     "PASS (BYPASS)" if (fp and lh_v.upper() != "PASS") else lh_v)
        self._set_pf(self.rh_pf, "RH",
                     "PASS (BYPASS)" if (fp and rh_v.upper() != "PASS") else rh_v)
        self._refresh_dashboard()

    def perform_capture_and_infer_payload(self, payload: dict) -> dict:
        """
        Payload: {"data1":{"SN":"...","result":""},"data2":{"SN":"...","result":""}}
        Returns same dict with result filled: "Pass"/"Fail" + failure symptoms.
        """
        if not isinstance(payload, dict):
            raise ValueError("payload must be dict")

        data1 = payload.get("data1") if isinstance(payload.get("data1"), dict) else {}
        data2 = payload.get("data2") if isinstance(payload.get("data2"), dict) else {}
        sn1   = str(data1.get("SN", "")).strip()
        sn2   = str(data2.get("SN", "")).strip()

        try:
            lh, rh = self._capture_pair()
        except RuntimeError as e:
            raise RuntimeError(str(e))
        if lh is None or rh is None:
            raise RuntimeError("Capture failed (open both cameras first)")

        if self.lh_rot.isChecked(): lh = cv2.rotate(lh, cv2.ROTATE_180)
        if self.rh_rot.isChecked(): rh = cv2.rotate(rh, cv2.ROTATE_180)

        t    = time.localtime()
        conf = max(0.0, min(1.0, float(self.conf_pct.value()) / 100.0))
        fp   = bool(self.force_pass.isChecked())
        default_pins = int(self.expected_pins.value())
        hp_base = {
            "std_thr":     float(self.pin_std_thr.value()),
            "br_thr":      int(self.surface_bright_thr.value()),
            "area_thr":    int(self.surface_area_thr.value()),
            "sat_thr":     int(self.surface_sat_thr.value()),
            "diff_thr":    int(self.surface_diff_thr.value()),
            "skip_phase4": bool(self.skip_phase4.isChecked()),
            "templates":   dict(self._templates),
        }

        lh_overlay, lh_v, lh_syms = self._infer_one(
            lh, True,  self.lh_engine.text().strip(), conf, fp,
            hybrid_params={**hp_base, "exp_pins": self._pins_from_sn(sn1) or default_pins},
        )
        rh_overlay, rh_v, rh_syms = self._infer_one(
            rh, False, self.rh_engine.text().strip(), conf, fp,
            hybrid_params={**hp_base, "exp_pins": self._pins_from_sn(sn2) or default_pins},
        )

        lh_pf = "Pass" if (("PASS" if fp else lh_v.upper()) == "PASS") else "Fail"
        rh_pf = "Pass" if (("PASS" if fp else rh_v.upper()) == "PASS") else "Fail"

        self._save_capture_images(t, sn1, sn2, lh_pf, rh_pf,
                                  lh, rh, lh_overlay, rh_overlay)
        self._show_on_label(self.lh_capture_label, lh_overlay)
        self._show_on_label(self.rh_capture_label, rh_overlay)
        self._set_pf(self.lh_pf, "LH",
                     "PASS (BYPASS)" if (fp and lh_v.upper() != "PASS") else lh_v)
        self._set_pf(self.rh_pf, "RH",
                     "PASS (BYPASS)" if (fp and rh_v.upper() != "PASS") else rh_v)
        self._refresh_dashboard()

        data1["result"]           = lh_pf
        data2["result"]           = rh_pf
        data1["failure symptoms"] = self._symptoms_from_scores(lh_syms, sn1)
        data2["failure symptoms"] = self._symptoms_from_scores(rh_syms, sn2)

        for k, v in list(payload.items()):
            if k.startswith("data") and k not in ("data1", "data2") and isinstance(v, dict):
                v.setdefault("result", "NA")
                v.setdefault("failure symptoms", "NA")
                payload[k] = v
        payload["data1"] = data1
        payload["data2"] = data2
        return payload

    # ── UI helpers ─────────────────────────────────────────────────────────

    def _refresh_dashboard(self):
        """Update Dashboard tab labels from the last _lh_detail / _rh_detail."""
        _PASS_STYLE = (
            "font-size:28px; font-weight:bold; border-radius:8px;"
            " background:#0A7D2A; color:white; padding:8px;"
        )
        _FAIL_STYLE = (
            "font-size:28px; font-weight:bold; border-radius:8px;"
            " background:#B00020; color:white; padding:8px;"
        )
        _IDLE_STYLE = (
            "font-size:28px; font-weight:bold; border-radius:8px;"
            " background:#2b2b2b; color:#e0e0e0; padding:8px;"
        )
        for det, verd_lbl, yolo_lbl, p3_lbl, p4_lbl, thumb_lbl in (
            (self._lh_detail,
             self.dash_lh_verdict, self.dash_lh_yolo,
             self.dash_lh_p3, self.dash_lh_p4, self.dash_lh_thumb),
            (self._rh_detail,
             self.dash_rh_verdict, self.dash_rh_yolo,
             self.dash_rh_p3, self.dash_rh_p4, self.dash_rh_thumb),
        ):
            if not det:
                continue

            v = det.get("verdict", "—")
            v_up = v.upper()
            if "PASS" in v_up:
                verd_lbl.setText(v)
                verd_lbl.setStyleSheet(_PASS_STYLE)
            elif "FAIL" in v_up:
                verd_lbl.setText(v)
                verd_lbl.setStyleSheet(_FAIL_STYLE)
            else:
                verd_lbl.setText(v)
                verd_lbl.setStyleSheet(_IDLE_STYLE)

            yolo_lbl.setText(
                f"YOLO: {det.get('yolo_connectors', 0)} connector(s), "
                f"{det.get('yolo_pins', 0)} pin(s) detected"
            )

            pin_cnt  = det.get("pin_count", 0)
            exp_cnt  = det.get("exp_pins",  0)
            std_y    = det.get("std_y",     0.0)
            v3, r3   = det.get("v3", "—"), det.get("r3", "—")
            p3_color = "#00CC44" if v3 == "PASS" else "#FF4444"
            p3_lbl.setText(
                f"<b style='color:{p3_color};'>Phase 3 (Pin)  {v3}</b>"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Count: {pin_cnt}/{exp_cnt}"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Alignment std-dev: {std_y:.1f} px"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"{r3}"
            )

            surf   = det.get("surface_area", 0)
            v4, r4 = det.get("v4", "—"), det.get("r4", "—")
            p4_color = "#00CC44" if v4 == "PASS" else "#FF4444"
            p4_lbl.setText(
                f"<b style='color:{p4_color};'>Phase 4 (Surface)  {v4}</b>"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Defect area: {surf} px"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"{r4}"
            )

            overlay = det.get("overlay")
            if overlay is not None:
                self._show_on_label(thumb_lbl, overlay)

    def _show_on_label(self, label: QtWidgets.QLabel, bgr: np.ndarray):
        qimg = qimage_from_bgr(bgr)
        pix  = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(
            pix.scaled(label.size(), QtCore.Qt.KeepAspectRatio,
                       QtCore.Qt.SmoothTransformation)
        )

    def _set_pf(self, lab: QtWidgets.QLabel, name: str, verdict: str):
        v = (verdict or "").upper()
        if v.startswith("PASS"):
            suffix = " (BYPASS)" if "BYPASS" in v else ""
            lab.setText(f"{name}: PASS{suffix}")
            lab.setStyleSheet(
                "font-weight:700; padding:4px 8px; border-radius:6px;"
                " background:#0A7D2A; color:white;"
            )
        elif v.startswith("FAIL"):
            lab.setText(f"{name}: FAIL")
            lab.setStyleSheet(
                "font-weight:700; padding:4px 8px; border-radius:6px;"
                " background:#B00020; color:white;"
            )
        else:
            lab.setText(f"{name}: -")
            lab.setStyleSheet(
                "font-weight:700; padding:4px 8px; border-radius:6px;"
                " background:#2b2b2b; color:#e0e0e0;"
            )

    # ── Settings actions ───────────────────────────────────────────────────

    def browse_save_path(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder", os.path.expanduser("~")
        )
        if d:
            self.save_path.setText(d)

    def browse_engine(self, is_lh: bool):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model File", os.path.expanduser("~"),
            "Model Files (*.engine *.onnx *.pt);;All files (*)",
        )
        if not f:
            return
        if is_lh:
            self.lh_engine.setText(f)
            self.lh_loaded_path = ""; self.lh_runner = None
        else:
            self.rh_engine.setText(f)
            self.rh_loaded_path = ""; self.rh_runner = None

    def on_save_config(self, silent: bool = False):
        self.cfg = {
            "save_path":              self.save_path.text().strip(),
            "lh_engine":              self.lh_engine.text().strip(),
            "rh_engine":              self.rh_engine.text().strip(),
            "conf_pct":               int(self.conf_pct.value()),
            "force_pass":             bool(self.force_pass.isChecked()),
            "lh_cam_index":           int(self.lh_idx.value()),
            "rh_cam_index":           int(self.rh_idx.value()),
            "lh_cam_device":          self.lh_dev.currentData() or "",
            "rh_cam_device":          self.rh_dev.currentData() or "",
            "lh_res_w":               int(self.lh_res_w.value()),
            "lh_res_h":               int(self.lh_res_h.value()),
            "rh_res_w":               int(self.rh_res_w.value()),
            "rh_res_h":               int(self.rh_res_h.value()),
            "lh_rot180":              bool(self.lh_rot.isChecked()),
            "rh_rot180":              bool(self.rh_rot.isChecked()),
            "trigger_mode": (
                "gpio" if (gpiod is not None and self.rb_gpio.isChecked())
                else ("web" if (FastAPI is not None and uvicorn is not None
                                and self.rb_web.isChecked())
                      else "manual")
            ),
            "gpio_chip":              self.gpio_chip_name.text().strip(),
            "gpio_line":              int(self.gpio_line_offset.value()),
            "gpio_poll_ms":           int(self.gpio_poll_ms.value()),
            "gpio_edge":              "rising" if self.edge_rising.isChecked() else "falling",
            "trigger_delay_ms":       int(self.trigger_delay_ms.value()),
            "capture_settle_ms":      int(self.capture_settle_ms.value()),
            "capture_discard_frames": int(self.capture_discard_frames.value()),
            "web_host":               self.web_host.text().strip(),
            "web_port":               int(self.web_port.value()),
            # Hybrid Vision
            "expected_pins":          int(self.expected_pins.value()),
            "pin_std_threshold":      float(self.pin_std_thr.value()),
            "surface_bright_thr":     int(self.surface_bright_thr.value()),
            "surface_area_thr":       int(self.surface_area_thr.value()),
            "surface_sat_thr":        int(self.surface_sat_thr.value()),
            "surface_diff_thr":       int(self.surface_diff_thr.value()),
            "skip_phase4":            bool(self.skip_phase4.isChecked()),
            # Input source
            "input_source": "image_file" if self.rb_testimg.isChecked() else "live",
        }
        save_config(self.cfg)
        if not silent:
            QtWidgets.QMessageBox.information(self, "Config", "Saved.")

    # ── Golden Template management ─────────────────────────────────────────

    def _load_all_templates(self):
        """Load all .npz template files from the template directory into self._templates."""
        self._templates.clear()
        if not os.path.isdir(self._template_dir):
            return
        for f in glob.glob(os.path.join(self._template_dir, "*.npz")):
            key = os.path.splitext(os.path.basename(f))[0]
            try:
                data = np.load(f, allow_pickle=True)
                tmpl = data["template"]
                meta = data["meta"][0]
                if hasattr(meta, "item") and not isinstance(meta, dict):
                    meta = meta.item()
                if not isinstance(meta, dict):
                    raise ValueError("template metadata is not a dictionary")
                self._templates[key] = (tmpl, meta)
                print(f"[template] loaded: {key}  n_pins={meta.get('n_pins')}  shape={tmpl.shape}")
            except Exception as exc:
                print(f"[template] load failed {f}: {exc}")

    # ── PeaPLED_Config.csv  (code → n_pins) ───────────────────────────────────

    def _load_pin_config(self):
        """Load PeaPLED_Config.csv: 2-digit code -> pin count."""
        self._pin_config: dict = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, PIN_CONFIG_FILENAME),
            os.path.join(os.path.dirname(script_dir), PIN_CONFIG_FILENAME),
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            print(f"[pin_config] {PIN_CONFIG_FILENAME} not found — using widget default")
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                code = row[0].strip().zfill(2)
                m = re.search(r'\d+', row[1])
                if m:
                    self._pin_config[code] = int(m.group())
        print(f"[pin_config] loaded {len(self._pin_config)} entries from {path}")

    def _pins_from_sn(self, sn: str) -> Optional[int]:
        """Return expected pin count from SN or filename by matching PE + 2 digits."""
        m = re.search(r'PE(\d{2})', sn, re.IGNORECASE)
        if m:
            return self._pin_config.get(m.group(1))
        return None

    def _refresh_template_status(self):
        if not hasattr(self, "_template_status_lbl"):
            return
        if self._templates:
            parts = [
                f"{k} ({v[1].get('n_pins','?')} pins, {v[1].get('n_averaged',1)} img avg)"
                for k, v in sorted(self._templates.items())
            ]
            self._template_status_lbl.setText("ACTIVE: " + ", ".join(parts))
            self._template_status_lbl.setStyleSheet("color: #00CC44; font-weight: bold;")
        else:
            self._template_status_lbl.setText(
                "No templates — colour detection active (load a Good image, analyze it, then Save as Template)"
            )
            self._template_status_lbl.setStyleSheet("color: #FF8844;")

    def _save_as_template(self, is_lh: bool):
        """Save the last analyzed frame as the golden template for its connector type."""
        bgr  = self._lh_last_raw_bgr if is_lh else self._rh_last_raw_bgr
        dets = self._lh_last_dets    if is_lh else self._rh_last_dets
        side = "LH" if is_lh else "RH"

        if bgr is None:
            QtWidgets.QMessageBox.warning(
                self, "Save Template",
                f"No {side} image has been analyzed yet.\n"
                "Load a Good connector image and analyze it first."
            )
            return

        roi_norm, meta = _extract_conn_roi_norm(bgr, dets)
        if roi_norm is None:
            QtWidgets.QMessageBox.warning(
                self, "Save Template",
                "No connector detected in the last analyzed image.\n"
                "Make sure YOLO finds a connector before saving a template."
            )
            return

        n_pins = meta["n_pins"]
        key    = f"p{n_pins}"
        os.makedirs(self._template_dir, exist_ok=True)
        path = os.path.join(self._template_dir, f"{key}.npz")

        # Incremental averaging: blend new image with existing template so that
        # natural specimen-to-specimen variation is averaged out over time.
        # Formula: new_avg = (old_avg * n + new_img) / (n+1)
        if key in self._templates:
            existing_tmpl, existing_meta = self._templates[key]
            n = int(existing_meta.get("n_averaged", 1))
            blended  = (existing_tmpl.astype(np.float32) * n +
                        roi_norm.astype(np.float32)) / (n + 1)
            roi_norm = blended.astype(np.uint8)
            meta["n_averaged"] = n + 1
            action_msg = f"Blended with {n} existing image(s)  →  now averaged over {n+1} images."
        else:
            meta["n_averaged"] = 1
            action_msg = "First template image saved."

        np.savez(path, template=roi_norm, meta=np.array([meta]))
        self._templates[key] = (roi_norm, meta)
        self._refresh_template_status()

        QtWidgets.QMessageBox.information(
            self, "Save Template",
            f"Template updated!\n"
            f"  Connector type : {key}  ({n_pins} pins)\n"
            f"  {action_msg}\n"
            f"  Saved to       : {path}\n\n"
            f"Tip: save 3-5 different Good connectors for a robust averaged template."
        )

    def _delete_one_template(self):
        if not self._templates:
            QtWidgets.QMessageBox.information(self, "Delete Template", "No templates saved yet.")
            return
        items = []
        for k, (_, m) in sorted(self._templates.items()):
            n_avg = m.get("n_averaged", 1)
            n_pins = m.get("n_pins", "?")
            items.append(f"{k}  ({n_pins} pins, averaged {n_avg} image(s))")
        key_labels = sorted(self._templates.keys())
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Delete Template", "Select template to delete:", items, 0, False
        )
        if not ok:
            return
        idx   = items.index(choice)
        key   = key_labels[idx]
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Template",
            f"Delete template  '{key}'?\n"
            "Phase 4 will fall back to colour detection for this connector type.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        path = os.path.join(self._template_dir, f"{key}.npz")
        try:
            os.remove(path)
        except Exception:
            pass
        self._templates.pop(key, None)
        self._refresh_template_status()
        QtWidgets.QMessageBox.information(self, "Delete Template", f"Template '{key}' deleted.")

    def _clear_templates(self):
        if not self._templates:
            QtWidgets.QMessageBox.information(self, "Clear Templates", "No templates to clear.")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Clear Templates",
            f"Delete all {len(self._templates)} saved template(s)?\n"
            "Phase 4 will fall back to colour detection.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        for f in glob.glob(os.path.join(self._template_dir, "*.npz")):
            try:
                os.remove(f)
            except Exception:
                pass
        self._templates.clear()
        self._refresh_template_status()
        QtWidgets.QMessageBox.information(self, "Clear Templates", "All templates deleted.")

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.on_save_config(silent=True)
        except Exception:
            pass
        self.preview_timer.stop()
        self._stop_gpio()
        self._stop_web_service()
        self.lh_cam.close()
        self.rh_cam.close()
        self.local_store.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("PeaPLED")
    app.setApplicationName("JetsonDualCam")
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(_QT_EXEC(app))


if __name__ == "__main__":
    main()
