"""Tier dispatch: capture -> OrientedCloud -> plan -> damage/scope -> result dict."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from roomscan.geometry.levels import manhattan_yaw, yaw_matrix
from roomscan.plan import build_plan


def _lidar_cloud(capture: Path, drift: bool, timing: dict, info: dict):
    from roomscan.io import strayscanner as ss
    from roomscan.tiers.lidar import build_cloud
    cap = ss.load(capture)
    t = time.perf_counter()
    poses = None
    info["drift"] = {"method": "none (poses as-is)", "enabled": False,
                     "correction_m": None, "loop_closures": 0}
    if drift:
        from roomscan.drift import correct_drift
        poses, info["drift"] = correct_drift(cap)
        timing["drift"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    c = build_cloud(cap, every=3, poses=poses)
    timing["fusion"] = round(time.perf_counter() - t, 2)
    P = np.array([f.T_wc[:3, 3] for f in cap.frames])
    path = float(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1).sum())
    info["n_inputs"] = len(cap.depth_frames)
    return c, path


def run(capture: Path, tier: str, drift: bool = True) -> dict:
    timing, info, warnings = {}, {}, []
    if tier == "lidar":
        c, path = _lidar_cloud(capture, drift, timing, info)
    else:
        raise NotImplementedError(f"{tier} tier not wired yet")
    c = c.transformed(yaw_matrix(manhattan_yaw(c)))
    t = time.perf_counter()
    # drift scales with walked path only when poses are used uncorrected
    plan = build_plan(c, tier, path_length=0.0 if info["drift"]["enabled"] else path)
    timing["plan"] = round(time.perf_counter() - t, 2)
    for r in plan["rooms"]:
        if r["ceiling_height"] is None:
            warnings.append(f"{r['id']}: ceiling not observed; sweep the phone over the ceiling")
    if not plan["calibrated"]:
        warnings.append("intervals use prior error model (no benchmark calibration found)")
    return {"schema_version": "1.0",
            "capture": {"id": capture.name, "path": str(capture), "device": None,
                        "n_inputs": info.get("n_inputs", 0)},
            **plan, "damage": [], "concealed_flags": [], "scope": [],
            "drift": info["drift"], "timing_s": timing, "warnings": warnings}
