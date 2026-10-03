"""Monocular metric geometry for the photo and video tiers (MoGe-2, MIT licence).

infer(rgb) -> camera-frame point map (OpenCV axes, metres), normals, validity
mask and intrinsics. Results are cached on disk keyed by a hash of the input
pixels + model name, so re-runs replay deterministically; delete
out/cache to force the live path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "out" / "cache" / "moge"
WEIGHTS = {"small": ROOT / "weights" / "small" / "model.pt",
           "large": ROOT / "weights" / "large" / "model.pt"}
MODEL_SIZE = "large"
RES_LEVEL = 0          # MoGe token budget 0..9; 0 = fastest (see notes D13)
SCALE_CORR = 1.0       # global multiplicative fix for MoGe's indoor metric bias,
                       # fitted against LiDAR on the sample captures (notes D14).
                       # Applied after the cache, so changing it needs no re-inference.
INFER_LONG_SIDE = 640

_model = None


@dataclass
class MonoView:
    points: np.ndarray     # (H,W,3) camera frame, OpenCV axes, metres
    normals: np.ndarray    # (H,W,3) camera frame, towards camera
    mask: np.ndarray       # (H,W) bool
    K: np.ndarray          # 3x3 at this (H,W)
    rgb: np.ndarray        # (H,W,3) uint8


def _load():
    global _model
    if _model is None:
        import torch
        from moge.model.v2 import MoGeModel
        w = WEIGHTS[MODEL_SIZE]
        if not w.exists():
            raise SystemExit(f"missing {w}: run  python scripts/fetch_weights.py --large")
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4)))
        _model = MoGeModel.from_pretrained(str(w)).eval()
    return _model


def resize_long(rgb: np.ndarray, long_side: int = INFER_LONG_SIDE) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def infer(rgb: np.ndarray, fov_x_deg: float | None = None) -> MonoView:
    """fov_x_deg: known horizontal field of view (from EXIF); conditions MoGe's
    geometry instead of letting it guess the lens (see notes D14)."""
    rgb = resize_long(rgb)
    fov_tag = f"-fov{fov_x_deg:.2f}" if fov_x_deg else ""
    key = hashlib.sha1(rgb.tobytes() + f"{MODEL_SIZE}-L{RES_LEVEL}{fov_tag}-v1".encode()).hexdigest()
    f = CACHE / f"{key}.npz"
    if f.exists():
        d = np.load(f)
        return MonoView(d["points"] * SCALE_CORR, d["normals"], d["mask"], d["K"], rgb)
    import torch
    m = _load()
    with torch.no_grad():
        o = m.infer(torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1),
                    resolution_level=RES_LEVEL, fov_x=fov_x_deg, use_fp16=False)
    h, w = rgb.shape[:2]
    K = o["intrinsics"].numpy().copy()
    K[0] *= w
    K[1] *= h
    pts = o["points"].numpy().astype(np.float32)
    nrm = o["normal"].numpy().astype(np.float32) if "normal" in o else _normals(pts)
    # MoGe normals: make sure they face the camera
    flip = np.sum(nrm * pts, -1) > 0
    nrm[flip] *= -1
    mask = o["mask"].numpy().astype(bool) & np.isfinite(pts).all(-1)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, points=pts, normals=nrm, mask=mask, K=K)
    return MonoView(pts * SCALE_CORR, nrm, mask, K, rgb)


def _normals(P: np.ndarray) -> np.ndarray:
    dx = np.gradient(P, axis=1)
    dy = np.gradient(P, axis=0)
    n = np.cross(dx, dy)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def gravity_up(v: MonoView, iters: int = 5) -> np.ndarray:
    """Up direction in camera coords from the floor/ceiling normals.

    Starts from the image's "up" (-Y for an upright photo) and refines by
    averaging normals within 25 deg of +-up, with walls (normals ~perpendicular
    to up) adding an orthogonality constraint via least squares.
    """
    n = v.normals[v.mask]
    n = n[np.linalg.norm(n, axis=1) > 0.5]
    # Image "up" (-Y) is the starting hypothesis: iPhone photos/videos are stored
    # upright via EXIF/rotation metadata. A Manhattan room can't disambiguate a
    # 90-degree-rotated image on its own (all three axes look alike), so a
    # sideways image without metadata is a documented failure mode.
    u = _refine_up(n, np.array([0.0, -1.0, 0.0]), iters)
    # floor sits below the camera: more upward-facing normals below than above
    P = v.points[v.mask][np.linalg.norm(v.normals[v.mask], axis=1) > 0.5]
    up_facing = (n @ u) > 0.95
    if up_facing.any() and np.median(P[up_facing] @ u) > 0:
        u = -u
    return u


def _refine_up(n: np.ndarray, u: np.ndarray, iters: int) -> np.ndarray:
    for _ in range(iters):
        c = n @ u
        horiz = n[np.abs(c) > np.cos(np.deg2rad(25))] * np.sign(c[np.abs(c) > np.cos(np.deg2rad(25))])[:, None]
        wall = n[np.abs(c) < np.sin(np.deg2rad(15))]
        # minimise sum |h x u|^2 + sum (w . u)^2  -> smallest eigvec of M
        M = (wall.T @ wall) / max(len(wall), 1) - (horiz.T @ horiz) / max(len(horiz), 1)
        w_, V = np.linalg.eigh(M)
        u_new = V[:, 0]
        if u_new @ u < 0:
            u_new = -u_new
        u = u_new
    return u
