"""Loader for Stray Scanner captures (the LiDAR tier of our stock-capture route).

Folder layout written by the app:
    camera_matrix.csv   3x3 intrinsics for the 1920x1440 RGB stream
    odometry.csv        per-frame ARKit pose (camera->world, xyz + quaternion)
    depth/NNNNNN.png    uint16 depth in millimetres, 256x192
    confidence/NNNNNN.png  ARKit confidence 0/1/2
    rgb.mp4             RGB video
    imu.csv             accelerometer + gyro (unused by the geometry path)

Conventions: world is gravity-aligned with +Y up (ARKit). Despite ARKit cameras looking down -Z,
the app exports poses for an OpenCV camera, so back-projected points (x right,
y down, z forward) are transformed by the pose directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

RGB_W, RGB_H = 1920, 1440
# Stray Scanner stores poses for an OpenCV-convention camera (x right, y down,
# z forward); verified empirically: this choice gives the sharpest fused map
# (39k occupied 5 cm voxels vs 147k with the ARKit flip, see notes/decisions.md).
CV_TO_ARKIT = np.eye(3)


@dataclass
class Frame:
    index: int
    timestamp: float
    T_wc: np.ndarray          # 4x4 camera->world, OpenCV camera axes, +Y-up world
    K_rgb: np.ndarray         # 3x3 intrinsics at RGB resolution
    depth_path: Path | None
    conf_path: Path | None


@dataclass
class LidarCapture:
    root: Path
    frames: list[Frame]

    @property
    def depth_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.depth_path is not None]

    def load_depth(self, f: Frame, min_conf: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Return (depth_m, K_depth) with low-confidence pixels set to 0."""
        d = cv2.imread(str(f.depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if f.conf_path is not None and f.conf_path.exists():
            c = cv2.imread(str(f.conf_path), cv2.IMREAD_UNCHANGED)
            d[c < min_conf] = 0.0
        h, w = d.shape
        s = np.array([w / RGB_W, h / RGB_H, 1.0])[:, None]
        return d, f.K_rgb * s

    def backproject(self, f: Frame, stride: int = 2, min_conf: int = 2,
                    max_range: float = 5.0) -> np.ndarray:
        """World-frame points (N,3) from one depth frame."""
        d, K = self.load_depth(f, min_conf)
        v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[v, u]
        ok = (z > 0.1) & (z < max_range)
        u, v, z = u[ok], v[ok], z[ok]
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        p_cam = (CV_TO_ARKIT @ np.stack([x, y, z]))
        R, t = f.T_wc[:3, :3], f.T_wc[:3, 3:]
        return (R @ p_cam + t).T


def _pose(row) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = row
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def load(root: str | Path) -> LidarCapture:
    root = Path(root)
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    od = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1)
    frames = []
    for r in od:
        idx = int(r[1])
        name = f"{idx:06d}.png"
        dp, cp = root / "depth" / name, root / "confidence" / name
        Kf = K.copy()
        if not np.isnan(r[9]):     # per-frame intrinsics when the app logs them
            Kf = np.array([[r[9], 0, r[11]], [0, r[10], r[12]], [0, 0, 1.0]])
        frames.append(Frame(idx, float(r[0]), _pose(r[2:9]), Kf,
                            dp if dp.exists() else None, cp if cp.exists() else None))
    return LidarCapture(root, frames)


def is_stray_capture(root: str | Path) -> bool:
    root = Path(root)
    return (root / "odometry.csv").exists() and (root / "depth").is_dir()
