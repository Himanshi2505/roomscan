"""Convert one ARKitScenes raw recording to the Stray Scanner folder layout.

    python tools/arkitscenes_to_stray.py data/public/arkitscenes/42445021 \
        data/public/stray/42445021 [--invert] [--flip]

ARKitScenes raw: lowres_depth/<vid>_<ts>.png (uint16 mm, 256x192),
confidence/<vid>_<ts>.png, lowres_wide/<vid>_<ts>.png (RGB 256x192),
lowres_wide_intrinsics/<vid>_<ts>.pincam ("w h fx fy cx cy"),
lowres_wide.traj ("ts rx ry rz tx ty tz", axis-angle + translation).

The pose convention (camera->world or world->camera; ARKit or OpenCV camera
axes) is chosen with --invert/--flip after testing all four on map crispness,
exactly as for Stray Scanner (notes D2, D19).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

SCALE = 7.5          # 256x192 -> the 1920x1440 frame the Stray loader assumes (both 4:3)


def _ts(p: Path) -> float:
    return float(p.stem.rsplit("_", 1)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--invert", action="store_true", help="traj is world->camera")
    ap.add_argument("--flip", action="store_true", help="traj camera has ARKit axes (y up, -z fwd)")
    ap.add_argument("--zup", action="store_true", help="world is Z-up: rotate it to the pipeline's Y-up")
    a = ap.parse_args()
    # (x, y, z) -> (x, z, -y): a proper rotation (det +1) taking +Z to +Y
    W = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]]) if a.zup else np.eye(4)
    depth = sorted((a.src / "lowres_depth").glob("*.png"), key=_ts)
    traj = np.loadtxt(a.src / "lowres_wide.traj")
    t_traj = traj[:, 0]
    (a.dst / "depth").mkdir(parents=True, exist_ok=True)
    (a.dst / "confidence").mkdir(parents=True, exist_ok=True)
    rows, K0, writer = [], None, None
    flip = np.diag([1.0, -1.0, -1.0]) if a.flip else np.eye(3)
    for i, dp in enumerate(depth):
        ts = _ts(dp)
        j = int(np.argmin(np.abs(t_traj - ts)))
        if abs(t_traj[j] - ts) > 0.05:          # no pose within 50 ms: skip frame
            continue
        R = Rotation.from_rotvec(traj[j, 1:4]).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, traj[j, 4:7]
        if a.invert:
            T = np.linalg.inv(T)
        T[:3, :3] = T[:3, :3] @ flip            # express pose for an OpenCV camera
        T = W @ T                               # world into the pipeline's +Y-up frame
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        n = len(rows)
        shutil.copy(dp, a.dst / "depth" / f"{n:06d}.png")
        cp = a.src / "confidence" / dp.name
        if cp.exists():
            shutil.copy(cp, a.dst / "confidence" / f"{n:06d}.png")
        pin = a.src / "lowres_wide_intrinsics" / (dp.stem + ".pincam")
        if K0 is None and pin.exists():
            w, h, fx, fy, cx, cy = np.loadtxt(pin)
            K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]) * np.array([[SCALE], [SCALE], [1]])
        rgb = cv2.imread(str(a.src / "lowres_wide" / dp.name))
        if rgb is not None:
            if writer is None:
                writer = cv2.VideoWriter(str(a.dst / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         10.0, (rgb.shape[1], rgb.shape[0]))
            writer.write(rgb)
        rows.append([ts, n, *T[:3, 3], *q])
    if writer:
        writer.release()
    np.savetxt(a.dst / "camera_matrix.csv", K0, delimiter=",", fmt="%.4f")
    # same columns as Stray Scanner: the loader reads per-frame fx, fy, cx, cy (cols 9-12)
    kcols = f"{K0[0, 0]:.4f}, {K0[1, 1]:.4f}, {K0[0, 2]:.4f}, {K0[1, 2]:.4f}"
    with open(a.dst / "odometry.csv", "w") as f:
        f.write("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy\n")
        for r in rows:
            f.write(f"{r[0]:.6f}, {r[1]:06d}, " + ", ".join(f"{x:.8f}" for x in r[2:])
                    + f", {kcols}\n")
    print(f"{len(rows)} frames -> {a.dst}")


if __name__ == "__main__":
    main()
