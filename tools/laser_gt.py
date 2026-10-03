"""Ground truth from an ARKitScenes static laser scan (mm-grade, no drift).

    python tools/laser_gt.py data/public/arkitscenes/laser/170107 [--every 20]

Reads the binary PLY by memory map (42M points, 44 B each), subsamples,
applies the scan pose and the Z-up -> Y-up rotation used by
tools/arkitscenes_to_stray.py, estimates normals by local PCA, then runs the
pipeline's wall-face extractor and floor/ceiling finder on the *laser* points.
Writes out/laser_gt/faces.json and a labelled top-down render
out/laser_gt/topdown.png from which the room's clockwise walls are read into
benchmark/public_gt/<visit>.json.

Caveat (stated in the report): the face extractor is shared code; the sensor
is independent (static terrestrial laser vs handheld iPad LiDAR).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from roomscan.geometry.cloud import OrientedCloud, voxel_downsample
from roomscan.geometry.levels import floor_ceiling, manhattan_yaw, yaw_matrix
from roomscan.geometry.walls import extract_faces

DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"),
                  ("b", "u1"), ("a", "u1"), ("quality", "<f8"), ("radius", "<f8")])
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])


def load_ply(path: Path, every: int) -> np.ndarray:
    raw = path.read_bytes()[:4096]
    hdr_end = raw.index(b"end_header") + len(b"end_header\n")
    n = int(raw.split(b"element vertex ")[1].split(b"\n")[0])
    mm = np.memmap(path, dtype=DTYPE, mode="r", offset=hdr_end, shape=(n,))
    sub = mm[::every]
    return np.stack([sub["x"], sub["y"], sub["z"]], 1).astype(np.float64)


def pca_normals(P: np.ndarray, k: int = 16) -> np.ndarray:
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k, workers=-1)
    Q = P[idx] - P[idx].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    _, V = np.linalg.eigh(C)
    return V[:, :, 0]                       # smallest eigenvector = surface normal


def load_aligned(stem: Path, every: int = 20) -> OrientedCloud:
    """Laser scan -> subsampled, Y-up, recentred, Manhattan-aligned oriented cloud."""
    P = load_ply(stem.with_suffix(".ply"), every)
    pose = np.loadtxt(str(stem) + "_pose.txt", delimiter=",")
    T = pose.T if np.allclose(pose[:3, 3], 0) else pose    # file stores the transpose
    P = (P @ T[:3, :3].T + T[:3, 3]) @ ZUP_TO_YUP.T
    P -= np.median(P, 0)                                    # recentre (scanner frame is geo-offset)
    N = pca_normals(P)
    c = voxel_downsample(OrientedCloud(P, N, np.zeros(len(P), np.int32)), 0.01)
    # PCA normals have no consistent sign; orient vertical ones by height, walls towards centre
    up = np.abs(c.normals[:, 1]) > 0.9
    floor0 = np.percentile(c.xyz[:, 1], 2)
    c.normals[up] *= np.sign(c.normals[up, 1:2]) * np.where(c.xyz[up, 1:2] < floor0 + 1.2, 1, -1)
    side = ~up
    to_c = -c.xyz[side][:, [0, 2]]
    flip = np.sum(c.normals[side][:, [0, 2]] * to_c, 1) < 0
    nn = c.normals[side]
    nn[flip] *= -1
    c.normals[side] = nn
    return c.transformed(yaw_matrix(manhattan_yaw(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", type=Path, help="path without extension, e.g. .../laser/170107")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    c = load_aligned(a.stem, a.every)
    P = c.xyz
    fl, ce = floor_ceiling(c)
    faces = extract_faces(c, fl.y, ce.y if ce else None, min_support=300, min_len=0.3)
    out = Path("out/laser_gt")
    out.mkdir(parents=True, exist_ok=True)
    rec = [{"i": i, "axis": f.axis, "sign": f.sign, "pos": round(f.pos, 4), "lo": round(f.lo, 3),
            "hi": round(f.hi, 3), "len": round(f.length, 3), "support": f.support}
           for i, f in enumerate(faces)]
    json.dump({"floor_y": fl.y, "ceiling_y": ce.y if ce else None,
               "ceiling_height": (ce.y - fl.y) if ce else None, "faces": rec},
              open(out / "faces.json", "w"), indent=1)
    print(f"{len(P)} pts, floor {fl.y:.4f}, ceiling {ce.y if ce else None}, "
          f"height {(ce.y - fl.y) if ce else None}, {len(faces)} faces")
    # top-down render with face ids
    res, S = 0.02, 1
    xz = c.xyz[:, [0, 2]]
    lo = xz.min(0) - 0.3
    W, H = ((xz.max(0) - lo + 0.3) / res).astype(int) + 1
    img = np.zeros((H, W))
    ij = ((xz - lo) / res).astype(int)
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1)
    vis = cv2.cvtColor((255 * np.clip(img / np.percentile(img[img > 0], 97), 0, 1) * 0.4).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    px = lambda x, z: (int((x - lo[0]) / res), int((z - lo[1]) / res))
    for i, f in enumerate(faces):
        if f.length < 0.6:
            continue
        p, q = ((px(f.pos, f.lo), px(f.pos, f.hi)) if f.axis == "x" else (px(f.lo, f.pos), px(f.hi, f.pos)))
        cv2.line(vis, p, q, (0, 0, 255) if f.sign > 0 else (255, 160, 0), 2)
        mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        cv2.putText(vis, f"{i}:{f.length:.2f}", mid, 0, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out / "topdown.png"), vis)


if __name__ == "__main__":
    main()
