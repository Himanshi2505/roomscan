"""Register a LiDAR recording's plan frame to the laser ground-truth frame.

    python tools/register_to_laser.py data/public/stray/42445021 out/public/42445021 \
        data/public/arkitscenes/laser/170107 [--no-drift]

Rebuilds the recording's cloud exactly as `roomscan run` did (same drift
correction, same Manhattan rotation -> the frame result.json's polygons live
in), then 2D point-to-plane ICP of its wall points onto the laser wall points
from every quarter turn x a grid of start shifts; keeps the best. Accept only
if the fine inlier ratio (5 cm) >= 0.7 and RMSE <= 3 cm.

Writes <result_dir>/laser_registration.json (SE2 plan->laser, quality) and
<result_dir>/laser_overlay.png (laser walls red, floor green, plan white).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roomscan import drift as D
from roomscan.geometry.levels import manhattan_yaw, yaw_matrix
from roomscan.pipeline import _lidar_cloud
from tools.laser_gt import load_aligned

ACCEPT_INLIERS, ACCEPT_RMSE = 0.7, 0.03


def wall2d(c, n_max=20000, seed=0):
    y = c.xyz[:, 1]
    floor = np.percentile(y, 2)
    m = (y > floor + 0.3) & (y < floor + 2.0) & (np.abs(c.normals[:, 1]) < 0.2)
    P, N = c.xyz[m][:, [0, 2]], c.normals[m][:, [0, 2]]
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    if len(P) > n_max:
        i = np.random.default_rng(seed).choice(len(P), n_max, replace=False)
        P, N = P[i], N[i]
    return P, N


def fit_quality(T, src, dst):
    from scipy.spatial import cKDTree
    p = src @ T[:2, :2].T + T[:2, 2]
    d, _ = cKDTree(dst).query(p, distance_upper_bound=0.05)
    ok = np.isfinite(d)
    return float(ok.mean()), float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("laser_stem", type=Path)
    ap.add_argument("--no-drift", action="store_true")
    a = ap.parse_args()
    c, _ = _lidar_cloud(a.capture, not a.no_drift, {}, {})
    c = c.transformed(yaw_matrix(manhattan_yaw(c)))
    lc = load_aligned(a.laser_stem)
    src, _ = wall2d(c)
    dst, dst_n = wall2d(lc, n_max=60000)
    # grid of start shifts: align centroids, then offsets over the laser extent
    span = dst.max(0) - dst.min(0)
    offs = [np.array([dx, dz]) for dx in np.arange(-span[0] / 2, span[0] / 2 + 0.01, 0.75)
            for dz in np.arange(-span[1] / 2, span[1] / 2 + 0.01, 0.75)]
    best = None
    for k in range(4):
        th = k * np.pi / 2
        R = D._se2(0, 0, th)[:2, :2]
        base = dst.mean(0) - src.mean(0) @ R.T
        for o in offs:
            t = base + o
            T0 = D._se2(t[0], t[1], th)
            T, inl, rmse = D._icp2d(src, dst, dst_n, T0, iters=30, max_d=0.3)
            if T is None:
                continue
            q = fit_quality(T, src, dst)
            if best is None or q[0] > best[1][0]:
                best = (T, q)
    T, (inl, rmse) = best
    ok = inl >= ACCEPT_INLIERS and rmse <= ACCEPT_RMSE
    out = {"T_plan_to_laser": T.tolist(), "inliers_5cm": round(inl, 3), "rmse_m": round(rmse, 4),
           "accepted": bool(ok), "drift": not a.no_drift}
    (a.result_dir / "laser_registration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    # overlay
    plan = json.loads((a.result_dir / "result.json").read_text())
    res = 0.02
    lo = lc.xyz[:, [0, 2]].min(0) - 1.0
    hi = lc.xyz[:, [0, 2]].max(0) + 1.0
    H, W = ((hi - lo) / res).astype(int)[::-1] + 1
    vis = np.zeros((H, W, 3), np.uint8)
    for P, col in ((dst, (0, 0, 255)),):
        ij = ((P - lo) / res).astype(int)
        vis[ij[:, 1].clip(0, H - 1), ij[:, 0].clip(0, W - 1)] = col
    for r in plan["rooms"]:
        V = np.array(r["polygon"]) @ T[:2, :2].T + T[:2, 2]
        q = ((V - lo) / res).astype(np.int32)
        cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
        cv2.putText(vis, f"{r['id']} {r['floor_area']['value']:.1f}m2", tuple(q.mean(0).astype(int) - [40, 0]),
                    0, 0.5, (255, 255, 0), 1)
    srcT = src @ T[:2, :2].T + T[:2, 2]
    ij = ((srcT - lo) / res).astype(int)
    vis[ij[:, 1].clip(0, H - 1), ij[:, 0].clip(0, W - 1)] = (0, 255, 0)
    cv2.imwrite(str(a.result_dir / "laser_overlay.png"), vis)


if __name__ == "__main__":
    main()
