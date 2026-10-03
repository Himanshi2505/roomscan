"""Video tier: handheld walkthrough clip, no depth, no poses.

1. Keyframes at ~KF_FPS, skipping blurry frames (variance of Laplacian).
2. Per keyframe: MoGe-2 metric point map + gravity (photo.make_view).
3. Visual-metric odometry: consecutive keyframes registered with the same
   SIFT + point-map 4-DOF RANSAC as the photo tier. Scale is NOT chained
   (that would random-walk): every frame keeps MoGe's own metric scale, so the
   plan's scale error is MoGe's average bias, not an accumulated drift.
4. Loop closures: non-consecutive keyframes within 2.5 m that register
   (the protocol ends the walk where it began, guaranteeing at least one).
5. Same (x, z, yaw) pose graph + Manhattan prior as the LiDAR tier.
6. Fuse corrected keyframes -> OrientedCloud -> shared plan builder.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from roomscan import drift as D
from roomscan.geometry.cloud import OrientedCloud, voxel_downsample
from roomscan.tiers import photo

KF_FPS = 2.0
BLUR_MIN = 40.0
MAX_KEYFRAMES = 400


def keyframes(path: Path) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / KF_FPS)))
    out, i = [], 0
    window = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % max(1, step // 3) == 0:          # look at 3 candidates per slot, keep the sharpest
            _, img = cap.retrieve()
            small = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
            sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            window.append((sharp, i, img))
        if (i + 1) % step == 0 and window:
            sharp, k, img = max(window, key=lambda x: x[0])
            if sharp >= BLUR_MIN:
                out.append((k, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            window = []
        i += 1
    if len(out) > MAX_KEYFRAMES:
        sel = np.linspace(0, len(out) - 1, MAX_KEYFRAMES).astype(int)
        out = [out[j] for j in sel]
    return out


def _rigid(s, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _se2_from(T):
    yaw = np.arctan2(T[0, 2], T[2, 2])   # heading of the gravity frame's +Z
    return D._se2(T[0, 3], T[2, 3], yaw)


def odometry(views: list[photo.View]) -> tuple[list[np.ndarray], list[int], list]:
    """Chained 4x4 poses (gravity frame k -> world); segment id per frame."""
    T = [np.eye(4)]
    seg = [0]
    odo = []
    for k in range(1, len(views)):
        r = None
        for back in (1, 2, 3):                    # skip a bad frame if needed
            if k - back < 0:
                break
            r = photo.register_pair(views[k - back], views[k], seed=k)
            if r is not None:
                break
        if r is None:
            T.append(T[-1].copy())                # lost: hold pose, flag new segment
            seg.append(seg[-1] + 1)
            continue
        s, R, t, _ = r
        Tk = T[k - back] @ _rigid(s, R, t)
        T.append(Tk)
        seg.append(seg[k - back])
        odo.append((k - back, k))
    return T, seg, odo


def build(path: Path, drift: bool = True):
    kfs = keyframes(path)
    if len(kfs) < 3:
        raise RuntimeError("video too short or too blurry: fewer than 3 usable keyframes")
    views = [photo.make_view(f"kf{k:06d}", rgb) for k, rgb in kfs]
    T, seg, odo_pairs = odometry(views)
    info = {"method": "none (chained visual odometry as-is)", "enabled": False,
            "correction_m": None, "loop_closures": 0, "segments": int(max(seg) + 1)}
    if drift and len(views) > 3:
        x2 = [_se2_from(t) for t in T]
        x2_inv = [np.linalg.inv(a) for a in x2]
        x0 = np.array([D._se2_params(a) for a in x2]).ravel()
        odo = [(i, j, x2_inv[i] @ x2[j]) for i, j in odo_pairs]
        pos = x0.reshape(-1, 3)[:, :2]
        lc = []
        n = len(views)
        pairs = {(i, j) for i in range(n) for j in range(i + 8, n)
                 if seg[i] == seg[j] and np.linalg.norm(pos[i] - pos[j]) < 2.5}
        # protocol: walk ends where it started -> always try start vs end
        pairs |= {(i, j) for i in range(min(4, n)) for j in range(max(0, n - 4), n) if j - i >= 8}
        for i, j in sorted(pairs):
            r = photo.register_pair(views[i], views[j], seed=7 * i + j)
            if r is None or r[3] < 40:
                continue
            lc.append((i, j, _se2_from(_rigid(*r[:3]))))
        normals = []
        for v in views:   # wall normals of each keyframe in its gravity frame (x, z)
            n = v.mv.normals[v.mv.mask][::20] @ v.R_g.T
            n = n[np.abs(n[:, 1]) < 0.2][:, [0, 2]]
            normals.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
        allN = np.concatenate([n @ x2[k][:2, :2].T for k, n in enumerate(normals)])
        man = D._manhattan_angle(allN)
        X = D.solve_pose_graph(x0, odo, lc, normals, man,
                               odo_sig=(0.03, 0.03, np.deg2rad(1.0)),
                               lc_sig=(0.05, np.deg2rad(1.0)))
        for k in range(len(T)):
            C2 = D._se2(*X[k]) @ x2_inv[k]
            C = np.eye(4)
            C[0, 0], C[0, 2], C[2, 0], C[2, 2] = C2[0, 0], C2[0, 1], C2[1, 0], C2[1, 1]
            C[0, 3], C[2, 3] = C2[0, 2], C2[1, 2]
            T[k] = C @ T[k]
        info = {"method": "2D keyframe pose graph: visual-metric odometry + feature loop "
                          "closures + Manhattan yaw prior", "enabled": True,
                "correction_m": round(float(np.max(np.linalg.norm(X[:, :2] - pos, axis=1))), 4),
                "loop_closures": len(lc), "segments": int(max(seg) + 1)}
    clouds = []
    for k, (v, Tk) in enumerate(zip(views, T)):
        v.T, v.scale = Tk, 1.0
        clouds.append(photo.view_cloud(v, k, stride=3))
    c = voxel_downsample(OrientedCloud.concat(clouds), 0.03)
    P = np.array([t[[0, 2], 3] for t in T])
    path_len = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return c, path_len, info, len(kfs), views
