"""Video tier: handheld walkthrough clip, no depth, no poses.

1. Keyframes chosen by tracking quality: KLT corners are tracked through every
   frame; a new keyframe starts when too few survive (blurry frames skipped).
2. Per keyframe: MoGe-2 metric point map + gravity (photo.make_view).
3. Visual-metric odometry: the KLT correspondences between consecutive
   keyframes, lifted by their MoGe point maps, give a 4-DOF RANSAC alignment
   (SIFT matching as fallback). Scale is NOT chained
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

KEEP_FRAC = 0.30      # new keyframe when < 30% of seeded corners survive
MIN_GAP_S, MAX_GAP_S = 0.3, 2.0
BLUR_MIN = 40.0
MAX_KEYFRAMES = 400   # ~10 s each with MoGe-L on CPU (notes D13)


def track_keyframes(path: Path):
    """One pass over the clip: KLT-track corners frame to frame and start a new
    keyframe when tracking degrades (tracked < KEEP_FRAC of the seeded corners),
    after MAX_GAP_S at most, never closer than MIN_GAP_S. Blurry candidates are
    skipped (the next sharp frame becomes the keyframe).

    Returns keyframes [(frame_idx, rgb)] and, per consecutive pair, the tracked
    pixel correspondences at MoGe inference resolution."""
    from roomscan import mono
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    kfs, tracks = [], []
    prev_g, p0, p, n_seed, last_kf = None, None, None, 0, -10 ** 9
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rgb_small = mono.resize_long(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        g = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
        if p is not None and len(p):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, p, None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev_g, p1, None, **lk)
            good = (st.ravel() == 1) & (st2.ravel() == 1) &                 (np.linalg.norm((pb - p).reshape(-1, 2), axis=1) < 1.0)
            p, p0 = p1[good], p0[good]
        gap = (i - last_kf) / fps
        degraded = p is None or len(p) < KEEP_FRAC * max(n_seed, 1) or gap >= MAX_GAP_S
        if (not kfs or (degraded and gap >= MIN_GAP_S)) and                 cv2.Laplacian(g, cv2.CV_64F).var() >= BLUR_MIN:
            if kfs:
                tracks.append((p0.reshape(-1, 2).copy(), p.reshape(-1, 2).copy()))
            kfs.append((i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            last_kf = i
            seed = cv2.goodFeaturesToTrack(g, 1500, 0.005, 7)
            p = seed.astype(np.float32) if seed is not None else np.zeros((0, 1, 2), np.float32)
            p0, n_seed = p.copy(), len(p)
        prev_g = g
        i += 1
    if len(kfs) > MAX_KEYFRAMES:
        raise RuntimeError(f"{len(kfs)} keyframes > {MAX_KEYFRAMES}: clip too long for one run; "
                           "split it per floor")
    return kfs, tracks


def self_calibrated_fov(kfs, n: int = 10) -> float:
    """One lens for the whole clip: median of MoGe's per-frame FOV estimates over
    n spread keyframes, then every frame is inferred with that fixed FOV. Per-frame
    FOV guesses scatter (fx ratio 0.89-1.66 vs ARKit on sample frames) and scale
    error follows the FOV error (notes D14)."""
    from roomscan import mono
    sel = np.linspace(0, len(kfs) - 1, min(n, len(kfs))).astype(int)
    fovs = []
    for i in sel:
        v = mono.infer(kfs[i][1])
        w = v.points.shape[1]
        fovs.append(np.degrees(2 * np.arctan(w / (2 * v.K[0, 0]))))
    return float(np.median(fovs))


def _rigid(s, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _se2_from(T):
    yaw = np.arctan2(T[0, 2], T[2, 2])   # heading of the gravity frame's +Z
    return D._se2(T[0, 3], T[2, 3], yaw)


def odometry(views: list[photo.View], tracks=None) -> tuple[list[np.ndarray], list[int], list]:
    """Chained 4x4 poses (gravity frame k -> world); segment id per frame.
    KLT tracks between consecutive keyframes are tried first, SIFT as fallback."""
    T = [np.eye(4)]
    seg = [0]
    odo = []
    for k in range(1, len(views)):
        r, back = None, 1
        if tracks is not None:
            qi, qj = tracks[k - 1]
            r = photo.fit_correspondences(views[k - 1], views[k], qi.astype(np.float32),
                                          qj.astype(np.float32), seed=k)
        for back in ((1, 2, 3) if r is None else (1,)):   # skip a bad frame if needed
            if k - back < 0:
                break
            if r is None:
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
    kfs, tracks = track_keyframes(path)
    if len(kfs) < 3:
        raise RuntimeError("video too short or too blurry: fewer than 3 usable keyframes")
    fov = self_calibrated_fov(kfs)
    views = [photo.make_view(f"kf{k:06d}", rgb, fov) for k, rgb in kfs]
    T, seg, odo_pairs = odometry(views, tracks)
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
