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
MIN_TRACKS = 40       # below this, take the keyframe even if the frame is blurry
BLUR_MIN = 40.0
MIN_DEPTH = 0.5       # m; closer keyframes are where KLT and ICP both fail (D16)
CLOSE_RETRY = 5       # frames to wait after rejecting a close-up (each check is a MoGe run)
MAX_KEYFRAMES = 400   # ~10 s each with MoGe-L on CPU (notes D13)


def too_close(mv, min_depth: float = MIN_DEPTH) -> bool:
    """True if the frame mostly shows a surface closer than min_depth (median
    range over valid pixels), or has no valid geometry at all."""
    P = mv.points[mv.mask]
    if len(P) == 0:
        return True
    return float(np.median(np.linalg.norm(P, axis=1))) < min_depth


def track_keyframes(path: Path):
    """One pass over the clip: KLT-track corners frame to frame and start a new
    keyframe when tracking degrades (tracked < KEEP_FRAC of the seeded corners),
    after MAX_GAP_S at most, never closer than MIN_GAP_S. Blurry candidates are
    skipped (the next sharp frame becomes the keyframe) unless tracking is dying;
    close-ups (median depth < MIN_DEPTH) are skipped, retried every CLOSE_RETRY frames.

    Returns keyframes [(frame_idx, rgb)] and, per consecutive pair, the tracked
    pixel correspondences at MoGe inference resolution."""
    from roomscan import mono
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    kfs, tracks = [], []
    prev_g, p0, p, n_seed, last_kf = None, None, None, 0, -10 ** 9
    retry_at = 0           # next frame index allowed to be checked after a close-up
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
            fb_err = np.linalg.norm((pb - p).reshape(-1, 2), axis=1)   # forward-backward check
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0)
            p, p0 = p1[good], p0[good]
        gap = (i - last_kf) / fps
        degraded = p is None or len(p) < KEEP_FRAC * max(n_seed, 1) or gap >= MAX_GAP_S
        sharp = cv2.Laplacian(g, cv2.CV_64F).var() >= BLUR_MIN
        dying = p is not None and len(p) < MIN_TRACKS
        # prefer sharp keyframes, but never let tracking die waiting for one:
        # MoGe tolerates mild blur far better than odometry tolerates a gap
        want_kf = (not kfs and sharp) or (kfs and degraded and gap >= MIN_GAP_S and (sharp or dying))
        mv = None
        if want_kf and i >= retry_at:
            mv = mono.infer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))   # reused for the view (A1)
            if too_close(mv):
                retry_at = i + CLOSE_RETRY      # keep tracking; try again a bit later
                want_kf = False
        if want_kf and i >= retry_at:
            if kfs:
                tracks.append((p0.reshape(-1, 2).copy(), p.reshape(-1, 2).copy()))
            kfs.append((i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mv))
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


def _wall2d(v: photo.View, stride: int = 4):
    """Wall points + normals (x, z) of a keyframe in its own gravity frame."""
    P = v.mv.points[::stride, ::stride][v.mv.mask[::stride, ::stride]] @ v.R_g.T
    N = v.mv.normals[::stride, ::stride][v.mv.mask[::stride, ::stride]] @ v.R_g.T
    m = np.abs(N[:, 1]) < 0.2
    n2 = N[m][:, [0, 2]]
    return P[m][:, [0, 2]], n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)


def camera_height(v: photo.View) -> float | None:
    """Camera height above the floor this keyframe sees (its own gravity frame),
    or None if no floor plane is visible."""
    from roomscan.geometry.levels import _plane_height
    P = v.mv.points[v.mv.mask] @ v.R_g.T
    N = v.mv.normals[v.mv.mask] @ v.R_g.T
    y = P[(N[:, 1] > 0.9) & (P[:, 1] < -0.5), 1]
    lv = _plane_height(y, "floor", min_support=200)
    return -lv.y if lv else None


def level_heights(T: list[np.ndarray], heights: list[float | None]) -> list[np.ndarray]:
    """Floor-anchored height: put every keyframe at its measured height above the
    floor (floor = world y 0) instead of the chained vertical shifts, which drift
    (stand-in: a phantom floor 1.9 m below the real one, D17). Keyframes without
    a visible floor are interpolated from their neighbours."""
    idx = [k for k, h in enumerate(heights) if h is not None]
    if not idx:
        return T
    h_all = np.interp(np.arange(len(T)), idx, [heights[k] for k in idx])
    out = []
    for Tk, h in zip(T, h_all):
        Tk = Tk.copy()
        Tk[1, 3] = h
        out.append(Tk)
    return out


def icp_pair(vi: photo.View, vj: photo.View):
    """Geometric fallback for textureless stretches: 2D point-to-plane ICP of j's
    wall points onto i's, started at 'no motion' (<= 2 s apart). Returns the
    same (s, R, t, n) tuple as the feature paths, or None."""
    Pi, Ni = _wall2d(vi)
    Pj, _ = _wall2d(vj)
    if len(Pi) < 200 or len(Pj) < 200:
        return None
    T, inl, rmse = D._icp2d(Pj, Pi, Ni, np.eye(3), max_d=0.25)
    # MoGe's point-map noise grows with range, so the residual limit does too
    # (a fixed 6 cm rejected pairs with 0.9-1.0 inliers at 1-2 m, notes D16)
    depth = float(np.median(np.linalg.norm(Pi, axis=1)))
    if T is None or inl < 0.6 or rmse > 0.03 + 0.05 * depth:
        return None
    th = np.arctan2(T[0, 1], T[0, 0])
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    return 1.0, R, np.array([T[0, 2], 0.0, T[1, 2]]), int(inl * 1000)


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
            if r is None:
                r = icp_pair(views[k - back], views[k])
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
    # one MoGe inference per keyframe: the result from the close-up check is
    # reused (it previously ran again with a fixed FOV, ~2x the time; D23)
    views = [photo.make_view(f"kf{k:06d}", rgb, mv=mv) for k, rgb, mv in kfs]
    T, seg, odo_pairs = odometry(views, tracks)
    # Segments have no relative pose between them; fusing them anyway produced a
    # merged blob with confident intervals (D17). Keep only the largest one.
    n_seg, n_total = int(max(seg) + 1), len(views)
    seg = np.array(seg)
    keep_id = int(np.bincount(seg).argmax())
    keep = np.flatnonzero(seg == keep_id)
    remap = {int(old): new for new, old in enumerate(keep)}
    views = [views[i] for i in keep]
    T = [T[i] for i in keep]
    odo_pairs = [(remap[i], remap[j]) for i, j in odo_pairs if i in remap and j in remap]
    seg = [0] * len(views)
    coverage = {"segments": n_seg, "kept_keyframes": len(views), "total_keyframes": n_total}
    info = {"method": "none (chained visual odometry as-is)", "enabled": False,
            "correction_m": None, "loop_closures": 0, **coverage}
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
                "loop_closures": len(lc), **coverage}
    T = level_heights(T, [camera_height(v) for v in views])
    clouds = []
    for k, (v, Tk) in enumerate(zip(views, T)):
        v.T, v.scale = Tk, 1.0
        clouds.append(photo.view_cloud(v, k, stride=3))
    c = voxel_downsample(OrientedCloud.concat(clouds), 0.03)
    P = np.array([t[[0, 2], 3] for t in T])
    path_len = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    # how much the pose graph had to bend the trajectory, relative to its length:
    # widens every interval by that fraction (plan.build_plan extra_rel)
    info["distortion_rel"] = round(float(info["correction_m"] or 0.0) / max(path_len, 1.0), 4)
    return c, path_len, info, len(kfs), views
