"""Drift correction for multi-room captures: 2D submap pose graph.

ARKit VIO drifts slowly in yaw and position over a long walk; it is gravity
aligned, so roll/pitch/height drift is negligible and the graph is solved in
(x, z, yaw). Nodes are submap anchors (~1.5 m of walking each).

Factors
  odometry      relative anchor motion from ARKit (sigma 1 cm + 1 %, 0.3 deg)
  loop closure  2D point-to-plane ICP between non-consecutive submaps that
                observe the same walls (revisits); robust (soft-L1) loss
  Manhattan     each submap's dominant wall angle (mod 90 deg) is pulled to the
                global Manhattan angle: "plane-anchored" yaw correction

Solved with scipy least_squares; per-frame corrections are interpolated
between anchors. Disabled with --no-drift for the ablation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from roomscan.geometry.cloud import depth_to_cloud
from roomscan.io import strayscanner as ss

SUBMAP_PATH = 1.5        # metres walked per submap
ODO_SIG_XY, ODO_SIG_REL, ODO_SIG_YAW = 0.01, 0.01, np.deg2rad(0.3)
MAN_SIG_YAW = np.deg2rad(0.5)
LC_SIG_XY, LC_SIG_YAW = 0.02, np.deg2rad(0.5)


def _yaw_of(R: np.ndarray) -> float:
    """Heading about +Y of the camera's forward (+Z) axis projected to x-z."""
    f = R[:, 2]
    return float(np.arctan2(f[0], f[2]))


def _se2(x, z, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s, x], [-s, c, z], [0, 0, 1.0]])   # rotation about +Y in (x,z)


def _se2_params(T):
    return T[0, 2], T[1, 2], np.arctan2(T[0, 1], T[0, 0])


def _submaps(cap: ss.LidarCapture):
    frames = cap.depth_frames
    pos = np.array([f.T_wc[[0, 2], 3] for f in frames])
    step = np.r_[0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    cum = np.cumsum(step)
    sid = (cum / SUBMAP_PATH).astype(int)
    groups = [np.flatnonzero(sid == k) for k in np.unique(sid)]
    return frames, [g for g in groups if len(g) >= 5]


def _wall_pts(cap, frames, idx, T_anchor_inv):
    """Wall-band points (+normals) of a submap, in the anchor's 2D frame."""
    P, N = [], []
    for i in idx[::4]:
        f = frames[i]
        d, K = cap.load_depth(f)
        c = depth_to_cloud(d, K, f.T_wc, f.index, stride=4)
        m = (np.abs(c.normals[:, 1]) < 0.2) & (c.xyz[:, 1] > f.T_wc[1, 3] - 1.0)
        P.append(c.xyz[m][:, [0, 2]])
        N.append(c.normals[m][:, [0, 2]])
    P, N = np.concatenate(P), np.concatenate(N)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    R, t = T_anchor_inv[:2, :2], T_anchor_inv[:2, 2]
    return P @ R.T + t, N @ R.T


def _icp2d(src, dst, dst_n, T0, iters=20, max_d=0.15):
    tree = cKDTree(dst)
    T = T0.copy()
    for _ in range(iters):
        p = src @ T[:2, :2].T + T[:2, 2]
        d, j = tree.query(p, distance_upper_bound=max_d)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            return None, 0.0, np.inf
        q, n, pp = dst[j[ok]], dst_n[j[ok]], p[ok]
        # linearised point-to-plane. Rotating p by small dth with [[c,s],[-s,c]]
        # gives p + dth*(p_z, -p_x), so d(n.p)/dth = n_x p_z - n_z p_x.
        A = np.c_[n, n[:, 0] * pp[:, 1] - n[:, 1] * pp[:, 0]]
        b = np.sum(n * (q - pp), 1)
        dx = np.linalg.lstsq(A, b, rcond=None)[0]
        c, s = np.cos(dx[2]), np.sin(dx[2])
        T = np.array([[c, s, dx[0]], [-s, c, dx[1]], [0, 0, 1]]) @ T
        if np.abs(dx).max() < 1e-4:
            break
    p = src @ T[:2, :2].T + T[:2, 2]
    d, _ = tree.query(p, distance_upper_bound=max_d)
    ok = np.isfinite(d)
    return T, ok.mean(), float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else np.inf


def _manhattan_angle(N):
    th = np.arctan2(N[:, 1], N[:, 0])
    return float(np.angle(np.sum(np.exp(4j * th))) / 4)


def correct_drift(cap: ss.LidarCapture):
    frames, groups = _submaps(cap)
    info = {"method": "2D submap pose graph: ARKit odometry + ICP loop closures "
                      "+ Manhattan yaw prior", "enabled": True,
            "correction_m": 0.0, "loop_closures": 0, "submaps": len(groups)}
    if len(groups) < 3:
        return None, info
    anchors = [frames[g[0]].T_wc for g in groups]
    a2d = [_se2(T[0, 3], T[2, 3], _yaw_of(T[:3, :3])) for T in anchors]
    a2d_inv = [np.linalg.inv(T) for T in a2d]
    sub = [_wall_pts(cap, frames, g, a2d_inv[k]) for k, g in enumerate(groups)]
    # global Manhattan angle from all submaps in the ARKit world frame
    allN = np.concatenate([n @ a2d[k][:2, :2].T for k, (_, n) in enumerate(sub)])
    man = _manhattan_angle(allN)

    x0 = np.array([_se2_params(T) for T in a2d]).ravel()
    odo = [(k, k + 1, a2d_inv[k] @ a2d[k + 1]) for k in range(len(groups) - 1)]

    # loop closures between non-adjacent submaps whose anchors are close
    lc = []
    pos = x0.reshape(-1, 3)[:, :2]
    for i in range(len(groups)):
        for j in range(i + 3, len(groups)):
            if np.linalg.norm(pos[i] - pos[j]) > 2.0:
                continue
            T0 = a2d_inv[i] @ a2d[j]                   # j's frame -> i's frame (ARKit)
            T, inl, rmse = _icp2d(sub[j][0], sub[i][0], sub[i][1], T0)
            if T is None or inl < 0.5 or rmse > 0.04:
                continue
            lc.append((i, j, T))
    info["loop_closures"] = len(lc)

    def resid(x):
        X = x.reshape(-1, 3)
        T = [_se2(*p) for p in X]
        r = []
        for edges, is_odo in ((odo, True), (lc, False)):
            for i, j, Z in edges:
                ex, ez, eth = _se2_params(np.linalg.inv(Z) @ np.linalg.inv(T[i]) @ T[j])
                if is_odo:
                    sxy = ODO_SIG_XY + ODO_SIG_REL * np.linalg.norm(Z[:2, 2])
                    r += [ex / sxy, ez / sxy, eth / ODO_SIG_YAW]
                else:
                    r += [ex / LC_SIG_XY, ez / LC_SIG_XY, eth / LC_SIG_YAW]
        for k, (_, n) in enumerate(sub):       # Manhattan prior per submap
            a = _manhattan_angle(n) - X[k, 2]  # submap wall angle in world under X
            dlt = np.angle(np.exp(4j * (a - man))) / 4
            r.append(dlt / MAN_SIG_YAW)
        r += list((X[0] - x0[:3]) / 1e-4)      # gauge: first anchor fixed
        return np.array(r)

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, max_nfev=200)
    X = sol.x.reshape(-1, 3)
    corr = [_se2(*X[k]) @ a2d_inv[k] for k in range(len(groups))]   # world -> world fix
    info["correction_m"] = round(float(np.max(np.linalg.norm(X[:, :2] - x0.reshape(-1, 3)[:, :2], axis=1))), 4)
    info["max_yaw_correction_deg"] = round(float(np.degrees(np.max(np.abs(
        np.angle(np.exp(1j * (X[:, 2] - x0.reshape(-1, 3)[:, 2]))))))), 3)

    poses = {}
    gid = np.zeros(len(frames), int)
    for k, g in enumerate(groups):
        gid[g[0]:] = k
    for i, f in enumerate(frames):
        C2 = corr[gid[i]]
        C = np.eye(4)
        C[0, 0], C[0, 2], C[2, 0], C[2, 2] = C2[0, 0], C2[0, 1], C2[1, 0], C2[1, 1]
        C[0, 3], C[2, 3] = C2[0, 2], C2[1, 2]
        poses[f.index] = C @ f.T_wc
    return poses, info
