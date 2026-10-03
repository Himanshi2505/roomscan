"""Oriented point cloud shared by all tiers.

Every tier (LiDAR, video, photos) ends up producing an OrientedCloud in a
gravity-aligned, metric, +Y-up world frame. Everything downstream (levels,
walls, rooms, openings, damage) only consumes this, which is what lets the
three tiers honour one output contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OrientedCloud:
    xyz: np.ndarray                 # (N,3) metres, +Y up
    normals: np.ndarray             # (N,3) unit, oriented towards the camera
    frame: np.ndarray               # (N,) index of the source frame/view
    cams: dict = field(default_factory=dict)   # frame index -> 4x4 T_wc

    def __len__(self):
        return len(self.xyz)

    def subset(self, m: np.ndarray) -> "OrientedCloud":
        return OrientedCloud(self.xyz[m], self.normals[m], self.frame[m], self.cams)

    def transformed(self, T: np.ndarray) -> "OrientedCloud":
        R, t = T[:3, :3], T[:3, 3]
        return OrientedCloud(self.xyz @ R.T + t, self.normals @ R.T, self.frame,
                             {k: T @ v for k, v in self.cams.items()})

    @staticmethod
    def concat(clouds: list["OrientedCloud"]) -> "OrientedCloud":
        cams = {}
        for c in clouds:
            cams.update(c.cams)
        return OrientedCloud(np.concatenate([c.xyz for c in clouds]),
                             np.concatenate([c.normals for c in clouds]),
                             np.concatenate([c.frame for c in clouds]), cams)


def depth_to_cloud(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray, frame_id: int,
                   stride: int = 2, max_range: float = 5.0,
                   max_grazing_deg: float = 75.0) -> OrientedCloud:
    """Back-project an OpenCV-convention metric depth map with per-pixel normals.

    Normals come from finite differences on the back-projected grid. Pixels seen
    at grazing angles are dropped: their depth error grows ~1/cos(angle) and they
    are the main source of "fuzzy" walls in the fused map.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    dx = X[1:-1, 2:] - X[1:-1, :-2]
    dy = X[2:, 1:-1] - X[:-2, 1:-1]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    P = X[1:-1, 1:-1]
    zc = z[1:-1, 1:-1]
    # orient normals towards the camera (origin)
    flip = np.sum(n * P, -1) > 0
    n[flip] *= -1
    ray = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9)
    cos_inc = -np.sum(n * ray, -1)
    # depth continuity: reject normals spanning a depth edge
    zc_n = [z[1:-1, 2:], z[1:-1, :-2], z[2:, 1:-1], z[:-2, 1:-1]]
    cont = np.all([np.abs(a - zc) < 0.05 * zc + 0.02 for a in zc_n], axis=0)
    ok = ((zc > 0.15) & (zc < max_range) & cont & (nn[..., 0] > 0)
          & (cos_inc > np.cos(np.deg2rad(max_grazing_deg)))
          & np.all([a > 0 for a in zc_n], axis=0))
    sel = np.zeros_like(ok)
    sel[::stride, ::stride] = True
    ok &= sel
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    return OrientedCloud(P[ok] @ R.T + t, n[ok] @ R.T,
                         np.full(int(ok.sum()), frame_id, np.int32), {frame_id: T_wc})


def voxel_downsample(c: OrientedCloud, voxel: float = 0.02) -> OrientedCloud:
    """Average points/normals per voxel; keeps the frame id of the first hit."""
    key = np.floor(c.xyz / voxel).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    xyz = np.stack([np.bincount(inv, c.xyz[:, i]) for i in range(3)], 1) / cnt[:, None]
    nrm = np.stack([np.bincount(inv, c.normals[:, i]) for i in range(3)], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    return OrientedCloud(xyz, nrm, c.frame[first], c.cams)
