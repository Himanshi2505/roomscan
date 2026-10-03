"""LiDAR tier: Stray Scanner depth + ARKit poses -> OrientedCloud."""
from __future__ import annotations

import numpy as np

from roomscan.geometry.cloud import OrientedCloud, depth_to_cloud, voxel_downsample
from roomscan.io import strayscanner as ss


def build_cloud(cap: ss.LidarCapture, every: int = 3, stride: int = 2,
                min_conf: int = 2, voxel: float = 0.02,
                poses: dict[int, np.ndarray] | None = None) -> OrientedCloud:
    """Fuse depth frames. `poses` overrides ARKit poses (used after drift correction)."""
    parts = []
    for f in cap.depth_frames[::every]:
        d, K = cap.load_depth(f, min_conf)
        T = poses.get(f.index, f.T_wc) if poses else f.T_wc
        parts.append(depth_to_cloud(d, K, T, f.index, stride=stride))
    return voxel_downsample(OrientedCloud.concat(parts), voxel)
