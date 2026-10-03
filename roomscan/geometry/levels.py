"""Floor / ceiling heights and the dominant (Manhattan) wall direction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.geometry.cloud import OrientedCloud


@dataclass
class Level:
    y: float            # metres, world +Y
    sigma: float        # 1-sigma statistical uncertainty of y
    support: int        # number of points on the plane


def _plane_height(y: np.ndarray, pick: str, bin_m: float = 0.01,
                  min_frac: float = 0.15, min_support: int = 2000) -> Level | None:
    """Height of the lowest ('floor') or highest ('ceiling') dominant horizontal plane.

    'Dominant' = histogram peak holding at least min_frac of the largest peak, so
    a tabletop never wins over the floor but a floor partly hidden by furniture
    still does. The peak is refined by a trimmed mean of points within +-2 cm.
    """
    if len(y) < 200:
        return None
    edges = np.arange(y.min() - bin_m, y.max() + 2 * bin_m, bin_m)
    h, _ = np.histogram(y, edges)
    h = np.convolve(h, [1, 2, 1], "same")
    peaks = [i for i in range(1, len(h) - 1)
             if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] >= min_frac * h.max()]
    if not peaks:
        return None
    i = peaks[0] if pick == "floor" else peaks[-1]
    y0 = 0.5 * (edges[i] + edges[i + 1])
    near = y[np.abs(y - y0) < 0.02]
    lo, hi = np.percentile(near, [10, 90])
    core = near[(near >= lo) & (near <= hi)]
    if len(near) < min_support:      # a few stray points, e.g. a shelf top seen from below
        return None
    return Level(float(core.mean()), float(core.std() / np.sqrt(len(core))), len(near))


def floor_ceiling(c: OrientedCloud) -> tuple[Level | None, Level | None]:
    up = c.normals[:, 1] > 0.9
    down = c.normals[:, 1] < -0.9
    floor = _plane_height(c.xyz[up, 1], "floor")
    ceil = _plane_height(c.xyz[down, 1], "ceiling")
    if floor and ceil and ceil.y - floor.y < 1.8:   # implausible: not a real ceiling
        ceil = None
    return floor, ceil


def manhattan_yaw(c: OrientedCloud) -> float:
    """Yaw (radians, about +Y) that aligns dominant wall normals with world X/Z.

    Uses the 4-fold symmetric circular mean of wall-normal angles, weighted by
    horizontality, so walls at 90 deg to each other vote for the same answer.
    """
    n = c.normals
    wall = np.abs(n[:, 1]) < 0.2
    th = np.arctan2(n[wall, 2], n[wall, 0])
    w = 1.0 - np.abs(n[wall, 1])
    z = np.sum(w * np.exp(4j * th))
    return float(np.angle(z) / 4.0)


def yaw_matrix(yaw: float) -> np.ndarray:
    """4x4 rotation about +Y by -yaw (rotates the dominant wall normal onto +X)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, -s, s, c
    return T
