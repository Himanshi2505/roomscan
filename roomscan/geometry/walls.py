"""Axis-aligned wall segments from an oriented cloud in the Manhattan frame.

After rotating by the Manhattan yaw, wall normals point along +-X or +-Z. Points
with a horizontal normal along an axis vote for a wall line at their coordinate
on that axis. Each histogram peak is refined to a sub-centimetre position (median
of thousands of points), then split into occupied extents along the wall.

A thin interior wall shows up as two parallel faces with opposite normals; the
gap between them is the wall thickness. Rooms are measured face-to-face
(interior dimensions), which is what a tape/laser measures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.geometry.cloud import OrientedCloud


@dataclass
class WallFace:
    axis: str           # 'x': face at constant x (normal +-X); 'z': constant z
    sign: int           # +1 normal points to +axis (room lies on the + side)
    pos: float          # coordinate of the face along `axis`
    pos_sigma: float    # 1-sigma of pos (statistical)
    lo: float           # extent along the other horizontal axis
    hi: float
    support: int

    @property
    def length(self) -> float:
        return self.hi - self.lo


def _runs(occ: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Index runs of True, bridging gaps of up to max_gap False cells."""
    idx = np.flatnonzero(occ)
    if len(idx) == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > max_gap + 1:
            runs.append((s, p))
            s = i
        p = i
    runs.append((s, p))
    return runs


def wall_points(c: OrientedCloud, floor_y: float, ceil_y: float | None,
                axis_tol: float = 0.9) -> dict:
    """Split wall-height points with axis-aligned horizontal normals by (axis, sign).

    Height band starts 0.25 m above the floor (skirting boards, clutter) and ends
    0.15 m under the ceiling (crown moulding); without a ceiling, 2.0 m is used.
    """
    top = (ceil_y - 0.15) if ceil_y is not None else floor_y + 2.0
    y = c.xyz[:, 1]
    band = (y > floor_y + 0.25) & (y < top)
    out = {}
    for axis, k in (("x", 0), ("z", 2)):
        for sign in (1, -1):
            m = band & (sign * c.normals[:, k] > axis_tol)
            out[(axis, sign)] = c.xyz[m]
    return out


def extract_faces(c: OrientedCloud, floor_y: float, ceil_y: float | None,
                  bin_m: float = 0.01, min_support: int = 150,
                  min_len: float = 0.25, gap_m: float = 0.12) -> list[WallFace]:
    faces = []
    for (axis, sign), P in wall_points(c, floor_y, ceil_y).items():
        if len(P) < min_support:
            continue
        k, o = (0, 2) if axis == "x" else (2, 0)
        a, b = P[:, k], P[:, o]
        edges = np.arange(a.min() - bin_m, a.max() + 2 * bin_m, bin_m)
        h, _ = np.histogram(a, edges)
        h = np.convolve(h, [1, 2, 3, 2, 1], "same")
        used = np.zeros(len(a), bool)
        for i in np.argsort(h)[::-1]:
            if h[i] < min_support:
                break
            c0 = 0.5 * (edges[i] + edges[i + 1])
            near = (np.abs(a - c0) < 0.04) & ~used
            if near.sum() < min_support:
                continue
            used |= np.abs(a - c0) < 0.06      # suppress neighbouring bins
            an, bn = a[near], b[near]
            # split along the wall into occupied extents (5 cm cells)
            cell = 0.05
            b0 = bn.min()
            occ = np.bincount(((bn - b0) / cell).astype(int))
            for s, e in _runs(occ >= 3, int(gap_m / cell)):
                seg = (bn >= b0 + s * cell) & (bn < b0 + (e + 1) * cell)
                if seg.sum() < min_support:
                    continue
                bs, as_ = bn[seg], an[seg]
                lo, hi = np.percentile(bs, [0.5, 99.5])
                if hi - lo < min_len:
                    continue
                med = np.median(as_)
                inl = as_[np.abs(as_ - med) < 0.02]
                faces.append(WallFace(axis, sign, float(np.mean(inl)),
                                      float(np.std(inl) / np.sqrt(len(inl))),
                                      float(lo), float(hi), int(seg.sum())))
    return faces
