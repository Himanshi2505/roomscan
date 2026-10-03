"""Door / window / passage detection on each room edge.

For an edge we build two along-wall column profiles (2 cm columns):
  wall[s, h]    hits within +-5 cm of the wall plane (the wall surface)
  beyond[s, h]  hits >10 cm *behind* the plane, measured from a camera standing
                inside the room, i.e. the sensor saw *through* the wall there.
A column is open at height h if the wall is missing and something was seen
through it. Missing wall with nothing seen through = occlusion (furniture in
front of the wall), which is NOT an opening. That rule is what keeps sofas and
wardrobes from becoming phantom doors.

  door/passage : open over most of [0.15, 1.85] m above the floor
  window       : wall present below 0.6 m (sill), open in [1.0, 1.5] m

Width is measured jamb-to-jamb from wall hits flanking the run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.geometry.cloud import OrientedCloud
from roomscan.geometry.polygon import Edge

COL = 0.02
H_BINS = np.arange(0.0, 2.45, 0.05)


@dataclass
class Opening:
    kind: str            # 'door' | 'window' | 'passage'
    edge_index: int
    s_lo: float          # jamb positions along the edge (world coordinate)
    s_hi: float
    width_sigma: float
    sill: float | None   # m above floor (windows)
    head: float | None   # m above floor, None if not observed
    see_through: int     # supporting hits behind the opening

    @property
    def width(self) -> float:
        return self.s_hi - self.s_lo


def _runs(b: np.ndarray, max_gap: int = 1):
    idx = np.flatnonzero(b)
    out = []
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > max_gap + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def edge_openings(c: OrientedCloud, e: Edge, ei: int, floor_y: float,
                  cam_side_ok: np.ndarray | None = None) -> list[Opening]:
    if e.s_hi - e.s_lo < 0.5:
        return []
    k, o = (0, 2) if e.axis == "x" else (2, 0)
    d = (c.xyz[:, k] - e.coord) * e.inward          # >0 inside room, <0 behind wall
    s = c.xyz[:, o]
    h = c.xyz[:, 1] - floor_y
    inrange = (s > e.s_lo) & (s < e.s_hi) & (h > 0) & (h < H_BINS[-1])
    # camera that measured each point must stand inside the room (in front of the wall)
    if c.cams:
        cam_d = np.array([(c.cams[f][k, 3] - e.coord) * e.inward if f in c.cams else 1.0
                          for f in c.frame[inrange]])
        front = np.zeros(len(c.xyz), bool)
        front[np.flatnonzero(inrange)[cam_d > 0.2]] = True
    else:
        front = inrange
    wall_m = inrange & (np.abs(d) < 0.05)
    beyond_m = inrange & front & (d < -0.10) & (d > -6.0)
    ncol = int(np.ceil((e.s_hi - e.s_lo) / COL))

    def prof(m):
        ci = np.clip(((s[m] - e.s_lo) / COL).astype(int), 0, ncol - 1)
        hi = np.clip(np.digitize(h[m], H_BINS) - 1, 0, len(H_BINS) - 2)
        P = np.zeros((ncol, len(H_BINS) - 1), np.int32)
        np.add.at(P, (ci, hi), 1)
        return P

    W, B = prof(wall_m) > 0, prof(beyond_m) > 0
    # smooth over 3 columns: LiDAR at 256x192 leaves 1-column holes in walls
    Wk = np.maximum.reduce([W, np.roll(W, 1, 0), np.roll(W, -1, 0)])
    hb = lambda lo, hi: (H_BINS[:-1] >= lo) & (H_BINS[:-1] < hi)
    door_band, sill_band, win_band = hb(0.15, 1.85), hb(0.1, 0.6), hb(1.0, 1.5)
    open_ = ~Wk & B
    door_col = open_[:, door_band].mean(1) > 0.6
    win_col = (open_[:, win_band].mean(1) > 0.6) & (Wk[:, sill_band].mean(1) > 0.5)
    out = []
    for kind, cols, min_w in (("door", door_col, 0.55), ("window", win_col & ~door_col, 0.30)):
        for a, b in _runs(cols):
            if (b - a + 1) * COL < min_w:
                continue
            lo, hi = e.s_lo + a * COL, e.s_lo + (b + 1) * COL
            lo, hi, sig = _refine_jambs(s[wall_m], h[wall_m], lo, hi)
            if hi - lo < min_w or hi - lo > 3.5:
                continue
            col_open = open_[a:b + 1]
            rows = np.flatnonzero(col_open.mean(0) > 0.5)
            head = float(H_BINS[rows.max() + 1]) if len(rows) and rows.max() + 2 < len(H_BINS) - 1 else None
            if head is not None:
                # lintel underside: lowest wall hit above 1.6 m inside the jambs
                # (5 cm bins are too coarse when the head anchors metric scale)
                inside = wall_m & (s > lo + 0.05) & (s < hi - 0.05) & (h > 1.6)
                if inside.sum() >= 10:
                    head = float(np.percentile(h[inside], 2))
            sill = float(H_BINS[rows.min()]) if kind == "window" and len(rows) else None
            kk = "passage" if kind == "door" and hi - lo > 1.3 else kind
            out.append(Opening(kk, ei, lo, hi, sig, sill, head,
                               int(B[a:b + 1].sum())))
    return out


def _refine_jambs(s_wall, h_wall, lo, hi, band=(0.3, 1.8), win=0.12):
    """Jamb = extreme wall hit just outside each end of the run (2nd/98th pct)."""
    m = (h_wall > band[0]) & (h_wall < band[1])
    sw = s_wall[m]
    left = sw[(sw > lo - win) & (sw < lo + 0.02)]
    right = sw[(sw < hi + win) & (sw > hi - 0.02)]
    sig = []
    if len(left) > 10:
        lo = float(np.percentile(left, 98))
        sig.append(np.std(left[left > lo - 0.03]) if (left > lo - 0.03).sum() > 3 else 0.01)
    else:
        sig.append(COL)
    if len(right) > 10:
        hi = float(np.percentile(right, 2))
        sig.append(np.std(right[right < hi + 0.03]) if (right < hi + 0.03).sum() > 3 else 0.01)
    else:
        sig.append(COL)
    return lo, hi, float(np.hypot(*sig))
