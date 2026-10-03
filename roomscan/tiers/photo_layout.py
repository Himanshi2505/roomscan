"""Photo tier, layout-first: one wall measured per photo, no cross-photo matching.

Protocol (docs/CAPTURE_PROTOCOL.md): 0.5x ultra-wide, stand with your back to
the middle of a wall and photograph the opposite wall, going clockwise round
the room; then one photo from each doorway.

Per wall photo (single image, MoGe metric point map, gravity + Manhattan):
  facing wall   the largest wall face in front of the camera, normal towards it
  length        distance between the two side-wall faces that meet it (both
                corners seen); if a corner is not seen the length is a lower
                bound and the photo is marked partial
  depth         camera -> facing wall distance (camera stands ~INSET from the
                wall behind it), which gives an estimate of the *other* room
                dimension and pairs the photos up
  openings      doors/windows on the facing wall with the same see-through
                test as the other tiers
Room            rectangle; photo k faces wall k (clockwise). Each dimension is
                the inverse-variance mean of every photo that measured it
                (lengths of facing walls + depths), so opposite photos give two
                independent estimates and their disagreement widens the interval.
Textureless white walls, which defeat feature matching, are exactly the case
this handles: it needs corners, not texture.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roomscan import uncertainty as U
from roomscan.geometry.cloud import OrientedCloud
from roomscan.geometry.levels import floor_ceiling, manhattan_yaw, yaw_matrix
from roomscan.geometry.openings import edge_openings
from roomscan.geometry.polygon import Edge
from roomscan.geometry.walls import extract_faces
from roomscan.tiers import photo

INSET = 0.25          # camera ahead of the wall behind the photographer (m)
INSET_SIG = 0.10
CORNER_TOL = 0.20     # side face must reach within this of the facing wall


@dataclass
class WallObs:
    name: str
    length: float
    length_sig: float
    complete: bool
    depth: float              # camera -> facing wall
    ceiling: float | None     # floor -> ceiling
    openings: list = field(default_factory=list)   # (kind, s_lo, s_hi, sig, head, sill), s from left corner


def observe(v: photo.View) -> WallObs | None:
    v.T, v.scale = np.eye(4), 1.0
    c = photo.view_cloud(v, 0, stride=1)
    M = yaw_matrix(manhattan_yaw(c))
    c = c.transformed(M)
    fl, ce = floor_ceiling(c)
    if fl is None:
        return None
    faces = extract_faces(c, fl.y, ce.y if ce else None, min_support=80, min_len=0.4)
    fwd = M[:3, :3] @ v.R_g @ np.array([0, 0, 1.0])
    ax = "x" if abs(fwd[0]) > abs(fwd[2]) else "z"
    k = 0 if ax == "x" else 2
    sgn = -int(np.sign(fwd[k]))          # facing wall's normal points back at the camera
    cand = [f for f in faces if f.axis == ax and f.sign == sgn and abs(f.pos) > 0.8
            and np.sign(f.pos) == np.sign(fwd[k])]
    if not cand:
        return None
    wall = max(cand, key=lambda f: f.length * f.support)
    other = "z" if ax == "x" else "x"
    # side walls: perpendicular faces whose extent reaches the facing wall
    reach = lambda f: min(abs(f.lo - wall.pos), abs(f.hi - wall.pos)) < CORNER_TOL or \
        (min(f.lo, f.hi) < wall.pos < max(f.lo, f.hi))
    left = [f for f in faces if f.axis == other and f.sign == 1 and reach(f)
            and abs(f.pos - wall.lo) < 0.4]
    right = [f for f in faces if f.axis == other and f.sign == -1 and reach(f)
             and abs(f.pos - wall.hi) < 0.4]
    lo = max(left, key=lambda f: f.support).pos if left else wall.lo
    hi = min(right, key=lambda f: f.support).pos if right else wall.hi
    complete = bool(left and right)
    sig = np.hypot(0.01 if left else 0.05, 0.01 if right else 0.05)
    H = (ce.y - fl.y) if ce else None
    e = Edge(ax, wall.pos, wall.pos_sigma, sgn, wall, lo, hi)
    ops = []
    for o in edge_openings(c, e, 0, fl.y):
        ops.append((o.kind, o.s_lo - lo, o.s_hi - lo, o.width_sigma, o.head, o.sill))
    return WallObs(v.name, float(hi - lo), float(sig), complete, float(abs(wall.pos)), H, ops)


def _combine(vals: list[tuple[float, float]]) -> tuple[float, float]:
    """Inverse-variance mean; sigma inflated by the observed scatter (Birge ratio)."""
    x = np.array([a for a, _ in vals])
    s = np.array([b for _, b in vals])
    w = 1 / s ** 2
    m = float(np.sum(w * x) / np.sum(w))
    se = float(np.sqrt(1 / np.sum(w)))
    if len(x) > 1:
        chi2 = np.sum(w * (x - m) ** 2) / (len(x) - 1)
        se *= max(1.0, float(np.sqrt(chi2)))
    return m, se


def room_plan(views: list[photo.View], room_id: str = "R1") -> tuple[dict | None, list[str]]:
    model = U.load_model("photo")
    obs, notes = [], []
    for v in views:
        o = observe(v)
        if o is None:
            notes.append(f"{v.name}: no facing wall found (doorway photo or too close)")
        else:
            obs.append(o)
    if not obs:
        return None, notes
    # photo k faces wall k clockwise: even k -> dimension A, odd k -> dimension B
    A, B = [], []
    for k, o in enumerate(obs):
        scale_sig = model["scale_rel"]
        ls = np.hypot(o.length_sig, scale_sig * o.length) * (1 if o.complete else 3)
        ds = np.hypot(INSET_SIG, scale_sig * o.depth)
        (A if k % 2 == 0 else B).append((o.length, ls))
        (B if k % 2 == 0 else A).append((o.depth + INSET, ds))
    if not B:
        notes.append("only one wall photographed: second dimension from depth only")
    a, sa = _combine(A)
    b, sb = _combine(B) if B else (obs[0].depth + INSET, INSET_SIG)
    Hs = [o.ceiling for o in obs if o.ceiling]
    H = float(np.median(Hs)) if Hs else None
    sH = float(np.hypot(np.std(Hs) / np.sqrt(len(Hs)) if len(Hs) > 1 else 0.02,
                        model["scale_rel"] * H)) if H else None
    # rectangle, clockwise from (0,0): wall k is faced by photo k
    V = np.array([[0, 0], [a, 0], [a, b], [0, b]], float)
    sides = [(V[3], V[2]), (V[2], V[1]), (V[1], V[0]), (V[0], V[3])]   # N, E, S, W
    lens = [(a, sa), (b, sb), (a, sa), (b, sb)]
    outward = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    walls, openings = [], []
    for i, ((p, q), (L, sL)) in enumerate(zip(sides, lens)):
        wid = f"{room_id}.W{i + 1}"
        walls.append({"id": wid, "start": p.tolist(), "end": q.tolist(),
                      "length": U.measure(L, sL, model),
                      "height": U.measure(H, sH, model) if H else None,
                      "measured": True, "openings": []})
    for k, o in enumerate(obs[:4]):
        p, q = sides[k]
        d = (q - p) / max(np.linalg.norm(q - p), 1e-9)
        for kind, s0, s1, sg, head, sill in o.openings:
            oid = f"{room_id}.O{len(openings) + 1}"
            c = p + d * (0.5 * (s0 + s1))
            openings.append({"id": oid, "kind": kind, "wall_id": walls[k]["id"],
                             "width": U.measure(s1 - s0, np.hypot(sg, model["sigma_face"]), model),
                             "height": U.measure(head, 0.05, model) if head else None,
                             "sill": U.measure(sill, 0.05, model) if sill is not None else None,
                             "center": c.round(3).tolist(), "outward": outward[k],
                             "connects_to": None})
            walls[k]["openings"].append(oid)
    area = a * b
    room = {"id": room_id, "name": room_id, "polygon": V.round(3).tolist(),
            "floor_area": U.measure(area, np.hypot(a * sb, b * sa), model, 2),
            "ceiling_height": U.measure(H, sH, model) if H else None,
            "ceiling_observed": H is not None, "walls": walls, "openings": openings}
    plan = {"tier": "photo", "calibrated": model["calibrated"], "rooms": [room],
            "adjacency": [], "footprint_area": room["floor_area"],
            "levels": {"floor_y": 0.0, "ceiling_y": H}, "layout": "rectangle-from-wall-photos",
            "wall_photos": len(obs)}
    return plan, notes
