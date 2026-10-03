"""Photo tier: place per-room plans (each in its own frame) into one property plan.

Every room plan is Manhattan-aligned, so rooms differ by a rotation that is a
multiple of 90 deg plus a translation. Rooms are joined through doors: the
shared door's centres coincide across a wall of thickness WALL_T and its
normals are opposite. Which door pairs with which comes from, in order:

  1. photo evidence: an unregistered (doorway) photo of room A that registers
     into room B's photos tells us which of B's doors it was taken from;
     A's door is then the width-compatible one giving a valid placement;
  2. jigsaw: try every width-compatible door pair and keep the placement with
     no overlap and the most shared wall length (rooms in a home touch).

Placement uncertainty (unknown wall thickness, door-centre error) is added to
the intervals of the inter-room relationship, not to per-room dimensions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

WALL_T = 0.12             # typical interior partition (m); +-0.05 in the error budget
DOOR_W_TOL = 0.15
MAX_OVERLAP = 0.05        # m2


@dataclass
class Placement:
    k: int                 # quarter turns
    t: np.ndarray          # translation (2,)

    def apply(self, p):
        p = np.asarray(p, float)
        R = _rot(self.k)
        return p @ R.T + self.t

    def apply_dir(self, d):
        return np.asarray(d, float) @ _rot(self.k).T


def _rot(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s], [s, c]], float)


def transform_room(room: dict, P: Placement, prefix: str) -> dict:
    r = copy.deepcopy(room)
    ren = lambda i: f"{prefix}.{i}" if i else i
    r["id"] = ren(r["id"])
    r["polygon"] = np.round(P.apply(r["polygon"]), 3).tolist()
    for w in r["walls"]:
        w["id"] = ren(w["id"])
        w["start"] = np.round(P.apply(w["start"]), 3).tolist()
        w["end"] = np.round(P.apply(w["end"]), 3).tolist()
        w["openings"] = [ren(o) for o in w["openings"]]
    for o in r["openings"]:
        o["id"], o["wall_id"] = ren(o["id"]), ren(o["wall_id"])
        o["center"] = np.round(P.apply(o["center"]), 3).tolist()
        o["outward"] = np.round(P.apply_dir(o["outward"]), 3).tolist()
        o["connects_to"] = None
    return r


def _doors(room):
    return [o for o in room["openings"] if o["kind"] in ("door", "passage")]


def place_via_doors(room_a: dict, door_a: dict, room_b: dict, door_b: dict) -> Placement | None:
    """Placement of B (in its own frame) so door_b meets door_a (A already placed)."""
    na, nb = np.array(door_a["outward"]), np.array(door_b["outward"])
    for k in range(4):
        if np.allclose(_rot(k) @ nb, -na, atol=1e-3):
            t = np.array(door_a["center"]) + WALL_T * na - _rot(k) @ np.array(door_b["center"])
            return Placement(k, t)
    return None


def _score(shape_b: Polygon, placed: list[Polygon]) -> tuple[float, float]:
    overlap = sum(shape_b.intersection(p).area for p in placed)
    contact = sum(shape_b.buffer(WALL_T + 0.1).intersection(p).area for p in placed)
    return overlap, contact


def stitch(rooms: list[dict], names: list[str], links: list[tuple[int, int, str | None]]) -> tuple[list[dict], list[dict], list[str]]:
    """rooms[i]: single-room plan dict (rooms[i]['rooms'][0] is used);
    links: (i, j, door_id_in_j or None) photo evidence that i and j are adjacent.
    Returns placed room dicts, adjacency list, warnings."""
    R = [r["rooms"][0] for r in rooms]
    n = len(R)
    warnings = []
    order = sorted(range(n), key=lambda i: -R[i]["floor_area"]["value"])
    root = order[0]
    placed = {root: Placement(0, np.zeros(2))}
    out = {root: transform_room(R[root], placed[root], names[root])}
    adjacency = []
    used_doors = set()

    def try_place(j, cands):
        best = None
        for i, da, db in cands:
            if (i, da["id"]) in used_doors:
                continue
            # door_a must be expressed in placed coordinates
            da_w = next(o for o in out[i]["openings"] if o["id"] == f"{names[i]}.{da['id']}")
            P = place_via_doors(out[i], da_w, R[j], db)
            if P is None:
                continue
            shp = Polygon(P.apply(R[j]["polygon"]))
            ov, ct = _score(shp, [Polygon(out[q]["polygon"]) for q in out])
            if ov > MAX_OVERLAP:
                continue
            key = (-ct, abs(da["width"]["value"] - db["width"]["value"]))
            if best is None or key < best[0]:
                best = (key, i, da_w, db, P)
        return best

    pending = [i for i in order if i != root]
    stalled = 0
    while pending and stalled < len(pending):
        j = pending.pop(0)
        cands = []
        # 1) photo evidence: links naming one of j's doors (or any door)
        ev = [(i, d) for (i, jj, d) in links if jj == j and i in placed] + \
             [(jj, None) for (i, jj, d) in links if i == j and jj in placed]
        for i, d in ev:
            dbs = [o for o in _doors(R[j]) if d is None or o["id"] == d]
            for db in dbs:
                for da in _doors(R[i]):
                    if abs(da["width"]["value"] - db["width"]["value"]) < DOOR_W_TOL:
                        cands.append((i, da, db))
        how = "photo"
        best = try_place(j, cands) if cands else None
        if best is None:              # 2) jigsaw over all placed rooms
            how = "jigsaw"
            cands = [(i, da, db) for i in placed for da in _doors(R[i]) for db in _doors(R[j])
                     if abs(da["width"]["value"] - db["width"]["value"]) < DOOR_W_TOL]
            best = try_place(j, cands)
        if best is None:
            pending.append(j)
            stalled += 1
            continue
        stalled = 0
        _, i, da_w, db, P = best
        placed[j] = P
        out[j] = transform_room(R[j], P, names[j])
        db_w = next(o for o in out[j]["openings"] if o["id"] == f"{names[j]}.{db['id']}")
        da_w["connects_to"], db_w["connects_to"] = out[j]["id"], out[i]["id"]
        used_doors.add((i, da_w["id"].split(".", 1)[1]))
        used_doors.add((j, db["id"]))
        adjacency.append({"rooms": sorted([out[i]["id"], out[j]["id"]]), "via": da_w["id"],
                          "kind": da_w["kind"], "evidence": how})
    for j in pending:
        warnings.append(f"room {names[j]} could not be connected (no compatible door); "
                        "placed to the side, unconnected")
        xs = np.concatenate([np.array(o["polygon"])[:, 0] for o in out.values()])
        P = Placement(0, np.array([xs.max() + 1.0 - np.min(np.array(R[j]["polygon"])[:, 0]), 0.0]))
        out[j] = transform_room(R[j], P, names[j])
    return [out[i] for i in range(n)], adjacency, warnings
