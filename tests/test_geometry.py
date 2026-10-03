"""Synthetic box rooms through the full geometry path (no sensor, no model)."""
import numpy as np

from roomscan.geometry.cloud import OrientedCloud
from roomscan.plan import build_plan

RNG = np.random.default_rng(0)
STEP = 0.02
NOISE = 0.003


def _plane(axis, pos, a_rng, b_rng, normal, hole=None):
    """Points on an axis-aligned plane; hole = (a0, a1, b0, b1) cut out."""
    a = np.arange(*a_rng, STEP)
    b = np.arange(*b_rng, STEP)
    A, B = np.meshgrid(a, b)
    A, B = A.ravel(), B.ravel()
    if hole:
        keep = ~((A > hole[0]) & (A < hole[1]) & (B > hole[2]) & (B < hole[3]))
        A, B = A[keep], B[keep]
    P = np.zeros((len(A), 3))
    if axis == "x":
        P[:, 0], P[:, 2], P[:, 1] = pos, A, B
    elif axis == "z":
        P[:, 2], P[:, 0], P[:, 1] = pos, A, B
    else:
        P[:, 1], P[:, 0], P[:, 2] = pos, A, B
    P += RNG.normal(0, NOISE, P.shape)
    return P, np.tile(normal, (len(P), 1))


def box_room(L=4.0, W=3.0, H=2.5, door=(1.0, 1.85), with_beyond=True):
    parts = [
        _plane("y", 0.0, (0, L), (0, W), [0, 1, 0]),                       # floor
        _plane("y", H, (0, L), (0, W), [0, -1, 0]),                        # ceiling
        _plane("x", 0.0, (0, W), (0, H), [1, 0, 0]),
        _plane("x", L, (0, W), (0, H), [-1, 0, 0]),
        _plane("z", W, (0, L), (0, H), [0, 0, -1]),
        _plane("z", 0.0, (0, L), (0, H), [0, 0, 1], hole=(door[0], door[1], -1, 2.05)),
    ]
    if with_beyond:   # corridor wall 1.2 m behind the door, seen through it
        parts.append(_plane("z", -1.2, (door[0] - 0.3, door[1] + 0.3), (0, 2.0), [0, 0, 1]))
    xyz = np.concatenate([p for p, _ in parts])
    n = np.concatenate([q for _, q in parts]).astype(float)
    cams = {}
    for k, (cx, cz) in enumerate([(1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0), (2.0, 1.5)]):
        T = np.eye(4)
        T[:3, 3] = [cx, 1.4, cz]
        cams[k] = T
    # assign each point to the nearest camera (as if it measured it)
    cpos = np.array([T[:3, 3] for T in cams.values()])
    frame = np.argmin(((xyz[:, None, [0, 2]] - cpos[None, :, [0, 2]]) ** 2).sum(-1), 1).astype(np.int32)
    return OrientedCloud(xyz, n, frame, cams)


def test_box_room_dimensions_and_door():
    plan = build_plan(box_room(), "lidar", single_room=True)
    assert len(plan["rooms"]) == 1
    r = plan["rooms"][0]
    lens = sorted(w["length"]["value"] for w in r["walls"])
    assert np.allclose(lens, [3.0, 3.0, 4.0, 4.0], atol=0.01), lens
    assert abs(r["floor_area"]["value"] - 12.0) < 0.08
    assert abs(r["ceiling_height"]["value"] - 2.5) < 0.01
    doors = [o for o in r["openings"] if o["kind"] == "door"]
    assert len(doors) == 1, r["openings"]
    assert abs(doors[0]["width"]["value"] - 0.85) < 0.03
    # every measurement carries an interval that contains the value
    for w in r["walls"]:
        lo, hi = w["length"]["ci90"]
        assert lo <= w["length"]["value"] <= hi


def test_occluded_wall_is_not_an_opening():
    """Missing wall with nothing seen behind it (furniture shadow) must not be a door."""
    c = box_room(with_beyond=False)
    plan = build_plan(c, "lidar", single_room=True)
    assert not [o for o in plan["rooms"][0]["openings"] if o["kind"] == "door"]
