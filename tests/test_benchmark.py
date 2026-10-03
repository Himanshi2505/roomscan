import numpy as np

from benchmark.evaluate import calibrate, gates, match_openings, match_walls, repeatability, score_capture


def m(v, s=0.01):
    return {"value": v, "sigma": s, "ci90": [v - 1.645 * s, v + 1.645 * s]}


def test_match_walls_cyclic_and_reversed():
    gt = [3.0, 2.0, 3.0, 2.0]
    pred = [2.01, 3.01, 1.99, 2.99]
    pairs = match_walls(pred, gt)          # rectangles have two equally good alignments
    assert len(pairs) == 4 and all(abs(pred[i] - gt[j]) < 0.02 for i, j in pairs)
    pairs = match_walls([2.0, 0.1, 3.0, 2.0, 3.0], gt)        # 0.1 m jog ignored
    assert all(i != 1 for i, _ in pairs) and len(pairs) == 4


def test_openings_missed_and_phantom_count_as_fails():
    pred = [{"kind": "door", "width": m(0.81)}, {"kind": "door", "width": m(0.60)}]
    gt = [{"kind": "door", "width": 0.80}, {"kind": "window", "width": 1.2}]
    pairs, missed, phantom = match_openings(pred, gt)
    assert pairs == [(0, 0)] and missed == [1] and phantom == [1]


def _res(lengths, ceil, ops):
    poly = [[0, 0], [lengths[0], 0], [lengths[0], lengths[1]], [0, lengths[1]]]
    walls = [{"id": f"R1.W{i}", "length": m(L)} for i, L in enumerate(lengths)]
    return {"rooms": [{"id": "R1", "polygon": poly, "walls": walls, "ceiling_height": m(ceil),
                       "openings": ops}], "adjacency": [], "footprint_area": m(lengths[0] * lengths[1]),
            "timing_s": {"total": 1.0}}


GT = {"rooms": {"k": {"walls": [{"id": "N", "length": 3.0}, {"id": "E", "length": 2.0},
                                {"id": "S", "length": 3.0}, {"id": "W", "length": 2.0}],
                      "ceiling_height": 2.6, "openings": [{"kind": "door", "width": 0.8}]}},
      "adjacency": [], "footprint_area": 6.0}
CAP = {"room_map": {"R1": "k"}}


def test_gates_pass_and_fail():
    s = score_capture(CAP, _res([3.005, 2.0, 3.0, 2.004], 2.61, [{"kind": "door", "width": m(0.81)}]), GT)
    g = gates("video", s)
    assert g["opening_widths"]["pass"] and g["ceiling_height"]["pass"] and g["wall_lengths"]["pass"]
    s = score_capture(CAP, _res([3.2, 2.0, 3.0, 2.0], 2.64, []), GT)
    g = gates("video", s)
    assert not g["opening_widths"]["pass"] and not g["ceiling_height"]["pass"] and not g["wall_lengths"]["pass"]


def test_repeatability_and_ceiling_verdicts():
    a = score_capture(CAP, _res([3.0, 2.0, 3.0, 2.0], 2.63, []), GT)
    b = score_capture(CAP, _res([3.004, 2.0, 3.0, 2.02], 2.632, []), GT)
    r = repeatability(a, b)
    assert not r["pass"] and r["ceiling"][0]["verdict"] == "repeatable-but-biased"


def test_calibration_factor():
    rng = np.random.default_rng(0)
    cal = [(e, 0.01) for e in rng.normal(0, 0.02, 400)]      # true sigma 2x the claimed one
    assert 1.7 < calibrate({"lidar": cal})["lidar"]["k"] < 2.3
