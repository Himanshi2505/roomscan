import numpy as np

from roomscan.drift import vertical_offsets


def test_vertical_offsets_follow_injected_drift():
    """Floor seen at +3 cm during seconds 4-8 (vertical pose drift) -> offsets
    recover it there and ~0 elsewhere; a window with no floor is interpolated."""
    rng = np.random.default_rng(0)
    frame_t = {k: k * 0.1 for k in range(120)}            # 12 s at 10 fps
    true_dy = {k: (0.03 if 4.0 <= t < 8.0 else 0.0) for k, t in frame_t.items()}
    fr, y = [], []
    for k, t in frame_t.items():
        if 10.0 <= t < 12.0:                               # last window: looking at the ceiling
            continue
        n = 200
        fr.append(np.full(n, k))
        y.append(true_dy[k] + rng.normal(0, 0.005, n))     # floor at 0 + drift + noise
    fr, y = np.concatenate(fr), np.concatenate(y)
    dy = vertical_offsets(fr, y, frame_t, floor_y=0.0, window_s=2.0)
    assert abs(dy[50] - 0.03) < 0.003                     # inside the drifted stretch
    assert abs(dy[10]) < 0.003                            # before it
    assert abs(dy[115] - dy[95]) < 0.003                  # floor-less window interpolated (held)


def test_vertical_offsets_without_floor_is_zero():
    dy = vertical_offsets(np.array([0, 1]), np.array([2.5, 2.5]), {0: 0.0, 1: 0.1}, floor_y=0.0)
    assert dy == {0: 0.0, 1: 0.0}
