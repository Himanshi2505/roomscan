import numpy as np

from roomscan.mono import MonoView
from roomscan.tiers.video import too_close


def _mv(depth, mask=True):
    h, w = 8, 6
    P = np.zeros((h, w, 3), np.float32)
    P[..., 2] = depth
    return MonoView(P, np.zeros_like(P), np.full((h, w), mask), np.eye(3), np.zeros((h, w, 3), np.uint8))


def test_too_close():
    assert too_close(_mv(0.3))            # phone held against a wall
    assert not too_close(_mv(1.2))        # normal walking distance
    assert too_close(_mv(1.2, mask=False))  # no valid geometry at all


def _floor_view(cam_h=None):
    """Gravity-aligned camera (R_g = I); floor plane cam_h below it, or a wall only."""
    from roomscan.tiers.photo import View
    h, w = 60, 80
    rng = np.random.default_rng(0)
    P = np.zeros((h, w, 3), np.float32)
    N = np.zeros_like(P)
    if cam_h is not None:
        P[..., 0] = rng.uniform(-2, 2, (h, w))
        P[..., 2] = rng.uniform(0.5, 4, (h, w))
        P[..., 1] = -cam_h + rng.normal(0, 0.003, (h, w))
        N[..., 1] = 1.0
    else:
        P[..., 2] = 2.0
        N[..., 2] = -1.0
    mv = MonoView(P, N, np.ones((h, w), bool), np.eye(3), np.zeros((h, w, 3), np.uint8))
    return View("v", mv, np.eye(3))


def test_camera_height():
    from roomscan.tiers.video import camera_height
    assert abs(camera_height(_floor_view(1.4)) - 1.4) < 0.01
    assert camera_height(_floor_view(None)) is None


def test_level_heights_removes_vertical_drift():
    from roomscan.tiers.video import level_heights
    T = [np.eye(4) for _ in range(4)]
    for k, Tk in enumerate(T):
        Tk[1, 3] = -0.5 * k                     # injected drift
        Tk[0, 3] = k                            # x must be untouched
    out = level_heights(T, [1.4, None, 1.5, 1.4])
    assert np.allclose([t[1, 3] for t in out], [1.4, 1.45, 1.5, 1.4])
    assert np.allclose([t[0, 3] for t in out], [0, 1, 2, 3])
