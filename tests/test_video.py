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
