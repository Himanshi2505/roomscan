"""Every module imports and the tier entry points exist (catches refactor slips)."""
import importlib
import pkgutil

import roomscan


def test_all_modules_import():
    for m in pkgutil.walk_packages(roomscan.__path__, "roomscan."):
        if m.name != "roomscan.__main__":
            importlib.import_module(m.name)


def test_tier_entry_points():
    from roomscan.tiers import photo, photo_layout, video
    assert callable(video.build) and callable(video.track_keyframes) and callable(video.self_calibrated_fov)
    assert callable(photo.reconstruct_room) and callable(photo_layout.room_plan)
