import numpy as np
from PIL import Image

from roomscan.check import check


def _jpg(path, focal35=None):
    im = Image.fromarray(np.full((40, 30, 3), 128, np.uint8))
    exif = Image.Exif()
    if focal35:
        exif.get_ifd(0x8769)[0xA405] = focal35
    im.save(path, exif=exif)


def _levels(rep):
    return [lvl for lvl, _ in rep.lines]


def test_photo_capture(tmp_path):
    good, few = tmp_path / "01_kitchen", tmp_path / "02_hall"
    good.mkdir(), few.mkdir()
    for k in range(4):
        _jpg(good / f"IMG_{k}.jpg", focal35=13)
    _jpg(few / "IMG_9.jpg")                         # 1 photo, no focal length
    rep = check(tmp_path)
    text = "\n".join(m for _, m in rep.lines)
    assert "detected tier: photo" in text
    assert "01_kitchen: 4 photos" in text
    assert any(l == "WARN" and "02_hall: 1 photos" in m for l, m in rep.lines)
    assert any(l == "WARN" and "no EXIF focal length" in m for l, m in rep.lines)
    assert not any("01_kitchen: no EXIF" in m for _, m in rep.lines)


def test_unknown_layout_and_missing_path(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    assert check(tmp_path).fails == 1                # tier cannot be detected
    assert check(tmp_path / "nope").fails == 1


def test_lidar_missing_files(tmp_path):
    (tmp_path / "depth").mkdir()
    (tmp_path / "odometry.csv").write_text("timestamp, frame\n")
    rep = check(tmp_path)
    assert "FAIL" in _levels(rep)
    assert any("rgb.mp4 missing" in m for _, m in rep.lines)
