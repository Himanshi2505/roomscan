"""Verify: keyframes -> odometry segment count on the stand-in clip (MoGe cached)."""
from pathlib import Path
from roomscan.tiers import video, photo
p = Path("data/raw/video_single_room/walk.mp4")
kfs, tr = video.track_keyframes(p); fov = video.self_calibrated_fov(kfs)
views = [photo.make_view(f"kf{k}", rgb, fov) for k, rgb in kfs]
T, seg, odo = video.odometry(views, tr)
print(len(views), "keyframes ->", max(seg) + 1, "segments;", len(odo), "odometry edges")
