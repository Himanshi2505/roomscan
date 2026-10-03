# Evidence tools

Scripts that produced the numbers cited in `notes/decisions.md`. Run them from the repo root with `PYTHONPATH=.`. They read the sample captures (`single_room/`, `single_scan_floor_only/`, `single_scan_with_ceiling/`) and write to `out/`.

| script | supports | command | runtime (CPU) |
|---|---|---|---|
| `moge_vs_lidar.py` | D14, D14b: MoGe metric-scale error vs LiDAR | `python tools/moge_vs_lidar.py large 8 0 [fov]` (args: model, frames per capture, resolution level, optional true FOV) | ~10 min |
| `crisp.py` | D9: drift correction makes walls crisper | `python tools/crisp.py single_scan_floor_only/1a8384c3f6 apt` | ~2 min |
| `profile_drift.py` | D15: drift solver 478 s → 30 s, same solution | `python tools/profile_drift.py` (run `crisp.py` first for the comparison) | ~1 min |
| `make_photo_set.py` | stand-in photo input (upright frames) | `python tools/make_photo_set.py single_room/c00a170fe1 data/raw/photo_single_room/room1 6` | ~1 min |
| `make_video_standin.py` | stand-in video input (upright; 23 fps from the 46 fps source) | `python tools/make_video_standin.py single_room/c00a170fe1 data/raw/video_single_room/walk.mp4 30` | ~1 min |
| `video_segments.py` | D16: odometry segment counts | `python tools/video_segments.py` | ~10 min uncached |
| `compare_tiers.py` | D17, D18: video vs LiDAR on the same capture | `python tools/compare_tiers.py out/video_single out/single_lidar` (after `roomscan run` on both) | seconds |
| `damage_audit.py` | D12: damage false-positive audit (crops to `out/damage_audit.png`) | `python tools/damage_audit.py single_room/c00a170fe1 out/single_lidar` | ~3 min |

Model outputs are cached under `out/cache`, so re-runs replay deterministically.
