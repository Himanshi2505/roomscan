# Public validation (third-party data, not our benchmark)

**What this is:** supplementary validation on Apple's public **ARKitScenes** dataset, used because no benchmark captures of our own exist yet.

**What this is not:** our benchmark. It is not captured by us, not on an iPhone 15, and not with tape ground truth. It does not replace the captures the brief requires.

| | |
|---|---|
| Dataset | ARKitScenes (Apple), raw split, Validation fold |
| Scene | visit **421380**: one room recorded twice (videos 42445021, 42445022) |
| Device | iPad Pro 2020 (LiDAR), recorded by Apple |
| Ground truth | static laser scan **170107** (survey-grade, no drift) |
| Licence | ARKitScenes LICENSE: commercial use permitted for organisations under 700M monthly active users |
| Not in git | fetched by `tools/fetch_arkitscenes.py` into `data/public/` |

## Reproduce

```bash
python tools/fetch_arkitscenes.py                       # ~3.5 GB
python tools/arkitscenes_to_stray.py data/public/arkitscenes/42445021 data/public/stray/42445021 --invert --zup
python tools/arkitscenes_to_stray.py data/public/arkitscenes/42445022 data/public/stray/42445022 --invert --zup
python tools/laser_gt.py data/public/arkitscenes/laser/170107          # ground-truth faces + render
python -m roomscan run data/public/stray/42445021 --out out/public/42445021
python tools/register_to_laser.py data/public/stray/42445021 out/public/42445021 data/public/arkitscenes/laser/170107
python tools/face_accuracy.py data/public/stray/42445021 out/public/42445021 benchmark/public_gt/arkitscenes_421380.json
```

## Conversion decisions

Both were found empirically, not assumed (notes D19):
- **Pose convention:** the `.traj` stores world→camera poses with OpenCV camera axes, giving 57,730 occupied 5 cm voxels against 330–460k for the alternatives.
- **World axes:** the world is Z-up and is rotated to the pipeline's Y-up.

## Ground truth

From `benchmark/public_gt/arkitscenes_421380.json`:
- **Room:** 2.826 × 2.979 m; ceiling 2.315 m.
- **Assumption:** the N and E walls are fully observed. The S and W walls **assume** the rectangle continues behind an L-shaped ~2.4 m opening.
- **Openings:** left out, because the jambs aren't clearly visible.

## Results (LiDAR tier)

| measure | result | notes |
|---|---|---|
| Registration of recording 1 to the laser | 82% of wall points within 5 cm, RMSE 2.1 cm | accepted |
| Registration of recording 2 | 43–45% | rejected: covers space outside the scan, plus a smeared region |
| Room N–S, wall face to face | −0.8 cm | distances measured in our own cloud, so not flattered by the registration |
| Room W–E | −0.3 cm | |
| Ceiling height, per room, after the fix loop | R1 −1.29 cm, R2 −0.59 cm | gate ≤ 1.5 cm: **PASS**; was +1.51 / +3.01 |
| Room segmentation | **fails**: the room is split, or merged with the corridor | partly because the ground truth's S/W walls are an assumption (notes D21) |

Full fix-loop story: `docs/FIX_LOOP.md`.

## Results (video tier): the same walkthrough's `.mov`, MoGe-2 ViT-L, no LiDAR

`python -m roomscan run data/public/arkitscenes/42445021/42445021.mov --out out/public/video_42445021`

The clip is 60 fps, 48 s, and decodes upright via its rotation metadata.

| measure | result |
|---|---|
| Tracking | 1 segment, 56 keyframes, 4 loop closures (held, unlike the stand-in clip) |
| Pose-graph correction | 1.24 m (8.6% of path): heavy distortion |
| Largest room's walls vs the LiDAR plan of the same recording | median off 33 cm (**18%**), max 58 cm; mixed short and long, so not a single scale factor |
| Ceiling height (R2) | 2.834 m vs laser 2.315 m (**+52 cm**) |
| Intervals | R1 area 9.24 m², 90% CI [2.2, 16.3]: very wide, but contains the LiDAR value (honest, not useful) |
| Time | 1,269 s on CPU at first; **568 s** after one inference per keyframe (D23) |

**Verdict:** the video tier **fails** the ±3% wall gate on this scene by a wide margin, and it is too slow on CPU for a live run. Its intervals widen instead of lying. Next candidates: scale and ceiling anchoring per keyframe, and fewer keyframes or a GPU.

## Not covered by this data

- photo tier on iPhone photos (stills from this scene are pending)
- the head-to-head against a consumer app
- staged damage
- one room recorded twice with tape ground truth
- multi-room adjacency against ground truth
