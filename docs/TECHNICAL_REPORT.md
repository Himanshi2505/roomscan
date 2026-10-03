# roomscan: technical report

*Sections marked ⏳ need our own benchmark captures, which don't exist yet. Real-data numbers below come from **third-party** public data (Apple ARKitScenes, `docs/PUBLIC_VALIDATION.md`), clearly labelled. Design evidence lives in `notes/decisions.md` (D1–D22); `tools/README.md` maps each number to the script that regenerates it.*

## 1. Architecture

One capture in, one schema-validated `result.json` and one rendered plan out: `python -m roomscan run <capture>`. The tier is detected from the input.

```
 photos/ ─► MoGe-2 per photo ─► gravity + Manhattan ─┬─► wall-per-photo layout ──┐
                                                    └─► SIFT registration ──────┤ per room
 video  ─► KLT keyframing ─► MoGe-2 per keyframe ─► visual-metric odometry ─┐   │
 LiDAR  ─► Stray Scanner depth + ARKit poses ─────────────────────────────┤   │
                                       2D pose graph (drift) ◄────────────┘   │
                                               │                              │
                    oriented point cloud (gravity-aligned, metric) ◄──────────┘
                                               │
   floor/ceiling levels ─► wall faces ─► free-space carving ─► room segmentation
     ─► line-arrangement polygons ─► openings (see-through test) ─► adjacency
                                               │
          photo tier: door-based stitching ◄───┤
                                               ▼
      damage (CLIP zero-shot, geometry-gated) ─► concealed-damage rules ─► scope
                                               ▼
              intervals (per-tier error model × calibrated k) ─► JSON + plan
```

Every tier converges on the same **oriented point cloud** (metric, +Y up, per-point source camera). Everything downstream is shared, which is how three very different inputs honour one output contract. The photo tier also has a layout-first path that skips the cloud for textureless rooms (§2.1).

**Dimensions come from wall faces, not pixels.** A wall face is a 1D histogram peak of point coordinates in the Manhattan frame, and its position is the mean of thousands of inliers. The 3 cm raster is used only for topology: which room is where. Room polygons are chosen from the arrangement of face lines (cell coverage plus an MDL-style penalty per line), so every edge lies exactly on a measured face (D4, D6).

**Openings** need see-through evidence. A missing wall with hits measured *behind* the plane by a camera *inside* the room is an opening. A missing wall with nothing behind it is occlusion. This is the main phantom-opening guard (D7), and it's covered by a regression test.

## 2. Tier design and device matrix

See `docs/DEVICE_MATRIX.md`. Summary:

| tier | hardware | metric scale from | pose from | drift handling |
|---|---|---|---|---|
| LiDAR | iPhone Pro (12+) | LiDAR depth | ARKit VIO | submap pose graph |
| video | any iPhone 15+ | MoGe-2 ViT-L, self-calibrated FOV, ×1.05 bias fix | KLT + point-map RANSAC | keyframe pose graph |
| photo | any iPhone 15+ | MoGe-2 ViT-L + EXIF FOV, door-head anchor | none (layout) / SIFT | n/a: rooms joined at doors |

### 2.1 Photo tier

The first design (corner photos plus SIFT registration) registered **0 of 15** photo pairs on the stand-in set, because white walls have no texture. A geometric check also showed that 1× corner photos are under-constrained even with perfect matching (D11). The shipped design is **layout-first**: 0.5× ultra-wide, back to the middle of a wall, one photo of the opposite wall, clockwise.
- Each photo measures one wall corner-to-corner on its own (the side-wall faces are the corners), plus the camera-to-wall depth, which estimates the other dimension.
- Opposite photos are independent estimates. Their disagreement inflates the interval (Birge ratio).
- Rooms are stitched through doors: doorway photos registered into the neighbouring room pick the door, with a jigsaw search (no overlap, maximum shared wall) as fallback.

**Metric scale is the photo tier's dominant error.** Measured against LiDAR on 24 frames (D14), MoGe-2 ViT-S is off by a median of 15%. ViT-L is off by 6–7.5% (p90 ~18%) and reads ~5% short. Mitigations:
- ViT-L
- EXIF focal length (cuts scatter by a third)
- a global ×1.05 correction (fitted on the sample captures, not on the benchmark)
- averaging over photos
- a door-head anchor: interior door openings are 2.05 ± 0.04 m, fused by inverse variance with MoGe's scale.

### 2.2 Video tier

Keyframes are chosen by tracking quality: KLT corners are tracked through every frame, and a new keyframe starts when fewer than 30% survive. On a stand-in clip (a fast close-range sweep, harsher than the protocol), fixed 1 fps keyframes broke the trajectory into 15 segments. Adaptive keyframing, an ICP fallback and close-up rejection brought this to 6. When tracking breaks, the plan keeps only the largest segment, warns about partial coverage, and widens intervals by the pose-graph distortion (D16–D18). **The video tier is the least reliable tier.** On a real third-party walkthrough (ARKitScenes 42445021) tracking held (1 segment, 4 loop closures), but walls were off by a median of 18% against the LiDAR plan, the ceiling by +52 cm, and the run took 21 min on CPU. It **fails** the ±3% gate; its intervals widened enough to contain the truth rather than lying. Scale is never chained, because each keyframe keeps MoGe's own metric scale, so scale error doesn't random-walk. The protocol ends the walk where it began, which guarantees a loop closure.

## 3. Drift handling

ARKit and visual odometry are gravity-aligned, so the graph is solved in (x, z, yaw). Height is handled separately (below).
- **Nodes:** 1.5 m submaps (LiDAR) or keyframes (video).
- **Factors:** relative odometry; ICP (LiDAR) or feature (video) loop closures with a robust loss; a **Manhattan yaw prior** per node, which is the plane-anchored correction that pulls each node's walls to the global 90° grid.

On the 9-room sample apartment: 36 submaps, 3 loop closures, maximum correction 0.55 m / 2.7°. Without ground truth the evidence is internal: wall crispness (occupied 2 cm wall cells) improves 5.5% with correction on (D9). Solver time went from 478 s to 30 s with an unchanged solution (D15). ⏳ footprint ablation vs ground truth (`--no-drift`).

**Vertical drift.** The original design assumed ARKit height drift is negligible (D9). Real data proved it wrong: on ARKitScenes 42445021 the floor reads up to +3.6 cm high during one stretch of a 48 s walk, and it follows *time*, not place. A **floor-anchored vertical correction** now removes, per 2 s window, that window's floor offset from its poses, interpolating where no floor is seen. Per-window floor offsets went from −0.5…+3.6 cm to within ±0.4 cm (fix loop iteration 2).

## 4. Error budget

σ² = σ²_stat + n_faces·σ²_face + (scale_rel·L)² + (drift_rel·path)², with a 90% interval of ±1.645·k·σ.

| term | LiDAR | video | photo | source |
|---|---|---|---|---|
| σ_face (per surface bias) | 0.6 cm | 1.5 cm | 3 cm | prior → calibrated |
| scale_rel | 0.3% | 1.2% | per-photo 8% (averages down) + common 4% (door anchor) | D14 |
| drift_rel | 0.2%/m (off when corrected) | 0.6%/m | – | prior |

⏳ measured contribution of each term on the benchmark.

## 5. Calibration analysis

k per tier is fitted conformally: the smallest k such that ≥90% of ground-truth residuals fall inside the interval (`benchmark.evaluate --calibrate`), and the coverage achieved is reported per tier. ⏳ k values: not fitted yet, because there are no benchmark captures; all intervals currently carry `calibrated: false`.

**Provisional:** video k = 2.1, from a stand-in clip against LiDAR (5 wall pairs needed 2.02; D18). The point is to avoid confident garbage before calibration exists.

## 6. Benchmark

⏳ **Our own benchmark** (gates at all three tiers, the repeatability table, the head-to-head, timing) requires captures that don't exist yet; the harness is built and tested (`benchmark/evaluate.py`).

**Third-party validation** (ARKitScenes visit 421380, iPad Pro LiDAR, laser ground truth; `docs/PUBLIC_VALIDATION.md`):

| measure | result |
|---|---|
| room dimensions, wall face to face | −0.8 cm and −0.3 cm |
| ceiling height after the fix loop | R1 −1.29 cm, R2 −0.59 cm (gate ≤ 1.5 cm: pass) |
| registration, recording 1 | 82% of wall points within 5 cm |
| registration, recording 2 | rejected (coverage outside the scan) |
| room segmentation | fails on this scene (split or merged) |

The measurement core is accurate. The room-level layer above it is the weak point.

## 7. Fix loop

Full record: `docs/FIX_LOOP.md`. Each declaration was committed before its code; before and after regenerate from the tags with `python tools/fixloop_ceiling.py`.

1. **First target, room segmentation.** Abandoned after three attempts (D21): the ground truth's walls at the opening were an assumption, so the gate couldn't be scored fairly. Kept from it: a principled, side-effect-free rule that furniture against a wall is not a wall.
2. **Ceiling height, gate ≤ 1.5 cm.** Before: +1.51 / +3.01 cm (FAIL).
   - **Iteration 1:** plane heights now come from raw horizontal points. The declared mechanism (voxel quantization) was wrong; isolation showed the cause is cross-surface mixing in voxels. Result: R1 passes, R2 fails.
   - **Iteration 2:** floor-anchored vertical drift correction. Result: R1 −1.29, R2 −0.59 cm, so the **gate passes**.
   - **Missed predictions, reported:** R1 was predicted within ±0.5 cm (a far-range ceiling bias remains), and the sample apartment's footprint moved +0.9% where no change was predicted.

## 8. Known failure modes

- **Mirrors and glass:** LiDAR and MoGe "see" a phantom room behind them. Free-space rays are blocked by a barrier of all wall-height vertical points, and interior components not touched by the walked path are dropped. A mirror spanning a whole wall can still extend a room.
- **Unswept ceiling:** reported as `null` with a warning, never guessed (sample apartment scan).
- **Textureless rooms:** handled in the photo tier by the layout path (rectangles only). An L-shaped textureless room falls back to the feature path, and may fail.
- **Non-Manhattan walls:** everything snaps to 90°. Angled walls become staircases.
- **Sideways images without orientation metadata:** gravity is initialised from image up (D11). iPhone photos and videos always carry it.
- **Damage:** zero-shot CLIP. After the audit, 2 single-view false positives remain on an undamaged room, and recall is unknown until the staged-damage room. Thresholds are a fix-loop candidate.
- **Closed doors** read as wall; the protocol requires open doors.
- **CPU time:** photo and video tiers run ~10 s per image with MoGe-L. A GPU makes this about 20× faster.
- **Low light and wet-look floors:** LiDAR confidence drops (filtered at confidence 2), and MoGe degrades in the dark. ⏳ measured.
- **Furniture as walls:** long furniture faces can split rooms. Faces with evidence of standing against a wall are excluded; free-standing furniture can still split a room (D21).
- **Wide openings:** gaps wider than 1.6 m merge spaces into one room by design (open plan). A room with a very wide doorway reads as one space with the corridor.
- **Far-range ceiling bias:** ceiling seen from 3–6 m at grazing angles reads ~2 cm low on LiDAR. It is not yet corrected, and it is the next fix-loop candidate.
- **Video tracking on textureless walls:** trajectories fragment. The output is restricted to the largest segment and flagged.
