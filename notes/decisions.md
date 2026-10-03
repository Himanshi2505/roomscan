# Design decisions log

Each entry: what we chose, why, and the evidence. This is the defense crib sheet.

## D1. Capture route: Route 2, stock protocol on Stray Scanner
- Route 1 (own iOS app) needs Xcode on macOS; the dev machine is Windows.
- Stray Scanner (free, App Store) records RGB video + LiDAR depth + confidence +
  ARKit poses + intrinsics + IMU in an open, documented folder format.
- The same app on a non-Pro iPhone records video without depth, so one app covers
  the video and LiDAR tiers. Photo tier uses the native Camera app.

## D2. Pose convention of Stray Scanner exports
- Hypothesis A: poses are ARKit camera (looks down -Z) -> need diag(1,-1,-1) flip.
- Hypothesis B: poses already for an OpenCV camera (looks down +Z).
- Test: fuse 172 depth frames from single_room, count occupied 5 cm voxels
  (a correct fusion makes walls thin -> fewer voxels).
  A: 147,001 voxels, floor not flat. B: 39,128 voxels, floor at y = -1.50 m.
- Chose B. Script: the check in roomscan/io/strayscanner.py comment.

## D3. Sample data inventory
- single_room/c00a170fe1: 37 s, 1715 depth frames, one room + part of a neighbour.
- single_scan_floor_only/1a8384c3f6: 115 s, ~6 rooms + corridor. Ceiling not
  scanned. Used as the multi-room stitch/drift capture.
- single_scan_with_ceiling/c7d28f72c6: 9745 poses but only 3001 depth frames
  (subsampled); loader must not assume a depth file per pose.

## D4. Wall faces, not raster edges, carry the dimensions
- Walls are axis-aligned faces found as peaks in the 1D histogram of point
  coordinates (after Manhattan rotation). Face position = mean of inliers within
  2 cm of the median: thousands of LiDAR points -> sub-mm statistical error.
  The raster (3 cm) is only used for topology (which room is where).

## D5. Room segmentation = free-space carving + masked watershed + door-width merge
- Floor points alone give a thin band around the walk path (floor under
  furniture/near walls is rarely seen). Free-space carving (camera->hit rays at
  0.3-1.8 m height) fills rooms out to their walls.
- Rays through glass leak outside: blocked by a barrier raster of every
  wall-height vertical-surface point (sills, cabinets), not just fitted faces.
- OpenCV watershed with an "outside" marker flooded along walls (strips);
  replaced by skimage's masked watershed.
- Seeds: distance-to-wall > 0.55 m; doorways (half-width <= 0.45 m) can't seed.
- Merge regions whose shared cut > 1.6 m (open-plan transition, not a door).
- Result on the apartment scan: 9 regions, matching the visible layout.

## D6. Room polygon = Manhattan line arrangement + greedy MDL simplification
- First try (contour -> approxPolyDP -> snap) gave 14-16 jagged edges/room and
  overlapping rooms: raster contours follow furniture, not walls.
- Now: every measured wall face near the room proposes a line; lines cut the
  plane into cells; a cell is "in" if the room mask covers >50% of it.
  Lines are removed greedily while IoU - LAMBDA * n_lines doesn't drop.
  Every final edge lies exactly on a measured face -> dimensions come from faces.
- LAMBDA sweep on the apartment: 0.003 -> up to 20 edges/room (notches of
  6-7 cm); 0.010 -> 4-10 edges, shapes match the visible walls. Chose 0.010.
- Overlaps (gate: "no room overlaps") resolved by giving each overlap to the
  room whose observed floor covers more of it.

## D7. Openings need "see-through" evidence, not just missing wall
- Furniture in front of a wall also removes wall hits. An opening column needs
  (a) no wall hit within 5 cm of the plane AND (b) hits >10 cm behind the plane
  measured from a camera inside the room. Missing + nothing behind = occlusion.
- Door: open over 60% of 0.15-1.85 m; window: sill present below 0.6 m, open
  at 1.0-1.5 m. Width jamb-to-jamb from the 98th/2nd percentile of wall hits.

## D8. Uncertainty: per-tier error model + conformal calibration factor
- sigma^2 = stat^2 + n_faces*sigma_face^2 + (scale_rel*L)^2 + (drift_rel*path)^2
- 90% CI = +-1.645*k*sigma; k fitted on benchmark ground truth so that 90% of
  residuals fall inside. Until then output says calibrated=false.

## D9. Drift: 2D submap pose graph (odometry + ICP loop closure + Manhattan prior)
- ARKit is gravity-aligned -> solve only x, z, yaw. Submaps every 1.5 m walked.
- Manhattan prior = plane-anchored yaw correction (walls should be at 90 deg).
- Apartment scan: 36 submaps, 3 loop closures, max correction 0.55 m / 2.7 deg.
- Evidence it helps (no GT yet): occupied 2 cm wall cells 34,795 -> 32,881
  (-5.5%, crisper walls = more consistent poses). Ablation: --no-drift.
- Cost: 149 s on CPU (to optimise). Plan changes (room merge R4) need GT.

## D10. Photo/video depth: MoGe-2 (MIT) over MASt3R/VGGT (non-commercial)
- Metric point map + intrinsics + normals from one image; CPU-capable.
- Outputs cached by content hash (spec allows deterministic caches; live path
  runs when cache is empty).

## D11. Photo tier is layout-first, not feature-first
- Stand-in photos (6 frames of the single-room video): 0/15 pairs registered
  with SIFT + point-map RANSAC. One frame had 1 keypoint: white walls are
  textureless. Real homes look like this, so features can't be the foundation.
- Geometry check of the old protocol (corner -> opposite corner, 1x lens):
  ~50 deg HFOV sees only part of the two far walls; opposite-corner photos
  share nothing -> under-constrained even with perfect matching.
- New protocol: 0.5x ultra-wide, back to the middle of a wall, photograph the
  opposite wall, clockwise. Each photo measures one wall corner-to-corner on
  its own (side-wall faces = corners), plus camera->wall depth which estimates
  the other dimension. Opposite photos = independent estimates; their scatter
  inflates the interval (Birge ratio). Rectangles only in this path; rectilinear
  rooms fall back to feature registration when texture allows. Known limit.

## D12. Damage: zero-shot CLIP made conservative after an audit
- First version on the (undamaged) single-room scan: 11 detections up to
  p=0.99, all false (wood grain -> "water stain", calendar -> "hole").
- Added: benign prompt bank (wood, paper, furniture, frames, tiles, mirror,
  blur), cosine margin damage-vs-benign >= 0.02, blob-shape test (3-50% of the
  patch deviates from its own surface colour), blur gate, >= 2 views.
  -> 2 single-view detections left, both still false. Thresholds to be set on
  the staged-damage benchmark room (fix-loop candidate).

## D13. MoGe resolution level 0
- CPU timings, one 640 px image: ViT-S 9.8 s (level 9) -> 3.2 s (level 0);
  ViT-L 35.4 s -> 9.4 s. Depth shift vs level 9: ViT-L <= 1.4%, ViT-S up to 4%.

## D14. MoGe metric scale vs LiDAR (24 upright frames, 3 captures)
| config | median |scale-1| | p90 | median scale |
|---|---|---|---|
| ViT-S, guessed FOV | 15.3% | 23% | 0.928 |
| ViT-S, true FOV | 10.7% | 33% | 0.902 |
| ViT-L, guessed FOV | 6.2% | 17% | 0.950 |
- Model size dominates; giving the true FOV did not help ViT-S. Photo tier
  uses ViT-L. Per-photo scale noise is large -> averaged over the room's
  photos (SCALE_IND) + common-mode bias (SCALE_BIAS) anchored by door heads.
- First run fed MoGe sideways frames (Stray Scanner stores sensor-orientation
  video): scales 0.68-1.63, discarded as an unfair test.

## D15. Drift solver speed: 478 s -> 30 s on the 9-room scan, same answer
- Profile: 340 s re-extracting wall points (full-res normals then striding),
  103 s ICP, 33 s solve (Manhattan angle recomputed inside every residual).
- Fixes: subsample depth before back-projection; precompute node angles;
  cap ICP source points with a fixed seed.
- ICP point cap sweep: 3000 -> 25 s but lost 1 of 3 loop closures and moved
  poses 10 cm (rejected); 8000 -> 30 s, 3 closures, within 1.3 cm of the
  full-resolution solution (chosen); 15000 -> 43 s, same as 8000.

## D14b. ViT-L with true FOV, and the global scale correction
- ViT-L + true FOV: median |scale-1| 7.5%, p90 18.8%, std 0.093
  (vs guessed FOV: 6.2%, 17.3%, std 0.138). Scatter drops by a third, median
  error about the same -> keep EXIF FOV for photos (less per-photo noise).
- Median scale 0.947-0.950 in both ViT-L runs: MoGe is ~5% short indoors.
  mono.SCALE_CORR = 1.05. Fitted on the 3 sample captures (24 frames), not on
  the benchmark rooms, so the benchmark remains an independent test of it.

## D16. Video odometry on textureless stretches
- Stand-in walk (single_room at 30 fps, 51 adaptive keyframes) split into 11
  segments: KLT tracks die on blank walls and fast turns.
- Blur gate relaxed: take a blurry keyframe when < 40 tracks remain (MoGe
  tolerates mild blur better than odometry tolerates a gap).
- Geometric fallback: 2D point-to-plane ICP of consecutive keyframes' MoGe
  wall points (KLT -> SIFT -> ICP).
- Diagnostic: rejected ICP pairs had inlier ratios 0.9-1.0 with RMSE
  4-11 cm at 1-2 m depth; the fixed 6 cm limit (from LiDAR) was too strict
  for monocular depth. Limit now 0.03 + 0.05 * median depth.
  Result: 11 -> 7 segments, 42 odometry edges over 51 keyframes.
- Remaining failures are close-ups (0.24-0.4 m from a wall): step 2.
- Step 2: skip keyframe candidates with median MoGe depth < 0.5 m (retry every
  5 frames, each check is a MoGe run). Stand-in: 51 -> 45 keyframes,
  7 -> 6 segments, 38 odometry edges. Modest: the remaining breaks are not
  close-ups. Protocol already says walk ~1 m from walls.
