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
