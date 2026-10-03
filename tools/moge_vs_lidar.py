"""Scale error of MoGe-2 metric depth vs LiDAR (sequential decode, no seeking)."""
import numpy as np, cv2, time, sys
from roomscan import mono
from roomscan.io import strayscanner as ss
mono.MODEL_SIZE = sys.argv[1]; per = int(sys.argv[2]); mono.RES_LEVEL = int(sys.argv[3]); use_fov = len(sys.argv) > 4
res = []
for s in ["single_room/c00a170fe1", "single_scan_floor_only/1a8384c3f6", "single_scan_with_ceiling/c7d28f72c6"]:
    cap = ss.load(s); fr = {f.index: f for f in cap.depth_frames}
    idxs = sorted(fr)[::max(1, len(fr) // per)][:per]; want = set(idxs)
    vid = cv2.VideoCapture(f"{s}/rgb.mp4"); i = 0
    while want:
        if not vid.grab(): break
        if i in want:
            want.discard(i); _, img = vid.retrieve(); f = fr[i]
            up = f.T_wc[:3, :3].T @ np.array([0, 1, 0.])          # rotate upright like a real photo
            k = int(np.round(np.degrees(np.arctan2(up[0], -up[1])) / 90)) % 4
            img = np.ascontiguousarray(np.rot90(img, k))
            t = time.time(); fov = None
            if use_fov:      # simulate EXIF: true horizontal FOV of the (rotated) image
                wpx = img.shape[1]; fpx = f.K_rgb[0, 0]
                fov = float(np.degrees(2 * np.arctan(wpx / 2 / fpx)))
            v = mono.infer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), fov_x_deg=fov); dt = time.time() - t
            h, w = v.points.shape[:2]; dm = v.points[..., 2]
            dl, _ = cap.load_depth(f); dl = np.ascontiguousarray(np.rot90(dl, k))
            dl = cv2.resize(dl, (w, h), interpolation=cv2.INTER_NEAREST)
            ok = (dl > 0.2) & v.mask & (dm > 0)
            r = np.median(dm[ok] / dl[ok]); absrel = np.median(np.abs(dm[ok] / r - dl[ok]) / dl[ok])
            fxr = v.K[0, 0] / (f.K_rgb[0, 0] * max(w, h) / 1920)
            res.append((r, absrel, fxr))
            print(f"{s.split('/')[0]:26s} {i:5d} scale {r:.3f} relshape {absrel:.3f} fx {fxr:.3f} {dt:.1f}s", flush=True)
        i += 1
R = np.array(res)
print(f"{sys.argv[1]} L{sys.argv[3]} fov={use_fov}: n={len(R)} scale median {np.median(R[:,0]):.3f} std {R[:,0].std():.3f} "
      f"|scale-1| median {np.median(np.abs(R[:,0]-1)):.3f} p90 {np.percentile(np.abs(R[:,0]-1),90):.3f}; "
      f"shape absrel {np.median(R[:,1]):.3f}; fx ratio {R[:,2].mean():.3f}+-{R[:,2].std():.3f}")
