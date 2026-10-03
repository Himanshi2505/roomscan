"""Stand-in photo tier input: N frames spread over a capture's video, JPEG, upright."""
import cv2, sys, numpy as np
from pathlib import Path
src, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, ".")
from roomscan.io import strayscanner as ss
poses = {f.index: f.T_wc for f in ss.load(src).frames}
cap = cv2.VideoCapture(f"{src}/rgb.mp4"); total = int(cap.get(7))
want = set(np.linspace(total*0.05, total*0.95, n).astype(int)); out.mkdir(parents=True, exist_ok=True)
i = 0
while want:
    ok = cap.grab()
    if not ok: break
    if i in want:
        _, img = cap.retrieve()
        # rotate upright like an iPhone photo with EXIF applied: use the pose's
        # world-up expressed in camera coords to pick the 90-degree rotation
        T = poses[i]; up = T[:3,:3].T @ np.array([0,1,0.])     # world up in camera (OpenCV) axes
        ang = np.degrees(np.arctan2(up[0], -up[1]))            # 0 = already upright
        k = int(np.round(ang / 90)) % 4
        img = np.ascontiguousarray(np.rot90(img, k))
        cv2.imwrite(str(out / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92]); want.discard(i)
    i += 1
print(sorted(p.name for p in out.iterdir()))
