"""Upright 720p stand-in .mp4 from a Stray Scanner capture (rotation as an iPhone .MOV would apply)."""
import cv2, sys, numpy as np
sys.path.insert(0, ".")
from roomscan.io import strayscanner as ss
src, out, fps_out = sys.argv[1], sys.argv[2], float(sys.argv[3])
cap = ss.load(src); poses = {f.index: f.T_wc for f in cap.frames}
vid = cv2.VideoCapture(f"{src}/rgb.mp4"); fps = vid.get(5); step = max(1, round(fps / fps_out))
# one rotation for the whole clip (phone held in portrait throughout)
ks = []
for f in cap.frames[::50]:
    up = f.T_wc[:3, :3].T @ np.array([0, 1, 0.]); ks.append(int(np.round(np.degrees(np.arctan2(up[0], -up[1])) / 90)) % 4)
k = max(set(ks), key=ks.count); print("rotation k", k)
w = None; i = 0
while True:
    ok = vid.grab()
    if not ok: break
    if i % step == 0:
        _, img = vid.retrieve(); img = np.ascontiguousarray(np.rot90(cv2.resize(img, (960, 720)), k))
        if w is None: w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (img.shape[1], img.shape[0]))
        w.write(img)
    i += 1
w.release(); print("frames", i // step)
