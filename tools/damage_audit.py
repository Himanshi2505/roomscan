import numpy as np, json, cv2, sys
from roomscan.io import strayscanner as ss
from roomscan.tiers.lidar import build_cloud
from roomscan.geometry.levels import manhattan_yaw, yaw_matrix
from roomscan import damage as DM
src, out = sys.argv[1], sys.argv[2]
plan = json.load(open(f"{out}/result.json"))
cap = ss.load(src)
c = build_cloud(cap, every=6); M = yaw_matrix(manhattan_yaw(c))
views = DM.lidar_views(cap, None, M, every_s=2.0)
regs = DM.detect(views, plan, plan["levels"]["floor_y"], plan["levels"]["ceiling_y"], min_views=1)
tiles = []
for r in regs[:24]:
    t = cv2.cvtColor(r.crop, cv2.COLOR_RGB2BGR); cv2.putText(t, f"{r.cls[:10]} {r.prob:.2f}", (3, 15), 0, 0.45, (0, 0, 255), 1); tiles.append(t)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
cv2.imwrite("out/damage_audit.png", np.vstack(rows)); print(len(regs), "regions")
