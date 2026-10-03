import numpy as np, pickle, sys, time, os
from roomscan.io import strayscanner as ss
from roomscan.tiers.lidar import build_cloud
from roomscan.drift import correct_drift
def crisp(c):
    m=(np.abs(c.normals[:,1])<0.3)
    return len(np.unique(np.floor(c.xyz[m][:,[0,2]]/0.02).astype(int),axis=0))
cap=ss.load(sys.argv[1]); tag=sys.argv[2]
pk=f"out/drift_{tag}.pkl"
if os.path.exists(pk): poses,info=pickle.load(open(pk,"rb"))
else: poses,info=correct_drift(cap); pickle.dump((poses,info),open(pk,"wb"))
print(info)
c0=build_cloud(cap,every=6); c1=build_cloud(cap,every=6,poses=poses)
print("crisp wall cells: as-is",crisp(c0)," corrected",crisp(c1))
