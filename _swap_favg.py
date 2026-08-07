"""One-off: swap frames_average channel order so [0]=415, [1]=470 for display.

This recording's DAQ led415/led470 TTLs were swapped, so the .dat (and thus
frames_average) channel order is [real-470, real-415]. Swapping here makes the
downstream mean_415_470 overlay label channels correctly. Original kept as
frames_average_datorder.npy. Spatial maps use SVTcorr+U and are unaffected.
"""
import numpy as np, os, shutil
res = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue\motion_corrected\wfield_local_results"
p = os.path.join(res, "frames_average.npy")
bak = os.path.join(res, "frames_average_datorder.npy")
favg = np.load(p)
if not os.path.exists(bak):
    shutil.copy2(p, bak)
    np.save(p, favg[::-1].copy())   # [ch0,ch1]=[470,415] -> [415,470]
    print("swapped frames_average to [415,470]; original -> frames_average_datorder.npy")
else:
    print("already swapped (backup exists); leaving as-is")
print("shape", np.load(p).shape)
