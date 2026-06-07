"""Per-session photobleaching for 2026-06-07 (reuses _photobleach_batch)."""
import os, _photobleach_batch as pb
OUT=r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0607"; os.makedirs(OUT,exist_ok=True); pb.OUT=OUT
D=r"E:\labcams_data\20260607"; Q=r"E:\DAQ_recorder_output"; DAT=r"raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat"
S=[("PS92_0607",fr"{D}\PS92_20260607_121538\{DAT}",fr"{Q}\PS92_20260607_121551.h5"),
   ("PS93_0607",fr"{D}\PS93_20260607_174844\{DAT}",fr"{Q}\PS93_20260607_174854.h5"),
   ("PS94_0607",fr"{D}\PS94_20260607_140731\{DAT}",fr"{Q}\PS94_20260607_140813.h5"),
   ("PS95_0607",fr"{D}\PS95_20260607_155000\{DAT}",fr"{Q}\PS95_20260607_155400.h5")]
if __name__=="__main__":
    out=[pb.analyze(*s) for s in S]; pb.summary(out); print("wrote",OUT)
