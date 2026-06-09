"""Per-session photobleaching for 2026-06-08."""
import os, _photobleach_batch as pb
OUT=r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0608"; os.makedirs(OUT,exist_ok=True); pb.OUT=OUT
D=r"E:\labcams_data\20260608"; Q=r"E:\DAQ_recorder_output"; DAT=r"raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat"
S=[("PS92_0608",fr"{D}\PS92_20260608_133759\{DAT}",fr"{Q}\PS92_20260608_133847.h5"),
   ("PS93_0608",fr"{D}\PS93_20260608_195203\{DAT}",fr"{Q}\PS93_20260608_195350.h5"),
   ("PS94_0608",fr"{D}\PS94_20260608_153651\{DAT}",fr"{Q}\PS94_20260608_153702.h5"),
   ("PS95_0608",fr"{D}\PS95_20260608_180943\{DAT}",fr"{Q}\PS95_20260608_180950.h5")]
if __name__=="__main__":
    out=[pb.analyze(*s) for s in S]; pb.summary(out); print("wrote",OUT)
