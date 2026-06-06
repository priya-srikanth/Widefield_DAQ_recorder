"""Photobleaching for 2026-06-06 (4 sessions): per-session 415/470 trends + summary.

Reuses analyze() + summary() from _photobleach_batch (ROI-median intensity per
channel from the raw .dat + DAQ LED TTLs). Outputs to _photobleach_out_0606\.
Run in the wfield env (h5py+scipy+matplotlib, no wfield import).
"""
import os
import _photobleach_batch as pb

OUT = r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0606"
os.makedirs(OUT, exist_ok=True)
pb.OUT = OUT

D = r"E:\labcams_data\20260606"
Q = r"E:\DAQ_recorder_output"
DAT = r"raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat"
SESSIONS = [
    ("PS92_0606", fr"{D}\PS92_20260606_122451\{DAT}", fr"{Q}\PS92_20260606_122508.h5"),
    ("PS93_0606", fr"{D}\PS93_20260606_180117\{DAT}", fr"{Q}\PS93_20260606_180219.h5"),
    ("PS94_0606", fr"{D}\PS94_20260606_140854\{DAT}", fr"{Q}\PS94_20260606_140912.h5"),
    ("PS95_0606", fr"{D}\PS95_20260606_160806\{DAT}", fr"{Q}\PS95_20260606_160825.h5"),
]

if __name__ == "__main__":
    out = [pb.analyze(*s) for s in SESSIONS]
    pb.summary(out)
    print("\nwrote per-session + summary to", OUT)
