import os, _photobleach_batch as pb
OUT=r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0807"; os.makedirs(OUT,exist_ok=True); pb.OUT=OUT
D=r"E:\labcams_data\20260807"; Q=r"E:\DAQ_recorder_output"
R="pco_edge_run000_00000000_2_460_480_uint16.dat"
S=[("PS92_0807",fr"{D}\PS92_20260807_150924\raw_widefield_data\{R}",fr"{Q}\PS92_20260807_151146.h5"),
   ("PS93_0807",fr"{D}\PS93_20260807_174403\raw_widefield_data\{R}",fr"{Q}\PS93_20260807_174416.h5"),
   ("PS94_0807",fr"{D}\PS94_20260807_104125\raw_widefield_data\{R}",fr"{Q}\PS94_20260807_105410.h5"),
   ("PS95_0807",fr"{D}\PS95_20260807_124637\raw_widefield_data\{R}",fr"{Q}\PS95_20260807_125106.h5")]
if __name__=="__main__":
    out=[pb.analyze(*s) for s in S]; pb.summary(out); print("wrote",OUT)
