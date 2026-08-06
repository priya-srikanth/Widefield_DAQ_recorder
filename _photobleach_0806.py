import os, _photobleach_batch as pb
OUT=r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0806"; os.makedirs(OUT,exist_ok=True); pb.OUT=OUT
D=r"E:\labcams_data\20260806"; Q=r"E:\DAQ_recorder_output"
S=[("PS92_0806",fr"{D}\PS92_20260806_124426\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat",fr"{Q}\PS92_20260806_124733.h5"),
   ("PS93_0806",fr"{D}\PS93_20260806_172253\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat",fr"{Q}\PS93_20260806_172316.h5"),
   ("PS94_0806",fr"{D}\PS94_20260806_074320\raw_widefield_data\pco_edge_run001_00000000_2_460_480_uint16.dat",fr"{Q}\PS94_20260806_074558.h5"),
   ("PS95_0806",fr"{D}\PS95_20260806_105300\raw_widefield_data\pco_edge_run001_00000000_2_460_480_uint16.dat",fr"{Q}\PS95_20260806_105651.h5")]
if __name__=="__main__":
    out=[pb.analyze(*s) for s in S]; pb.summary(out); print("wrote",OUT)
