import os, _photobleach_batch as pb
OUT=r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0805"; os.makedirs(OUT,exist_ok=True); pb.OUT=OUT
D=r"E:\labcams_data\20260805"; Q=r"E:\DAQ_recorder_output"
S=[("PS92_0805",fr"{D}\PS92_20260805_181150\raw_widefield_data\pco_edge_run001_00000000_2_460_480_uint16.dat",fr"{Q}\PS92_20260805_182111.h5"),
   ("PS93_0805",fr"{D}\PS93_20260805_201110\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat",fr"{Q}\PS93_20260805_202005.h5"),
   ("PS94_0805",fr"{D}\PS94_20260805_124758\raw_widefield_data\pco_edge_run002_00000000_2_460_480_uint16.dat",fr"{Q}\PS94_20260805_131025.h5"),
   ("PS95_0805",fr"{D}\PS95_20260805_155615\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat",fr"{Q}\PS95_20260805_160437.h5")]
if __name__=="__main__":
    out=[pb.analyze(*s) for s in S]; pb.summary(out); print("wrote",OUT)
