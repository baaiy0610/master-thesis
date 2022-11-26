import zarr
import torch
downsample_ratio = 32
nT = 2560
file_path = f"./training_data/1d_torch_2048_{downsample_ratio}x_randvel_iter_TVD_downsampleT_gaussian_qinit.zarr"

ds = zarr.open(f"{file_path}", "r")
print("Total train dataset shape:", ds.q.shape) 

wtf = []
n_simulation = 20
for i in range(n_simulation):
    a = torch.isnan(torch.as_tensor(ds.q[i*nT:(i+1)*nT, ...])).any()
    if a == True:
        wtf.append(i)
print(wtf)
