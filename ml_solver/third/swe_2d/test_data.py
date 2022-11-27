import zarr
import torch
downsample_ratio = 16
file_path = f"./training_data/quad_torch_1024_1024_{downsample_ratio}x_randvel_iter_TVD_downsampleT_perlin2d_qinit.zarr"
ds = zarr.open(f"{file_path}", "r")
print("Total train dataset shape:", ds.q.shape) 

offset = 2560
wtf = []

n_simulation = 10
for i in range(n_simulation):
    data = torch.as_tensor(ds.q[i*offset:(i+1)*offset, ...])
    a = torch.isnan(data).any()
    if a == True:
        wtf.append(i)
print(wtf)


