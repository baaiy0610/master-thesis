import numpy as np
import zarr
import torch

dataset1 = "data_bilinear_2d"
dataset2 = "data_bilinear_3d"
dataset3 = "data_s2d_2d"
dataset4 = "data_s2d_3d"
dataset_train = dataset1
data_train = zarr.open(f"./{dataset_train}/Data_Train.zarr", mode="r")
print(data_train.shape) #(nsample, variables, M, N)
data = torch.tensor(np.array(data_train))
data[:, 1:, :, :] = data[:, 1:, :, :]/(data[:, 0, :, :].unsqueeze(1))

print(data.shape)
mu = torch.mean(data, dim=(0,2,3), keepdim=True)
std = torch.std(data, dim=(0,2,3), keepdim=True)
torch.save([mu, std], f"./mu_std_{dataset_train}.pt")
