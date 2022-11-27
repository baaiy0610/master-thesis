import torch
import zarr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import scipy.spatial
import xarray as xr
from tqdm import tqdm
import shallow1d
from shallow1d import fvm_TVD_RK, boundary_condition_periodic, g, flux_roe, fvm_2ndorder_space_classic, initial_condition

fvm = fvm_TVD_RK
boundary_condition = boundary_condition_periodic
shallow1d.fvm_2ndorder_space = fvm_2ndorder_space_classic
shallow1d.boundary_condition = boundary_condition_periodic
shallow1d.flux = flux_roe
g = 9.8
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def integrate(ds_out, q_init, dx, nhalo, downsample_ratio=1, save_interval=10):
    """
    Solve 1D shallow water equation 
    ds_out.q: double[nT, 2, N//downsample_ratio]
    q_init: double[2, M, N], initial condition
    dx: double[N], matrix of cell area 
    """
    nT = ds_out.q.shape[0]
    _, N = q_init.shape
    device = q_init.device
    q0 = boundary_condition(q_init, nhalo)

    it = 0
    save_it = 0

    while save_it < nT:
        dt = 1e-3*(128/N)
        q0 = fvm(q0, dx, dt, nhalo)
        it += 1
        if it % save_interval == 0:
            ds_out.q.values[save_it, ...] = F.avg_pool1d(q0[:, nhalo:-nhalo].unsqueeze(0), kernel_size = downsample_ratio).squeeze(0).cpu().numpy()
            save_it += 1
        if it % 5000 == 0:
            tmp = torch.zeros((2, N), device=device)
            tmp[0, ...] = 0.5 * torch.square(q0[0, nhalo:-nhalo]) * g                       #0.5* g*h^2 
            tmp[1, ...] = 0.5 * torch.square(q0[1, nhalo:-nhalo]) / q0[0, nhalo:-nhalo]     #u^2*h^2 / h = 0.5 * u^2*h
            sum_ = (tmp * dx).sum()
            print("T:",dt*it,"it =", it, "Total energy:", sum_)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=2048, type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scale", default=32, type=int)
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    nhalo = 3
    downsample_ratio = args.scale
    N = args.resolution_x
    assert N % downsample_ratio == 0

    x_range = (0., 10.)
    dx = (x_range[1]-x_range[0]) / N
    nsimulations = 20
    output_path = "./training_data"

    nT = 2560
    save_interval = downsample_ratio
    ds_out_template = xr.Dataset(
                                {"q": (["sample", "variable", "x"], np.zeros((nT, 2, N//downsample_ratio)))},
                                attrs={"nT":nT, "dx": dx*downsample_ratio, "downsample_ratio": downsample_ratio, "save_interval": save_interval, "N": N//downsample_ratio}
                                )

    file_name = f"1d_torch_{N}_{downsample_ratio}x_randvel_iter_TVD_downsampleT_gaussian_qinit.zarr"
    file_exist = os.path.exists(f"{output_path}/{file_name}")
    synchronizer = zarr.ProcessSynchronizer(f"{output_path}/{file_name}.sync")
    plt.figure(figsize=(10, 5), dpi=400)

    for isimulation in tqdm(range(nsimulations)):
        q_init = initial_condition(N)
        q_init = q_init.to(device)
        ds_out = ds_out_template.copy()
        integrate(ds_out, q_init, dx, nhalo, downsample_ratio=downsample_ratio, save_interval=save_interval)
        append_dim = "sample" if isimulation > 0 or file_exist else None
        ds_out.chunk({"sample":N//downsample_ratio}).to_zarr(f"{output_path}/{file_name}", mode="a", append_dim=append_dim, synchronizer=synchronizer)

        if args.plot:
            plt.imshow(ds_out.q.values[::nT//(N//downsample_ratio), 0, :], origin="lower", cmap = "viridis")
            plt.title(f"high to low")
            plt.colorbar()
            plt.show()
            File_Path = f"./data_jpg/{downsample_ratio}/"
            if not os.path.exists(File_Path):
                os.makedirs(File_Path)
            plt.savefig(f"./data_jpg/{downsample_ratio}/{N}_{downsample_ratio}-{isimulation}.jpg")
            plt.clf()
