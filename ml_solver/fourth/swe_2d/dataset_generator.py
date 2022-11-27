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
import shallow2d
from shallow2d import fvm_TVD_RK, boundary_condition, g, flux_roe, fvm_2ndorder_space_classic
import MeshGrid

fvm = fvm_TVD_RK
shallow2d.fvm_2ndorder_space = fvm_2ndorder_space_classic
shallow2d.flux = flux_roe
#---------------------------------------------------------------------------------------------------------------------------------------------
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def minmaxscaler(data):
    min_data = torch.min(data)
    max_data = torch.max(data)
    return (data - min_data) / (max_data - min_data)

def initial_condition_perlin_noise2d(M, N):
    q = torch.zeros((3, M, N), dtype=torch.double)
    a = (0.05-(-0.05)) * np.random.random() - 0.05
    q[0, ...] = minmaxscaler(torch.from_numpy(generate_perlin_noise_2d([M, N], [2,2]))) + (0.15-0.05) * np.random.random() + 0.05
    q[1, ...] = a * minmaxscaler(torch.from_numpy(generate_perlin_noise_2d([M, N], [2,2])))
    q[2, ...] = a * minmaxscaler(torch.from_numpy(generate_perlin_noise_2d([M, N], [2,2])))
    return q

#---------------------------------------------------------------------------------------------------------------------------------------------

def integrate(ds_out, q_init, vert, cell_area, nhalo, downsample_ratio=1, save_interval=10):
    """
    Solve 2D shallow water equation 
    ds_out: double[nT, 3, N//downsample_ratio]
    q_init: double[3, M, N], initial condition
    vert: double[2, M+1, N+1]
    cell_area: double[M, N], matrix of cell area 
    """
    nT = ds_out.q.shape[0]
    _, M, N = q_init.shape
    device = q_init.device
    q0 = boundary_condition(q_init, nhalo)

    it = 0
    save_it = 0

    while save_it < nT:
        dt = 1e-3*(128/max(N, M))
        q0 = fvm(q0, vert, cell_area, dt, nhalo)
        it += 1
        if it % save_interval == 0:
            ds_out.q.values[save_it, ...] = F.avg_pool2d(q0[:, nhalo:-nhalo, nhalo:-nhalo].unsqueeze(0), kernel_size = downsample_ratio).squeeze(0).cpu().numpy()
            save_it += 1
        if it % 100 == 0:
            tmp = torch.zeros((2, M, N), dtype=torch.double, device=device)
            tmp[0, ...] = 0.5 * torch.square(q0[0, nhalo:-nhalo, nhalo:-nhalo]) * g                      
            tmp[1, ...] = 0.5 * (torch.square(q0[1, nhalo:-nhalo, nhalo:-nhalo]) + torch.square(q0[2, nhalo:-nhalo, nhalo:-nhalo])) / q0[0, nhalo:-nhalo, nhalo:-nhalo]
            sum_ = torch.sum(tmp * cell_area).item()
            print("T:",dt*it,"it =", it, "Total energy:", sum_)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=1024, type=int)
    parser.add_argument("--resolution_y", default=1024, type=int)
    parser.add_argument("--scale", default=8, type=int)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    nhalo = 3
    M = args.resolution_x
    N = args.resolution_y
    downsample_ratio = args.scale

    assert N % downsample_ratio == 0

    x_range = (0., 10.)
    y_range = (0., 10.)
    
    nsimulations = 10
    output_path = "./training_data"

    vert, cell_area = MeshGrid.regular_grid(x_range, y_range, M, N)
    vert_lowers, cell_area_lowers = MeshGrid.regular_grid(x_range, y_range, M//downsample_ratio, N//downsample_ratio)

    nT = 2560
    dt = 1e-3*(128/max(N, M))
    save_interval = downsample_ratio
    ds_out_template = xr.Dataset(
                                {"q": (["sample", "variable", "x", "y"], np.zeros((nT, 3, M//downsample_ratio, N//downsample_ratio))),
                                "vert": (["dimension_2d", "xv", "yv"], vert_lowers),
                                "cell_area": (["x", "y"], cell_area_lowers),
                                },
                                attrs={"nT":nT, "downsample_ratio": downsample_ratio, "save_interval": save_interval, "M": M//downsample_ratio, "N": N//downsample_ratio}
                                )
    file_name = f"quad_torch_{M}_{N}_{downsample_ratio}x_randvel_iter_TVD_downsampleT_perlin2d_qinit.zarr"
    file_exist = os.path.exists(f"{output_path}/{file_name}")
    synchronizer = zarr.ProcessSynchronizer(f"{output_path}/{file_name}.sync")

    for isimulation in tqdm(range(nsimulations)):
        q_init = initial_condition_perlin_noise2d(M, N)
        q_init = q_init.to(device)
        vert = vert.to(device)
        cell_area = cell_area.to(device)
        ds_out = ds_out_template.copy()
        integrate(ds_out, q_init, vert, cell_area, nhalo, downsample_ratio=downsample_ratio, save_interval=save_interval)
        append_dim = "sample" if isimulation > 0 or file_exist else None
        ds_out.chunk({"sample": N//downsample_ratio}).to_zarr(f"{output_path}/{file_name}", mode="a", append_dim=append_dim, synchronizer=synchronizer)

        if args.plot:
            delta_t = nT//8
            plt.figure(figsize=(9, 9))
            for i in range(8): 
                plt.subplot(2, 4, i + 1)
                plt.imshow(ds_out.q.values[i*delta_t, 0, ...], cmap='viridis', origin="lower", extent=[0., 10., 0., 10.])
                plt.title(f"time step:{(nT//8)*i}")
                plt.xlabel('x')
                plt.ylabel('y')
            File_Path = f"./data_jpg/{downsample_ratio}/"
            if not os.path.exists(File_Path):
                os.makedirs(File_Path)
            plt.show()
            plt.savefig(f"./data_jpg/{downsample_ratio}/{M}-{N}_{downsample_ratio}x-{isimulation}.jpg")
