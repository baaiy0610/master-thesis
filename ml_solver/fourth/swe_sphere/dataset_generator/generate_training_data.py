import SWE_sphere
from SWE_sphere import fvm_heun, boundary_condition, g, limiter, flux_roe, fvm_2ndorder_space_classic
import torch
import torch.nn.functional as F
import MeshGrid
import zarr
import xarray as xr
import numpy as np
from tqdm import tqdm
import os

flux = flux_roe
fvm = fvm_heun
SWE_sphere.fvm_2ndorder_space = fvm_2ndorder_space_classic
SWE_sphere.flux = flux_roe

def initial_condition_perlin_serial(cell_center_3d, scale_factor = 4.0):
    """
    cell_center_3d: [3, M, N]
    out: [4, M, N], initial condition for h, hu, hv, hw
    """
    from noise import snoise3
    _, M, N = cell_center_3d.shape
    out = torch.zeros((4, M, N))
    cell_center_3d = cell_center_3d.cpu()*scale_factor
    offsets = np.random.randint(10, 1000, size=(4, 3))
    for i in range(M):
        for j in range(N):
            coord_h = cell_center_3d[:, i, j] + offsets[0, :]
            coord_hu = cell_center_3d[:, i, j] + offsets[1, :]
            coord_hv = cell_center_3d[:, i, j] + offsets[2, :]
            coord_hw = cell_center_3d[:, i, j] + offsets[3, :]
            out[0, i, j] = (snoise3(coord_h[0], coord_h[1], coord_h[2]) + 1.0)*0.5*0.4e-3 + 1.3e-3
            out[1, i, j] = snoise3(coord_hu[0], coord_hu[1], coord_hu[2]) * 1.6e-3
            out[2, i, j] = snoise3(coord_hv[0], coord_hv[1], coord_hv[2]) * 1.6e-3
            out[3, i, j] = snoise3(coord_hw[0], coord_hw[1], coord_hw[2]) * 1.6e-3

    tmp = out[1,...]*cell_center_3d[0,...] + out[2,...]*cell_center_3d[1,...] + out[3,...]*cell_center_3d[2,...]
    out[1,...] -= tmp*cell_center_3d[0,...]
    out[2,...] -= tmp*cell_center_3d[1,...]
    out[3,...] -= tmp*cell_center_3d[2,...]

    return out

def initial_condition_perlin_vectorized(cell_center_3d, scale_factor = 4.0):
    """
    cell_center_3d: [3, M, N]
    out: [4, M, N], initial condition for h, hu, hv, hw
    """
    from perlin import noise3_vec
    _, M, N = cell_center_3d.shape
    out = np.zeros((4, M, N))
    cell_center_3d = cell_center_3d.cpu().numpy()
    cell_center_3d_scaled = cell_center_3d*scale_factor
    noise3_vec(out[0,...], cell_center_3d_scaled)
    noise3_vec(out[1,...], cell_center_3d_scaled)
    noise3_vec(out[2,...], cell_center_3d_scaled)
    noise3_vec(out[3,...], cell_center_3d_scaled)
    out[0, ...] = (out[0, ...] + 1.0)*0.5*0.4e-3 + 1.3e-3
    out[1:, ...] = out[1:, ...] * 1.6e-3

    tmp = out[1,...]*cell_center_3d[0,...] + out[2,...]*cell_center_3d[1,...] + out[3,...]*cell_center_3d[2,...]
    out[1,...] -= tmp*cell_center_3d[0,...]
    out[2,...] -= tmp*cell_center_3d[1,...]
    out[3,...] -= tmp*cell_center_3d[2,...]

    return torch.from_numpy(out)

def fvm_flux_ur(q0, vert_3d, cell_center, cell_area, dt, r, nhalo):
    """
    Finite volume method for shallow water equation
    q0: double[4, M+nhalo*2, N+nhalo*2]
    vert_3d: double[3, M+1, N+1]
    cell_center: double[3, M, N]
    cell_area: double[M, N]
    dt: double
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 2 > 0

    #q0l 0 q0u 1 q0r 2 q0b 3
    #compute x-direction slope
    fdx = q0[:, nhalo:-nhalo+2, nhalo-1:-nhalo+1] - q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1]
    bdx = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - q0[:, nhalo-2:-nhalo, nhalo-1:-nhalo+1]
    cdx = 0.5 * (q0[:, nhalo:-nhalo+2, nhalo-1:-nhalo+1] - q0[:, nhalo-2:-nhalo, nhalo-1:-nhalo+1])
    slope_x = limiter(fdx, cdx, bdx)
    #compute y-direction slope
    fdy = q0[:, nhalo-1:-nhalo+1, nhalo:-nhalo+2] - q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1]
    bdy = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - q0[:, nhalo-1:-nhalo+1, nhalo-2:-nhalo]
    cdy = 0.5 * (q0[:, nhalo-1:-nhalo+1, nhalo:-nhalo+2] - q0[:, nhalo-1:-nhalo+1, nhalo-2:-nhalo])
    slope_y = limiter(fdy, cdy, bdy)

    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]
    #M+1, N+1
    q0l = torch.zeros_like(q0)
    q0r = torch.zeros_like(q0)
    q0u = torch.zeros_like(q0)
    q0b = torch.zeros_like(q0)

    q0l[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - 0.5*slope_x
    q0r[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] + 0.5*slope_x
    q0u[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] + 0.5*slope_y
    q0b[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - 0.5*slope_y

    qrp = q0l[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qrm = q0r[:, nhalo:-nhalo, nhalo:-nhalo]
    
    qup = q0b[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qum = q0u[:, nhalo:-nhalo, nhalo:-nhalo]

    flux_u = flux(qum, qup, vertul, vertur, r)
    flux_r = flux(qrm, qrp, vertur, vertbr, r)

    return flux_u, flux_r

def integrate(ds_out, q_init, vert_3d, cell_center, cell_area, r, nhalo, downsample_ratio=1, save_interval=10, output_flux=True):
    """
    Solve shallow water equation on sphere
    ds_out.flux_u: double[nT, 4, M//downsample_ratio, N//downsample_ratio]
    ds_out.flux_r: double[nT, 4, M//downsample_ratio, N//downsample_ratio]
    ds_out.q: double[nT, 4, M//downsample_ratio, N//downsample_ratio]
    q_init: double[4, M, N], initial condition
    vert_3d: double[3, M+1, N+1], matrix of vertices
    cell_center: double[3, M, N], matrix of cell area 
    cell_area: double[M, N], matrix of cell area 
    """
    device = q_init.device
    nT = ds_out.q.shape[0]
    _, M, N = q_init.shape
    q1 = boundary_condition(q_init, nhalo)

    T = 0.0
    it = 0
    save_it = 0

    while save_it < nT:
        dt = 0.001*(128/max(N, M/2))
        q0 = q1
        q1 = fvm(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn=None)
        it += 1
        T += dt
        if it % save_interval == 0:
            #tmp = torch.zeros((2, M, N), device=device)
            #tmp[0, ...] = 0.5 * torch.square(q1[0, nhalo:-nhalo, nhalo:-nhalo]) * g
            #tmp[1, ...] = 0.5 * (torch.square(q1[1, nhalo:-nhalo, nhalo:-nhalo]) + torch.square(q1[2, nhalo:-nhalo, nhalo:-nhalo]) + torch.square(q1[3, nhalo:-nhalo, nhalo:-nhalo])) / q1[0, nhalo:-nhalo, nhalo:-nhalo]
            #sum_ = torch.sum(tmp * cell_area[None, ...]).item()
            #print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)
            # TODO: using area as weight
            ds_out.q.values[save_it, ...] = F.avg_pool2d(q1[:, nhalo:-nhalo, nhalo:-nhalo].unsqueeze(0), (downsample_ratio, downsample_ratio)).squeeze(0).cpu().numpy()
            if output_flux:
                flux_u, flux_r = fvm_flux_ur(q0, vert_3d, cell_center, cell_area, dt, r, nhalo)
                ds_out.flux_u.values[save_it, ...] = F.avg_pool2d(flux_u[:, :, downsample_ratio-1::downsample_ratio].unsqueeze(0), (downsample_ratio, 1)).squeeze(0).cpu().numpy() * downsample_ratio
                ds_out.flux_r.values[save_it, ...] = F.avg_pool2d(flux_r[:, downsample_ratio-1::downsample_ratio, :].unsqueeze(0), (1, downsample_ratio)).squeeze(0).cpu().numpy() * downsample_ratio
            save_it += 1

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    M = 2000
    N = M // 2
    
    downsample_ratio = 8
    assert M % downsample_ratio == 0 and N % downsample_ratio == 0
    x_range = (-3., 1.)
    y_range = (-1., 1.)
    r = 1.0
    nhalo = 3
    nsimulations = 7 #70
    scale_factor = 4.0
    output_path = "/scratch/lhuang/SWE/training_data"
    
    #os.system(f"rm -r {output_path}/{file_name}")

    vert_3d, _, cell_area, cell_center_3d = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
    vert_3d_lowres, vert_lonlat_lowres, cell_area_lowres, cell_center_3d_lowres = MeshGrid.sphere_grid(x_range, y_range, M//downsample_ratio, N//downsample_ratio, r)

    output_flux = False

    if output_flux:
        nT = 256
        save_interval = 10
        ds_out_template = xr.Dataset({
            "q": (["sample", "variable", "x", "y"], np.zeros((nT, 4, M//downsample_ratio, N//downsample_ratio))),
            "flux_u": (["sample", "variable", "x", "y"], np.zeros((nT, 4, M//downsample_ratio, N//downsample_ratio))),
            "flux_r": (["sample", "variable", "x", "y"], np.zeros((nT, 4, M//downsample_ratio, N//downsample_ratio))),
            "vert_3d": (["dimension_3d", "xv", "yv"], vert_3d_lowres),
            "vert_lonlat": (["dimension_lonlat", "xv", "yv"], vert_lonlat_lowres),
            "cell_area": (["x", "y"], cell_area_lowres),
            "cell_center_3d": (["dimension_3d", "x", "y"], cell_center_3d_lowres),
        }, attrs={"nT":nT, "downsample_ratio": downsample_ratio, "save_interval": save_interval, "M": M//downsample_ratio, "N": N//downsample_ratio, "scale_factor": scale_factor})
        file_name = f"sphere_torch_{M}_{downsample_ratio}x_randvel_highfreqinit.zarr"
    else:
        nT = 2560
        save_interval = downsample_ratio
        ds_out_template = xr.Dataset({
            "q": (["sample", "variable", "x", "y"], np.zeros((nT, 4, M//downsample_ratio, N//downsample_ratio))),
            "vert_3d": (["dimension_3d", "xv", "yv"], vert_3d_lowres),
            "vert_lonlat": (["dimension_lonlat", "xv", "yv"], vert_lonlat_lowres),
            "cell_area": (["x", "y"], cell_area_lowres),
            "cell_center_3d": (["dimension_3d", "x", "y"], cell_center_3d_lowres),
        }, attrs={"nT":nT, "downsample_ratio": downsample_ratio, "save_interval": save_interval, "M": M//downsample_ratio, "N": N//downsample_ratio, "scale_factor": scale_factor})
        file_name = f"sphere_torch_{M}_{downsample_ratio}x_randvel_iter_heun_downsampleT_highfreqinit.zarr"
        
    file_exist = os.path.exists(f"{output_path}/{file_name}")
    synchronizer = zarr.ProcessSynchronizer(f"{output_path}/{file_name}.sync")

    for isimulation in tqdm(range(nsimulations)):
        q_init = initial_condition_perlin_vectorized(cell_center_3d, scale_factor=scale_factor)
        q_init = q_init.to(device)
        vert_3d = vert_3d.to(device)
        cell_area = cell_area.to(device)
        cell_center_3d = cell_center_3d.to(device)
        ds_out = ds_out_template.copy()
        integrate(ds_out, q_init, vert_3d, cell_center_3d, cell_area, r, nhalo, downsample_ratio, save_interval=save_interval, output_flux=output_flux)

        append_dim = "sample" if isimulation > 0 or file_exist else None
        ds_out.chunk({"sample":64}).to_zarr(f"{output_path}/{file_name}", mode="a", append_dim=append_dim, synchronizer=synchronizer)

"""
    import matplotlib.pyplot as plt
    plt.imshow(ds_out.q[-1,0,...])
    plt.colorbar()
    plt.savefig("qout.png")
"""