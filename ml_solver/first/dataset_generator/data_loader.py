import meshio
import numpy as np
import xesmf as xe
import xarray as xr
import os
import re 
from tqdm import tqdm
import zarr

def read_mesh_data(filename, M, N):
    """
    Read data from XDMF file
    """
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        verts, cells = reader.read_points_cells()
        data = np.zeros((reader.num_steps, 4, M, N))
        data_type = ["h", "hu", "hv", "hw"]
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k)
            for j in range(len(data_type)):
                data_tmp = cell_data[data_type[j]][0].reshape(M, N)
                data[k, j, :, :] = data_tmp[...]            #(num_steps, 4, M, N)
    return data

def transfer_circle(x0, y0, r):
    """
    Generate structure sphere mesh grid
    """
    r1 = r
    d = np.maximum(np.abs(x0), np.abs(y0))
    d = np.maximum(d ,10**-10)
    
    D = r1 * d * (2-d)/np.sqrt(2)
    R = r1
    center = D - np.sqrt(R**2 - D**2)
    xp = D/d * np.abs(x0)
    yp = D/d * np.abs(y0)

    i,j = np.where(np.abs(y0) >= np.abs(x0))
    yp[i,j] = center[i,j] + np.sqrt(R**2 - xp[i,j]**2)
    
    i,j = np.where(np.abs(x0) >= np.abs(y0))
    xp[i,j] = center[i,j] + np.sqrt(R**2 - yp[i,j]**2)

    xp = np.sign(x0) * xp
    yp = np.sign(y0) * yp
    return xp, yp 

def transfer_sphere_surface(vert_3d, vert, r):
    assert vert_3d.shape[0:2] == vert.shape[0:2]
    
    x0 = vert[:,:,0]
    y0 = vert[:,:,1]
    i, j = np.where(x0 < -1)
    x0[i,j] = -2 -x0[i,j]
    xp, yp = transfer_circle(x0, y0, r)
    zp = np.sqrt(r**2 - (xp**2 + yp**2))
    zp[i,j] = -zp[i,j]
    vert_3d[:] = np.stack((xp, yp, zp), axis=-1)

def xyz_to_latlon(vert_3d, r):
    """
    Convert Sphere point from 3d coordinate to lnglat
    vert_sphere: double[M, N, 2], matrix of sphere vertices
    vert_3d: double[M, N, 3], matrix of 3d vertices
    """
    M, N = vert_3d.shape[0], vert_3d.shape[1]
    vert_sphere = np.zeros((M, N, 2))
    vert_sphere[:,:,0] = np.rad2deg(np.arcsin(vert_3d[:,:,2]/r))
    vert_sphere[:,:,1] = np.rad2deg(np.arctan2(vert_3d[:,:,1], vert_3d[:,:,0]))
    return vert_sphere

def sphere_grid(x_range, y_range, M, N, r):
    """
    generate a sphere mesh grid with shape (M, N)
    """
    #initial regular mesh grid
    x = np.linspace(*x_range, M+1)
    y = np.linspace(*y_range, N+1)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    #initial vert_3d
    vert_3d = np.zeros((M+1, N+1, 3))
    #generate sphere grid and yield 3d
    transfer_sphere_surface(vert_3d, vert, r)
    return vert_3d

def sphere_cell_center(x_range, y_range, M, N, r):
    cell_center_3d = np.zeros((M, N, 3))
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = np.linspace(xmin+dx/2, xmax-dx/2, M)
    y = np.linspace(ymin+dy/2, ymax-dy/2, N)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    transfer_sphere_surface(cell_center_3d, vert, r)
    return cell_center_3d

def vector3d_norm(data):
    dis = np.linalg.norm(data, axis=2)
    data[:,:,0] /= dis
    data[:,:,1] /= dis
    data[:,:,2] /= dis
    return data

def expand_boundary(data_center, data_hori, data_vert):
    num_steps, M, N = data_center.shape[0], data_center.shape[2], data_center.shape[3]
    assert data_hori.shape == (num_steps, 4, M+1, N)
    assert data_vert.shape == (num_steps, 4, M, N+1)

    data_true = np.zeros((num_steps, 4, M, N, 4))
    data_left = data_hori[:, :, :-1, :]
    data_up = data_vert[:,:, :, 1:]
    data_right = data_hori[:, :, 1:, :]
    data_bottom = data_vert[:, :, :, :-1]
    data_true[:, :, :, :, 0] = data_left[...]
    data_true[:, :, :, :, 1] = data_up[...]
    data_true[:, :, :, :, 2] = data_right[...]
    data_true[:, :, :, :, 3] = data_bottom[...]
    return data_center, data_true

def Regridding(data, regridder_center, regridder_vert, regridder_hori):
    data_center = regridder_center(data)
    data_hori = regridder_hori(data.swapaxes(-1,-2)).swapaxes(-1,-2)
    data_vert = regridder_vert(data.swapaxes(-1,-2)).swapaxes(-1,-2)
    data_train, data_true = expand_boundary(data_center, data_hori, data_vert)
    return data_train, data_true

if __name__ == "__main__":
    xmin, xmax = -3., 1.
    ymin, ymax = -1., 1.
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)
    r = 1.0
    M, N = 1000, 500
    scale = 10
    num_steps = 101
    #Initialize original grid
    vert_3d = sphere_grid(x_range, y_range, M, N, r)
    vert_sphere = xyz_to_latlon(vert_3d, r)
    cell_center_3d = sphere_cell_center(x_range, y_range, M, N, r)
    cell_center_sphere = xyz_to_latlon(cell_center_3d, r)
    lat_center, lon_center = cell_center_sphere[:,:,0], cell_center_sphere[:,:,1]
    lat_vert, lon_vert = vert_sphere[:,:,0], vert_sphere[:,:,1]
    #Initialize scaling grid
    vert_3d_scale = sphere_grid(x_range, y_range, M//scale, N//scale, r)
    vert_sphere_scale = xyz_to_latlon(vert_3d_scale, r)
    cell_center_3d_scale = sphere_cell_center(x_range, y_range, M//scale, N//scale, r)
    cell_center_sphere_scale = xyz_to_latlon(cell_center_3d_scale, r)
    lat_center_scale, lon_center_scale = cell_center_sphere_scale[:,:,0], cell_center_sphere_scale[:,:,1]
    lat_vert_scale, lon_vert_scale = vert_sphere_scale[:,:,0], vert_sphere_scale[:,:,1]

    #Regridding to low resolution center
    grid_in_center = {"lon": lon_center, "lat": lat_center, "lon_b": lon_vert, "lat_b": lat_vert}
    grid_out_center = {"lon": lon_center_scale, "lat": lat_center_scale, "lon_b": lon_vert_scale, "lat_b": lat_vert_scale}

    #Regridding to low resolution up and right boundary
    grid_in_boundary = {"lon": lon_center.T, "lat": lat_center.T}
    #up boundary
    vert_3d_vert_out_3d = 0.5*(vert_3d_scale[0:-1, :] + vert_3d_scale[1:, :])
    vert_3d_vert_out_3d = vector3d_norm(vert_3d_vert_out_3d)
    vert_3d_vert_out = xyz_to_latlon(vert_3d_vert_out_3d, r)
    lat_vert_out = vert_3d_vert_out[:,:,0]
    lng_vert_out = vert_3d_vert_out[:,:,1]
    grid_vert_out = {"lon": lng_vert_out.T, "lat": lat_vert_out.T}
    #right boundary
    vert_3d_hori_out_3d = 0.5*(vert_3d_scale[:, 0:-1] + vert_3d_scale[:, 1:])
    vert_3d_hori_out_3d = vector3d_norm(vert_3d_hori_out_3d)
    vert_3d_hori_out = xyz_to_latlon(vert_3d_hori_out_3d, r)
    lat_hori_out = vert_3d_hori_out[:,:,0]
    lng_hori_out = vert_3d_hori_out[:,:,1]
    grid_hori_out = {"lon": lng_hori_out.T, "lat": lat_hori_out.T}

    #Regridding function
    regridder_center = xe.Regridder(grid_in_center, grid_out_center, "conservative")
    # regridder_center = xe.Regridder(grid_in_center, grid_out_center, "conservative_normed")
    regridder_vert = xe.Regridder(grid_in_boundary, grid_vert_out, "bilinear", periodic=True)
    regridder_hori = xe.Regridder(grid_in_boundary, grid_hori_out, "bilinear", periodic=True)
    # regridder_vert = xe.Regridder(grid_in_boundary, grid_vert_out, "nearest_d2s", periodic=True)
    # regridder_hori = xe.Regridder(grid_in_boundary, grid_hori_out, "nearest_d2s", periodic=True)

    #Reading data from xdmf
    path = f"{os.path.dirname(os.path.realpath(__file__))}/../clawpack/data/"
    # path = f"{os.path.dirname(os.path.realpath(__file__))}/../clawpack/test_data/"
    file_names = os.listdir(path)
    xdmf_data_set = []
    for file_name in file_names:
        match = re.search("\.xdmf$", file_name)
        if match:
            xdmf_data_set.append(f"../clawpack/data/{file_name}")
            # xdmf_data_set.append(f"../clawpack/test_data/{file_name}")
            
    # xdmf_data_set = []
    # for i in range(301):
    #     xdmf_data_set.append(f"../clawpack/data2/classic-{i}.xdmf")
        
    # #Construct training set
    Data_train = zarr.open('data_bilinear_3d/Data_Train.zarr', mode='w', shape=(len(xdmf_data_set)*num_steps, 4, M//scale, N//scale), chunks=(1, 100, M//scale, N//scale), dtype=float)
    Data_true = zarr.open('data_bilinear_3d/Data_True.zarr', mode='w', shape=(len(xdmf_data_set)*num_steps, 4, M//scale, N//scale, 4), chunks=(1, 400, M//scale, N//scale, 4), dtype=float)
    print(Data_train.shape, Data_true.shape)

    with tqdm(range(len(xdmf_data_set)), unit="MB") as pbar:
        for i in pbar:
            data = read_mesh_data(xdmf_data_set[i], M, N)
            data_train, data_true = Regridding(data, regridder_center, regridder_vert, regridder_hori)
            Data_train[i*num_steps:(i+1)*num_steps, ...] = data_train[...]
            Data_true[i*num_steps:(i+1)*num_steps, ...] = data_true[...]
            pbar.set_description("Processing %s" %i)

    print(f"XDMF dataset unpacked completely and Train dataset successfully generate!!  \n Train dataset shape: ", Data_train.shape)