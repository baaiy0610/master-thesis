import meshio
import numpy as np
import os
import re
from tqdm import tqdm
import zarr
import torch
import torch.nn.functional as F
import numpy as np


g = 11489.57219

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

def boundary_condition(q, nhalo, unsqueeze=True):
    """
    q: double[4, M, N]
    return: double[4, M+4, N+4]
    """
    if unsqueeze: 
        q = q.unsqueeze(0)
    # periodic boundary condition for dim 1
    q = F.pad(q, (0, 0, nhalo, nhalo), "circular").squeeze()
    # special boundary condition for dim 2
    q = torch.cat((torch.flip(q[..., 0:nhalo], [-2, -1]), q, torch.flip(q[..., -nhalo:], [-2, -1])), dim=-1)
    return q

def d_sphere_3d(point1, point2, r):
    """
    Compute the great circle length crossing point1 and point2
    point1: double[3, ...]
    point2: double[3, ...]
    return: double[...]
    """
    delta = point1 - point2
    distance = torch.sqrt(torch.square(delta).sum(dim=0, keepdim=True))/r
    delta_sigma = 2*torch.arcsin(0.5*distance)
    return r * delta_sigma

def cross(a, b):
    """
    Compute cross product axb
    a: double[3, ...]
    b: double[3, ...]
    return: double[3, ...]
    """
    out = torch.stack((a[1,...]*b[2,...] - a[2,...]*b[1,...],
                       a[2,...]*b[0,...] - a[0,...]*b[2,...],
                       a[0,...]*b[1,...] - a[1,...]*b[0,...]), dim=0)
    return out

def rotate_3d(point1, point2):
    """
    Compute the unit normal vector crossing edge defined by point1 and point2
    point1: double[3, ...]
    point2: double[3, ...]
    return: double[3, ...]
    """
    mid_point = 0.5*(point1 + point2) 
    mid_point = mid_point / torch.sqrt(torch.square(mid_point).sum(dim=0, keepdim=True))
    edge_vector = point2 - point1
    edge_vector = edge_vector / torch.sqrt(torch.square(edge_vector).sum(dim=0, keepdim=True))
    tmp = cross(mid_point, edge_vector)
    out = tmp / torch.sqrt(torch.square(tmp).sum(dim=0, keepdim=True))
    return out

def velocity_tan(q, vector_nor, vector_tan):
    """
    Convert the velocity in q into 2d tangent plane defined by normal and tangent vectors
    q: double[4, ...]
    vector_nor: double[3, ...]
    vector_tan: double[3, ...]
    return: double[3, ...]
    """
    out = torch.stack((q[0, ...],
                       (q[1:,...]*vector_nor).sum(dim=0),
                       (q[1:,...]*vector_tan).sum(dim=0)), dim=0)
    return out

def velocity_xyz(q, vector_nor, vector_tan):
    """
    Convert the velocity in q into 3D Cartesian plane
    q: double[3, ...]
    vector_nor: double[3, ...]
    vector_tan: double[3, ...]
    return: double[4, ...]
    """
    out = torch.cat((q[0, ...].unsqueeze(0),
                     q[1, ...].unsqueeze(0) * vector_nor + q[2, ...].unsqueeze(0) * vector_tan), dim=0)
    return out

def max_eigenvalue(q):
    """
    Maximum eigenvalue
    q: double, state vector
    """
    h = q[0, ...]
    uh = q[1, ...]
    vh = q[2, ...]
    u = uh / h
    v = vh / h
    res = torch.sqrt(torch.abs(g*h)) + torch.abs(u)
    return res

def roe_average(ql, qr):
    """
    Compute Roe average of ql and qr
    ql: double[3, ...]
    qr: double[3, ...]
    return: hm, um, vm: double[...]
    """
    hl = ql[0, ...]
    uhl = ql[1, ...]
    vhl = ql[2, ...]

    hr = qr[0, ...]
    uhr = qr[1, ...]
    vhr = qr[2, ...]

    ul = uhl/hl
    vl = vhl/hl

    ur = uhr/hr
    vr = vhr/hr

    sqrthl = torch.sqrt(hl)
    sqrthr = torch.sqrt(hr)

    hm = 0.5*(hl+hr)
    um = (sqrthl*ul + sqrthr*ur)/(sqrthl + sqrthr)
    vm = (sqrthl*vl + sqrthr*vr)/(sqrthl + sqrthr)

    return hm, um, vm

def f(q):
    """
    Flux function for shallow water equation on the direction of x-axis
    q: double[3, ...]
    return: double[3, ...]
    """
    h = q[0, ...]
    uh = q[1, ...]
    vh = q[2, ...]
    u = uh/h
    v = vh/h
    out = torch.stack((uh,
                       u*uh + 0.5*g*h*h,
                       u*vh), dim=0)
    return out

def flux_roe(ql, qr, point1, point2, r):
    """
    Roe's numerical flux for shallow water equation
    ql: double[4,...]
    qr: double[4,...]
    point1: double[3,...]
    point2: double[3,...]
    r: double
    """
    #compute the length of the edge
    edge_length = d_sphere_3d(point1, point2, r)
    #compute tan and normal vector
    edge_vector = point2 - point1
    vector_tan = edge_vector / torch.sqrt(torch.square(edge_vector).sum(dim=0, keepdim=True))
    vector_nor = rotate_3d(point1, point2)
    #compute the velocity in ql and qr aligned with normal and tangent vector
    Tql = velocity_tan(ql, vector_nor, vector_tan)
    Tqr = velocity_tan(qr, vector_nor, vector_tan)

    hm, um, vm = roe_average(Tql, Tqr)
    cm = torch.sqrt(g*hm)

    dq = Tqr - Tql
    dq0 = dq[0, ...]
    dq1 = dq[1, ...]
    dq2 = dq[2, ...]

    # eigval_abs * (eigvecs_inv @ dq)
    k0 = torch.abs(um - cm) * (0.5*(um + cm)/cm * dq0 - 0.5/cm * dq1 + 0.0)
    k1 = torch.abs(um) * (-vm * dq0 + 0.0 + 1.0 * dq2)
    k2 = torch.abs(um + cm) * (-0.5*(um - cm)/cm * dq0 + 0.5/cm * dq1 + 0.0)

    d0 = k0 + 0.0 + k2
    d1 = (um - cm) * k0 + 0.0 + (um + cm) * k2
    d2 = vm * k0 + k1 + vm * k2

    diffq = torch.stack((d0, d1, d2), dim=0)
    fql = f(Tql)
    fqr = f(Tqr)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffq)
    out = velocity_xyz(tmp, vector_nor, vector_tan)
    return out

def minmod(a, b, c = None):
    if c is None:
        return torch.sign(a)*torch.minimum(torch.abs(a), torch.abs(b))*(a*b > 0.0)
    else:
        return torch.sign(a)*torch.minimum(torch.minimum(torch.abs(a), torch.abs(b)), torch.abs(c))*(a*b > 0.0)*(a*c > 0.0)

def maxmod(a, b):
    return torch.sign(a)*torch.maximum(torch.abs(a), torch.abs(b))*(a*b > 0.0)

def limiter_minmod(fd, cd, bd):
    return minmod(fd, cd, bd)

def limiter_superbee(fd, bd):
    return maxmod(minmod(2*bd, fd), minmod(bd, 2*fd))

def limiter_mc(fd, cd, bd):
    return minmod(2*fd, cd, 2*bd)

limiter = limiter_mc
flux = flux_roe

def compute_flux(q0, vert_3d, r, nhalo, scale):
    """
    Finite volume method for shallow water equation
    q0: double[4, M+nhalo*2, N+nhalo*2]
    vert_3d: double[3, M+1, N+1]
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 2 > 0
    M, N = q0.shape[1], q0.shape[2]
    q0 = boundary_condition(q0, nhalo)
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

    #q0l 0 q0u 1 q0r 2 q0b 3
    # qc = q0[:, nhalo:-nhalo, nhalo:-nhalo]
    # qlp = q0r[:,nhalo-1:-nhalo-1, nhalo:-nhalo]
    # qlm = q0l[:,nhalo:-nhalo, nhalo:-nhalo]

    qrp = q0l[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qrm = q0r[:, nhalo:-nhalo, nhalo:-nhalo]

    # qbp = q0u[:, nhalo:-nhalo, nhalo-1:-nhalo-1]
    # qbm = q0b[:, nhalo:-nhalo, nhalo:-nhalo]
    
    qup = q0b[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qum = q0u[:, nhalo:-nhalo, nhalo:-nhalo]
    # flux_l = flux(qlm, qlp, vertbl, vertul, r)
    flux_u = flux(qum, qup, vertul, vertur, r)
    flux_r = flux(qrm, qrp, vertur, vertbr, r)
    # flux_b = flux(qbm, qbp, vertbr, vertbl, r)

    flux_u_scale = flux_u.reshape(4, M//scale, scale, N//scale, scale)
    flux_u_scale = flux_u_scale.sum(dim=-1).sum(dim=-2)

    flux_r_scale = flux_r.reshape(4, M//scale, scale, N//scale, scale)
    flux_r_scale = flux_r_scale.sum(dim=-1).sum(dim=-2)

    return torch.stack((flux_u_scale, flux_r_scale), axis=-1)

def create_dataset(data, vert_3d, r, nhalo, scale):
    num_steps, M, N = data.shape[0], data.shape[2], data.shape[3]
    data = torch.from_numpy(data)
    data_true = np.zeros((num_steps, 4, M//scale, N//scale, 2))
    data_train = np.zeros((num_steps, 4, M//scale, N//scale))
    for i in range(num_steps):
        data_true[i, ...] = compute_flux(data[i], vert_3d, r, nhalo, scale).numpy()
        data_train_ = data[i].reshape(4, M//scale, scale, N//scale, scale)
        data_train_ = data_train_.moveaxis(2, -1)
        data_train[i, ...] = data_train_.reshape(4, M//scale, N//scale, scale*scale).mean(-1).numpy()
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

    xdmf_data_set = []
    for i in range(301):
        xdmf_data_set.append(f"../clawpack/data2/classic-{i}.xdmf")

    nhalo = 3
    vert_3d = torch.from_numpy(vert_3d)
    vert_3d = vert_3d.moveaxis(-1, 0)

    # Construct training set
    Data_train = zarr.open('data_flux_ur/Data_Train.zarr', mode='w', shape=(len(xdmf_data_set)*num_steps, 4, M//scale, N//scale), chunks=(1, 100, M//scale, N//scale), dtype=float)
    Data_true = zarr.open('data_flux_ur/Data_True.zarr', mode='w', shape=(len(xdmf_data_set)*num_steps, 4, M//scale, N//scale, 2), chunks=(1, 400, M//scale, N//scale, 4), dtype=float)
    print(Data_train.shape, Data_true.shape)

    with tqdm(range(len(xdmf_data_set)), unit="MB") as pbar:
        for i in pbar:
            data = read_mesh_data(xdmf_data_set[i], M, N)
            data_train, data_true = create_dataset(data, vert_3d, r, nhalo, scale, regridder_center)
            Data_train[i*num_steps:(i+1)*num_steps, ...] = data_train
            Data_true[i*num_steps:(i+1)*num_steps, ...] = data_true
            pbar.set_description("Processing %s" %i)

    print(f"XDMF dataset unpacked completely and Train dataset successfully generate!!  \n Train dataset shape: ", Data_train.shape)