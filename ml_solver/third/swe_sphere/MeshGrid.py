import numpy as np
import torch
import meshio
import pathlib, shutil

#generate sturcture regular quad mesh grid
def regular_grid(x_range, y_range, M, N):
    """
    generate a regular mesh grid with shape (M, N)
    """
    x = torch.linspace(*x_range, M+1, dtype=torch.double)
    y = torch.linspace(*y_range, N+1, dtype=torch.double)
    vert = torch.stack(torch.meshgrid(x, y, indexing="ij"), axis=0)
    dx = (x_range[1] - x_range[0]) / M
    dy = (y_range[1] - y_range[0]) / N
    cell_area = torch.ones(M, N, dtype=torch.double)*(dx*dy)
    return vert, cell_area

def guarantee_boundary(vert_3d):
    # periodic boundary condition in x direction
    l = vert_3d[:, 0, :]
    r = vert_3d[:, -1, :]
    m = 0.5*(l+r)
    vert_3d[:, 0, :] = m
    vert_3d[:, -1, :] = m

    # special boundary condition in y direction
    u = vert_3d[:, :, -1]
    b = vert_3d[:, :, 0]
    vert_3d[:, :, -1] = torch.flip(u, [1])
    vert_3d[:, :, 0] = torch.flip(b, [1])
         
def save_quad_grid_vtk(filename, q_out, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    q_out: double[3, M, N], matrix of state
    vert: double[2, M+1, N+1], matrix of vertices
    """
    vert = torch.moveaxis(vert, 0, -1)
    q = torch.moveaxis(q_out, 0, -1)

    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]

    vert = vert.reshape((-1, 2))
    q_cell = q.reshape((-1, 3))
    meshio.write_points_cells(f"{filename}.vtk", vert, cells, cell_data={"h": [q_cell[:,0]], "hu": [q_cell[:,1]], "hv": [q_cell[:,2]]})

def save_quad_grid_xdmf(filename, q_out, vert, ):
    """
    save data on structured mesh to xdmf format
    filename: str, path to xdmf file
    q_out: double[nT, 3, M, N], matrix of state
    vert: double[2, M+1, N+1], matrix of vertices
    """
    vert = torch.moveaxis(vert, 0, 2)                           # [2, M+1, N+1] -> [M+1, N+1, 2]

    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    vert = vert.reshape((-1, 2))
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(vert, cells)
        for t, q in q_out:
            q = np.moveaxis(q, 0, -1)                         # [nT, 3, M, N] -> [nT, M, N, 3]
            if q.shape[2]==3:
                q_cell = q.reshape((-1, 3))
                writer.write_data(t, cell_data={"h": [q_cell[:,0]], "hu": [q_cell[:,1]], "hv": [q_cell[:,2]]})
            elif q.shape[2]==4:
                q_cell = q.reshape((-1, 4))
                writer.write_data(t, cell_data={"h": [q_cell[:,0] + q_cell[:,3]], "hu": [q_cell[:,1]], "hv": [q_cell[:,2]], "bed":[q_cell[:,3]]})
        filename = pathlib.Path(filename)
        if str(filename.parent) != ".":
            shutil.move(f"{filename.stem}.h5", str(filename.parent))


#generate structure sphere mesh grid
def transfer_circle(x0, y0, r):
    d = torch.maximum(torch.abs(x0), torch.abs(y0))
    d = torch.where(d<10**-10, 1e-10, d)
    
    D = r * d * (2.-d)/np.sqrt(2)
    center = D - torch.sqrt(r**2 - D**2)
    xp = D/d * torch.abs(x0)
    yp = D/d * torch.abs(y0)

    yp = torch.where(torch.abs(y0)>=torch.abs(x0), center + torch.sqrt(r**2 - xp**2), yp)
    xp = torch.where(torch.abs(x0)>=torch.abs(y0), center + torch.sqrt(r**2 - yp**2), xp)

    xp = torch.sign(x0) * xp
    yp = torch.sign(y0) * yp
    return xp, yp

def transfer_sphere_surface(vert_3d, vert, r):
    assert vert_3d.shape[1:] == vert.shape[1:]
    x = vert[0, :, :]
    y0 = vert[1, :, :]
    x0 = torch.where(x < -1., -2. - x, x)
    xp, yp = transfer_circle(x0, y0, r)
    zp = torch.sqrt(r**2 - (xp**2 + yp**2))
    zp = torch.where(x < -1., -zp, zp)
    vert_3d[...] = torch.stack((xp, yp, zp), axis=0)

def xyz_to_latlon(vert_sphere, vert_3d, r):
    """
    Convert Sphere point from 3d coordinate to lonlat
    vert_sphere: double[2, M+1, N+1], matrix of latlon vertices
    vert_3d: double[3, M+1, N+1], matrix of 3d vertices
    """
    vert_sphere[0,...] = torch.rad2deg(torch.asin(vert_3d[2,:,:] / r))
    vert_sphere[1,...] = torch.rad2deg(torch.atan2(vert_3d[1,:,:], vert_3d[0,:,:]))

def sphere_grid(x_range, y_range, M, N, r):
    """
    generate a sphere mesh grid with shape (M, N)
    """
    #initial regular mesh grid
    x = torch.linspace(*x_range, M+1, dtype=torch.double)
    y = torch.linspace(*y_range, N+1, dtype=torch.double)
    vert = torch.stack(torch.meshgrid(x, y, indexing="ij"), axis=0)
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin) / M
    dy = (ymax - ymin) / N
    x_cell = torch.linspace(xmin+dx/2, xmax-dx/2, M, dtype=torch.double)
    y_cell = torch.linspace(ymin+dy/2, ymax-dy/2, N, dtype=torch.double)
    cell = torch.stack(torch.meshgrid(x_cell, y_cell, indexing="ij"), axis=0)

    #initial vert_sphere, vert_3d
    vert_3d = torch.zeros(3, M+1, N+1, dtype=torch.double)
    cell_center_3d = torch.zeros(3, M, N, dtype=torch.double)
    vert_sphere = torch.zeros(2, M+1, N+1, dtype=torch.double)
    cell_area = torch.zeros(M, N, dtype=torch.double)
    
    #generate sphere grid and yield 3d and sphere coordinate
    transfer_sphere_surface(vert_3d, vert, r)
    guarantee_boundary(vert_3d)
    xyz_to_latlon(vert_sphere, vert_3d, r)
    transfer_sphere_surface(cell_center_3d, cell, r)

    #compute cell area
    compute_sphere_cell_area(cell_area, vert_3d, r)
    return vert_3d, vert_sphere, cell_area, cell_center_3d

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

def compute_sphere_cell_area(cell_area, vert_3d, r):
    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]
    length_l = d_sphere_3d(vertbl, vertul, r)
    length_u = d_sphere_3d(vertul, vertur, r)
    length_r = d_sphere_3d(vertur, vertbr, r)
    length_b = d_sphere_3d(vertbr, vertbl, r)
    length_mid = d_sphere_3d(vertul, vertbr, r)
    l1 = (length_l + length_mid + length_b)*0.5
    l2 = (length_u + length_mid + length_r)*0.5
    cell_area[...] = torch.sqrt(l1*(l1-length_l)*(l1-length_mid)*(l1-length_b)) + torch.sqrt(l2*(l2-length_r)*(l2-length_mid)*(l2-length_u))

def save_sphere_grid_vtk(filename, q, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    q_out: double[4, M, N], matrix of state
    vert: double[3, M+1, N+1], matrix of vertices
    """
    vert = torch.moveaxis(vert, 0, -1)
    q = torch.moveaxis(q, 0, -1)

    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    
    vert = vert.reshape((-1, 3))
    q_cell = q.reshape((-1, 4))
    meshio.write_points_cells(f"{filename}.vtk", vert, cells, cell_data={"h": [q_cell[:,0]], "hu": [q_cell[:,1]], "hv": [q_cell[:,2]], "hw": [q_cell[:,3]]})


def save_sphere_grid_xdmf(filename, q_out, vert):
    """
    save data on structured mesh to xdmf format
    filename: str, path to xdmf file
    q_out: double[nT, 4, M, N], matrix of state
    vert: double[3, M+1, N+1], matrix of vertices
    """
    vert = torch.moveaxis(vert, 0, 2)                           # [2, M+1, N+1] -> [M+1, N+1, 2]

    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    vert = vert.reshape((-1, 3))
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(vert, cells)
        for t, q in q_out:
            q = np.moveaxis(q, 0, -1)                           # [nT, 3, M, N] -> [nT, M, N, 3]
            q_cell = q.reshape((-1, 4))
            writer.write_data(t, cell_data={"h": [q_cell[:,0]], "hu": [q_cell[:,1]], "hv": [q_cell[:,2]], "hw": [q_cell[:,3]]})
        filename = pathlib.Path(filename)
        if str(filename.parent) != ".":
            shutil.move(f"{filename.stem}.h5", str(filename.parent))