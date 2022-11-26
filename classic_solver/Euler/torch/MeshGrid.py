import numpy as np
import torch
import meshio
import pathlib, shutil

#generate sturcture regular quad mesh grid
def regular_grid(x_range, y_range, M, N):
    """
    generate a regular mesh grid with shape (M, N)
    """
    #initial vert point
    x = torch.linspace(*x_range, M+1, dtype=torch.double)
    y = torch.linspace(*y_range, N+1, dtype=torch.double)
    vert = torch.stack(torch.meshgrid(x, y, indexing="ij"), axis=0)
    dx = (x_range[1] - x_range[0]) / M
    dy = (y_range[1] - y_range[0]) / N
    x_cell = torch.linspace(x_range[0]+dx/2, x_range[1]-dx/2, M, dtype=torch.double)
    y_cell = torch.linspace(y_range[0]+dy/2, y_range[1]-dy/2, N, dtype=torch.double)
    cell_center = torch.stack(torch.meshgrid(x_cell, y_cell, indexing="ij"), axis=0)
    cell_area = torch.ones(M, N, dtype=torch.double)*(dx*dy)
    return vert, cell_area, cell_center
         
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
            q_cell = q.reshape((-1, 4))
            writer.write_data(t, cell_data={"rho": [q_cell[:,0]], "rhou": [q_cell[:,1]], "rhov": [q_cell[:,2]], "E": [q_cell[:,3]]})
        filename = pathlib.Path(filename)
        if str(filename.parent) != ".":
            shutil.move(f"{filename.stem}.h5", str(filename.parent))