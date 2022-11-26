import numpy as np
import meshio
import pathlib, shutil

def regular_grid(x_range, y_range, M, N):
    """
    generate a regular mesh grid with shape (M, N)
    """
    x = np.linspace(*x_range, M+1)
    y = np.linspace(*y_range, N+1)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    dx = (x_range[1] - x_range[0]) / M
    dy = (y_range[1] - y_range[0]) / N
    cell_area = np.ones((M, N))*(dx*dy)
    return vert, cell_area
         
def save_quad_grid_vtk(filename, qout, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    q: double[M, N, 4], matrix of state
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]

    q = qout
    vert = vert.reshape((-1, 2))
    q_cell = q.reshape((-1, 4))
    meshio.write_points_cells(f"{filename}.vtk", vert, cells, cell_data={"rho": [q_cell[:,0]], "rhou": [q_cell[:,1]], "rhov": [q_cell[:,2]], "E": [q_cell[:,3]]})

def save_quad_grid_xdmf(filename, qout, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    q: double[M, N, 4], matrix of state
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    vert = vert.reshape((-1, 2))
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(vert, cells)
        for t, q in qout:
            q_cell = q.reshape((-1, 4))
            writer.write_data(t, cell_data={"rho": [q_cell[:,0]], "rhou": [q_cell[:,1]], "rhov": [q_cell[:,2]], "E": [q_cell[:,3]]})
        filename = pathlib.Path(filename)
        if str(filename.parent) != ".":
            shutil.move(f"{filename.stem}.h5", str(filename.parent))