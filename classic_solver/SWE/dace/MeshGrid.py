import numpy as np
import meshio
import pathlib, shutil

#generate sturcture regular quad mesh grid
def compute_regular_cell_area(cell_area, vert):
    """
    Compute cell_areas according to coordinates of vertices
    cell_area: double[M, N], matrix of cell area
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    M, N = cell_area.shape
    assert vert.shape == (M+1, N+1, 2)
    for j in range(N):
        for i in range(M):
            vertbl = vert[i, j, :]
            vertul = vert[i, j+1, :]
            vertur = vert[i+1, j+1, :]
            vertbr = vert[i+1, j, :]
            area = 0.5*(vertul[1]*vertur[0]-vertul[0]*vertur[1]
                       +vertur[1]*vertbr[0]-vertur[0]*vertbr[1]
                       +vertbr[1]*vertbl[0]-vertbr[0]*vertbl[1]
                       +vertbl[1]*vertul[0]-vertbl[0]*vertul[1])
            cell_area[i, j] = np.abs(area)

def regular_grid(x_range, y_range, M, N):
    """
    generate a regular mesh grid with shape (M, N)
    """
    x = np.linspace(*x_range, M+1)
    y = np.linspace(*y_range, N+1)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    cell_area = np.zeros((M, N))
    compute_regular_cell_area(cell_area, vert)
    return vert, cell_area
         
def save_quad_grid_vtk(filename, uout, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    u: double[M, N, 3], matrix of state
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]

    u = uout
    vert = vert.reshape((-1, 2))
    u_cell = u.reshape((-1, 3))
    meshio.write_points_cells(f"{filename}.vtk", vert, cells, cell_data={"h": [u_cell[:,0]], "hu": [u_cell[:,1]], "hv": [u_cell[:,2]]})

def save_quad_grid_xdmf(filename, uout, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    u: double[M, N, 3], matrix of state
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    vert = vert.reshape((-1, 2))
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(vert, cells)
        for t, u in uout:
            u_cell = u.reshape((-1, 3))
            writer.write_data(t, cell_data={"h": [u_cell[:,0]], "hu": [u_cell[:,1]], "hv": [u_cell[:,2]]})
        filename = pathlib.Path(filename)
        if str(filename.parent) != ".":
            shutil.move(f"{filename.stem}.h5", str(filename.parent))

#generate structure sphere mesh grid
def transfer_circle(x0, y0, r):
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
    
def xyz_to_latlng(vert_sphere, vert_3d, r):
    """
    Convert Sphere point from 3d coordinate to lnglat
    vert_sphere: double[M, N, 2], matrix of sphere vertices
    vert_3d: double[M, N, 3], matrix of 3d vertices
    """
    vert_sphere[:,:,0] = np.rad2deg(np.arcsin(vert_3d[:,:,2]/r))
    vert_sphere[:,:,1] = np.rad2deg(np.arctan2(vert_3d[:,:,1], vert_3d[:,:,0]))

def d_sphere_3d(point1_3d, point2_3d, r):
    delta = point1_3d - point2_3d
    distance = np.sqrt(delta[...,0]*delta[...,0] + delta[...,1]*delta[...,1] + delta[...,2]*delta[...,2]) / r
    delta_sigma = 2*np.arcsin(distance/2.0)
    return r * delta_sigma

def compute_sphere_cell_area(cell_area, vert_3d, r):
    vertbl = vert_3d[:-1, :-1, :]
    vertul = vert_3d[:-1, 1:, :]
    vertur = vert_3d[1:, 1:, :]
    vertbr = vert_3d[1:, :-1, :]
    length_l = d_sphere_3d(vertbl, vertul, r)
    length_u = d_sphere_3d(vertul, vertur, r)
    length_r = d_sphere_3d(vertur, vertbr, r)
    length_b = d_sphere_3d(vertbr, vertbl, r)
    length_mid = d_sphere_3d(vertul, vertbr, r)
    l1 = (length_l + length_mid + length_b)*0.5
    l2 = (length_u + length_mid + length_r)*0.5
    cell_area[...] = np.sqrt(l1*(l1-length_l)*(l1-length_mid)*(l1-length_b)) + np.sqrt(l2*(l2-length_r)*(l2-length_mid)*(l2-length_u))

def sphere_grid(x_range, y_range, M, N, r):
    """
    generate a sphere mesh grid with shape (M, N)
    """
    #initial regular mesh grid
    x = np.linspace(*x_range, M+1)
    y = np.linspace(*y_range, N+1)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    
    #initial vert_sphere, vert_3d
    vert_3d = np.zeros((M+1, N+1, 3))
    vert_sphere = np.zeros((M+1, N+1, 2))
    cell_area = np.zeros((M, N))
    
    #generate sphere grid and yield 3d and sphere coordinate
    transfer_sphere_surface(vert_3d, vert, r)
    xyz_to_latlng(vert_sphere, vert_3d, r)
    
    #compute cell area
    compute_sphere_cell_area(cell_area, vert_3d, r)
    
    return vert_3d, vert_sphere, cell_area

def sphere_cell_center(x_range, y_range, M, N, r):
    cell_center_3d = np.zeros((M, N, 3))
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = np.linspace(xmin+dx/2, xmax-dx/2, M)
    y = np.linspace(ymin+dy/2, ymax-dy/2, N)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    #initial vert_sphere, vert_3d

    transfer_sphere_surface(cell_center_3d, vert, r)
    return cell_center_3d

def save_sphere_grid_vtk(filename, u, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    u: double[M, N, 3], matrix of state
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    
    vert = vert.reshape((-1, 3))
    u_cell = u.reshape((-1,4))
    meshio.write_points_cells(f"{filename}.vtk", vert, cells, cell_data={"h": [u_cell[:,0]], "hu": [u_cell[:,1]], "hv": [u_cell[:,2]], "hw": [u_cell[:,3]]})

def save_sphere_grid_xdmf(filename, uout, vert):
    """
    save data on structured mesh to vtk format
    filename: str, path to vtk file
    u: double[M, N, 3], matrix of state
    vert: double[M+1, N+1, 2], matrix of vertices
    """
    indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
    cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
    cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
    cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
    vert = vert.reshape((-1, 3))
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(vert, cells)
        for t, u in uout:
            u_cell = u.reshape((-1, 4))
            writer.write_data(t, cell_data={"h": [u_cell[:,0]], "hu": [u_cell[:,1]], "hv": [u_cell[:,2]], "hw": [u_cell[:,3]]})
        filename = pathlib.Path(filename)
        if str(filename.parent) != ".":
            shutil.move(f"{filename.stem}.h5", str(filename.parent))

            