from distutils.filelist import findall
from scipy.interpolate import griddata
import meshio
import numpy as np
import re
import os

def calc_cell_centers(verts, cells):
    ncell, nvert = cells.shape
    _, ndim = verts.shape
    cells_flatten = cells.reshape(-1)
    verts_in_cell = verts[cells_flatten, :].reshape((ncell, nvert, ndim))
    return verts_in_cell.mean(axis=1)
    
def interpolate_to_mesh(cell_centers_target, cell_centers, cell_data):
    h = griddata(cell_centers, cell_data["h"][0], cell_centers_target, method='nearest')
    hu = griddata(cell_centers, cell_data["hu"][0], cell_centers_target, method='nearest')
    hv = griddata(cell_centers, cell_data["hv"][0], cell_centers_target, method='nearest')
    hw = griddata(cell_centers, cell_data["hw"][0], cell_centers_target, method='nearest')

    return {"h": [h], "hu": [hu], "hv": [hv], "hw": [hw]}

def read_mesh_data(filename):
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        verts, cells = reader.read_points_cells()
        t, _, cell_data = reader.read_data(reader.num_steps - 1)
    return t, verts, cells[0].data, cell_data

#https://www.cnblogs.com/li12242/p/4996299.html
def validation(test_file, ref_file, save_path):
    t1, verts1, cells1, cell_data1 = read_mesh_data(test_file)
    t2, verts2, cells2, cell_data2 = read_mesh_data(ref_file)
    assert t1 == t2
    test_cell_centers = calc_cell_centers(verts1, cells1)
    ref_cell_centers = calc_cell_centers(verts2, cells2)
    interpolated_ref_data = interpolate_to_mesh(test_cell_centers, ref_cell_centers, cell_data2)
    print(f"Test file: {test_file}, Ref file: {ref_file}")
    data1 = cell_data1["h"][0]
    data2 = interpolated_ref_data["h"][0]
    delta = data1 - data2
    print("Error percent: {:.2f}%".format((abs(delta)/abs(data2)).mean()*100))

if __name__ == "__main__":
    fluxes = ["rusanov","roe","hll","hlle","hllc"]
    periods = ["1s"]

    path = os.path.dirname(os.path.realpath(__file__))
    for period in periods:
        print(period)
        for flux in fluxes:
            validation(f"{path}/{period}/{flux}/sphere-fvm-100-50-{flux}-classic.xdmf",f"{path}/{period}/classic-1000-500-notrans.xdmf", save_path=f"{path}/{period}/{flux}")
            validation(f"{path}/{period}/{flux}/sphere-fvm-200-100-{flux}-classic.xdmf",f"{path}/{period}/classic-1000-500-notrans.xdmf",save_path=f"{path}/{period}/{flux}")
            validation(f"{path}/{period}/{flux}/sphere-fvm-400-200-{flux}-classic.xdmf",f"{path}/{period}/classic-1000-500-notrans.xdmf",save_path=f"{path}/{period}/{flux}")