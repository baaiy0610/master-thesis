from distutils.filelist import findall
from scipy.interpolate import griddata
import meshio
import numpy as np
import pandas as pd
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
    return {"h": [h], "hu": [hu], "hv": [hv]}

def read_mesh_data(filename):
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        verts, cells = reader.read_points_cells()
        t, _, cell_data = reader.read_data(reader.num_steps - 1)
    return t, verts, cells[0].data, cell_data

#https://www.cnblogs.com/li12242/p/4996299.html
def validation(test_file, ref_file, save_path):
    t1, verts1, cells1, cell_data1 = read_mesh_data(test_file)
    t2, verts2, cells2, cell_data2 = read_mesh_data(ref_file)
    print(t1, t2)
    assert t1 == t2
    test_cell_centers = calc_cell_centers(verts1, cells1)
    ref_cell_centers = calc_cell_centers(verts2, cells2)
    print(ref_cell_centers.shape)
    print("finish")
    interpolated_ref_data = interpolate_to_mesh(test_cell_centers, ref_cell_centers, cell_data2)
    df = pd.DataFrame(columns=['Var', 'Linf(error)', 'L2(error)','L1(error)'])
    print(f"Test file: {test_file}, Ref file: {ref_file}")
    print(f"Var\tLinf(error)\tL2(error)\tL1(error)")
    test_file_n = re.findall(r"\d+",test_file)
    ref_file_n = re.findall(r"\d+",ref_file)
    for k in ["h", "hu", "hv"]:
        data1 = cell_data1[k][0]
        data2 = interpolated_ref_data[k][0]
        delta = data1 - data2
        df = df.append({'Var':k,'Linf(error)': np.abs(delta).max(), 'L2(error)': np.sqrt(np.square(delta).sum()/len(delta)), 'L1(error)': np.abs(delta).mean()}, ignore_index=True)
        print(f"{k}\t{np.abs(delta).max()}\t{np.sqrt(np.square(delta).sum()/len(delta))}\t{np.abs(delta).mean()}")
    print(f"{test_file_n}", f"{ref_file_n}")
    df.to_excel(f'{save_path}/torch-{test_file_n[1]}-{test_file_n[2]}-claw-{ref_file_n[0]}-{ref_file_n[1]}.xlsx')

if __name__ == "__main__":
    fluxes = ["rusanov","roe","hll","hlle","hllc"]
    orders = ["1order", "2order"]
    path = os.path.dirname(os.path.realpath(__file__))
    for order in orders:
        for flux in fluxes:
            validation(f"{path}/{order}/{flux}/quad-fvm-100-100-{flux}.xdmf",f"{path}/sharpclaw-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            validation(f"{path}/{order}/{flux}/quad-fvm-200-200-{flux}.xdmf",f"{path}/sharpclaw-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            validation(f"{path}/{order}/{flux}/quad-fvm-400-400-{flux}.xdmf",f"{path}/sharpclaw-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            validation(f"{path}/{order}/{flux}/quad-fvm-800-800-{flux}.xdmf",f"{path}/sharpclaw-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            validation(f"{path}/{order}/{flux}/quad-fvm-1000-1000-{flux}.xdmf",f"{path}/sharpclaw-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")

            # validation(f"{path}/{order}/{flux}/quad-fvm-100-100-{flux}.xdmf",f"{path}/classic-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            # validation(f"{path}/{order}/{flux}/quad-fvm-200-200-{flux}.xdmf",f"{path}/classic-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            # validation(f"{path}/{order}/{flux}/quad-fvm-400-400-{flux}.xdmf",f"{path}/classic-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            # validation(f"{path}/{order}/{flux}/quad-fvm-800-800-{flux}.xdmf",f"{path}/classic-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")
            # validation(f"{path}/{order}/{flux}/quad-fvm-1000-1000-{flux}.xdmf",f"{path}/classic-1000-1000.xdmf", save_path=f"{path}/{order}/{flux}")