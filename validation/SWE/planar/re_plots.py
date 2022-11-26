import numpy as np
import matplotlib.pyplot as plt
import meshio
import os
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_mesh_data(filename, M, N):
    """
    Read data from XDMF file
    """
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        verts, cells = reader.read_points_cells()
        data = np.zeros((reader.num_steps, 3, M, N))
        data_type = ["h", "hu", "hv"]
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k)
            for j in range(len(data_type)):
                data_tmp = cell_data[data_type[j]][0].reshape(M, N)
                data[k, j, :, :] = data_tmp[...]            #(num_steps, 3, M, N)
    return data


if __name__ == "__main__":
    fluxes = ["rusanov","roe","hlle", "reference"]
    orders = ["2order"]
    periods = [0, 2, 4, 6, 8, 10]
    delta_t = 0.3
    resolution = 400
    path = os.path.dirname(os.path.realpath(__file__))
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(20, 10))
    # subplot_kw={'xticks': [], 'yticks': []}
    # vnorm = mpl.colors.Normalize(vmin=0.64, vmax=1.4)
    for order in orders:
        for i in range(len(fluxes)):
            for j in range(len(periods)):
                if fluxes[i] == "reference":
                    filename = f"{path}/{order}/hllc/quad-fvm-1000-1000-hllc.xdmf"
                    data = read_mesh_data(filename, 1000, 1000)
                else:
                    filename = f"{path}/{order}/{fluxes[i]}/quad-fvm-{resolution}-{resolution}-{fluxes[i]}.xdmf"
                    data = read_mesh_data(filename, resolution, resolution)
                # axs[j,i].imshow(data[periods[j], 0, ...], cmap='viridis', norm=vnorm)
                pcm = axs[i,j].imshow(data[periods[j], 0, ...], cmap='viridis', origin="lower", extent=[0,1,0,1])
                # fig.colorbar(pcm, ax=axs[-1,i])
                if j == 0:
                    axs[i,j].set_title(f"{fluxes[i].capitalize()} t=0s")
                else:
                    axs[i,j].set_title(f"t={round(j*delta_t, 2)}s")
                divider = make_axes_locatable(axs[i,j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(pcm, cax=cax)

    plt.tight_layout()
    plt.savefig("SWE_planar.png")
    plt.show()