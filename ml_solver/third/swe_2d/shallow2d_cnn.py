import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import scipy.spatial
import numpy as np
import torch.utils.data as Data
import zarr
from torch.utils.data import Dataset
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from shallow2d import flux_roe, flux_rusanov, fvm_TVD_RK, fvm_2ndorder_space_classic, fvm_1storder
from dataset_generator import initial_condition_perlin_noise2d
from cnn import ConvBlock, LightningCnn
import shallow2d
import MeshGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

g = 9.8
flux = flux_roe
shallow2d.fvm_2ndorder_space = fvm_2ndorder_space_classic
shallow2d.flux = flux_roe

def fvm_step(q0, vert, cell_area, dt, cnn):
    M, N = cell_area.shape
    dt = 1e-3*(128/max(M, N))
    flux_l_pred, flux_b_pred = cnn.forward(q0)

    vertbl = vert[:, :-1, :-1]
    vertul = vert[:, :-1, 1:]
    vertur = vert[:, 1:, 1:]
    vertbr = vert[:, 1:, :-1]

    flux_l = flux_l_pred[:, :, :-1, :]
    flux_r = -flux_l_pred[:, :, 1:, :]
    flux_b = flux_b_pred[:, :, :, :-1]
    flux_u = -flux_b_pred[:, :, :, 1:]

    flux_sum = flux_l + flux_u + flux_r + flux_b

    qc = q0.moveaxis(0, 1)                                                                  #(3, nbatch, M, N)
    q1 = qc + -dt/cell_area * flux_sum

    q1 = q1.moveaxis(1, 0)                                                                  #(nbatch, 3, M, N)
    return q1

def fvm_cnn(q0, vert, cell_area, dt, cnn):
    #stage1
    qs = fvm_step(q0, vert, cell_area, dt, cnn)
    #stage2
    qss = (3/4)*q0 + (1/4)*fvm_step(qs, vert, cell_area, dt, cnn)
    #stage3
    q1 = (1/3)*q0 + (2/3)*fvm_step(qss, vert, cell_area, dt, cnn)
    return q1

def integrate_cnn(q_out, q_init, vert, cell_area, Te, save_ts, cnn):
    nT, _, M,  N = q_out.shape
    assert nT == len(save_ts)
    device = q_out.device

    T = 0.0
    it = 0
    save_it = 0
    q0 = q_init
    while save_it < nT:
        if save_ts[save_it] - T < 1e-10:
            q_out[save_it, ...] = q0[...]
            save_it += 1
            if save_it == nT:
                break
        dt = 1e-3*(128/max(N, M))
        q0 = fvm_cnn(q0.unsqueeze(0), vert, cell_area, dt, cnn).squeeze(0)
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = torch.zeros((2, M, N), dtype=torch.double, device=device)
            tmp[0, ...] = 0.5 * torch.square(q0[0, ...]) * g                      
            tmp[1, ...] = 0.5 * (torch.square(q0[1, ...]) + torch.square(q0[2, ...])) / q0[0, ...]
            sum_ = torch.sum(tmp * cell_area).item()
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

def energy_cal(q, cell_area):
    potential = 0.5 * torch.square(q[:, 0, ...]) * g
    kinetic = 0.5 * (torch.square(q[:, 1,...]) + torch.square(q[:, 2, ...])) / q[:, 0, ...]
    energy = ((potential + kinetic) * cell_area).sum(dim=(-2, -1))
    potential = (potential * cell_area).sum(dim=(-2, -1))
    Total = torch.stack((energy, potential), dim=0)
    return Total

if __name__ == "__main__":
    from argparse import ArgumentParser
    from shallow2d import integrate
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=1024, type=int)
    parser.add_argument("--resolution_y", default=1024, type=int)
    parser.add_argument("--scale", default=16, type=int)
    parser.add_argument("--period", default=1.0, type=float)
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    downsample_ratio = args.scale
    M = args.resolution_x//downsample_ratio
    N = args.resolution_y//downsample_ratio
    Te = args.period

    x_range = (0., 10.)
    y_range = (0., 10.)
    nhalo = 3
    save_ts = torch.arange(0, args.period+0.01, args.period/4., device=device)

    q_init = initial_condition_perlin_noise2d(M, N)
    vert, cell_area = MeshGrid.regular_grid(x_range, y_range, M, N)
    q_out = torch.zeros((len(save_ts), 3, M, N), dtype=torch.double, device=device)         #cnn
    q_out2 = torch.zeros((len(save_ts), 3, M, N), dtype=torch.double, device=device)        #1storder
    q_out3 = torch.zeros((len(save_ts), 3, M, N), dtype=torch.double, device=device)        #2ndorder

    cnn = LightningCnn.load_from_checkpoint(f"./test-{downsample_ratio}.ckpt")
    cnn.eval()

    cnn.to(device)
    q_init = q_init.to(device)
    vert = vert.to(device)
    cell_area = cell_area.to(device)

    print("Cnn result:")
    with torch.no_grad():
        integrate_cnn(q_out, q_init, vert, cell_area, Te, save_ts, cnn)

    print("Classic low result:")
    shallow2d.fvm = fvm_1storder
    integrate(q_out2, q_init, vert, cell_area, Te, save_ts, nhalo)

    print("Classic high result:")
    shallow2d.fvm = fvm_TVD_RK
    integrate(q_out3, q_init, vert, cell_area, Te, save_ts, nhalo)

    energy1 = energy_cal(q_out2, cell_area)
    energy2 = energy_cal(q_out3, cell_area)
    energy3 = energy_cal(q_out, cell_area)

    q_out = q_out.cpu().numpy()                 
    q_out2 = q_out2.cpu().numpy()
    q_out3 = q_out3.cpu().numpy()
    vert = vert.cpu()

    save_ts = save_ts.cpu().numpy()
    dataset = np.array([q_out, q_out2, q_out3])
    fname = ["cnn", "classic 1st", "classic 2nd"]

    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 15), dpi=200)
    for i in range(len(fname)):
        for j in range(5):
            pcm = axs[i,j].imshow(dataset[i, j, 0, ...], cmap='viridis', origin="lower", extent=[0,10,0,10])
            if j == 0:
                axs[i,j].set_title(f"{fname[i].capitalize()} t=0s")
            else:
                axs[i,j].set_title("t={time}s".format(time = '%.1f'%save_ts[j]))
            # divider = make_axes_locatable(axs[i,j])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # fig.colorbar(pcm, cax=cax)
            axs[i,j].set_xlabel('x')
            axs[i,j].set_ylabel('y')
            # axs[i,j].axis("off")

    plt.tight_layout()
    plt.savefig(f"test-cnn-{downsample_ratio}.png")
    plt.show()

    energy1 = energy1.cpu().numpy()
    energy2 = energy2.cpu().numpy()
    energy3 = energy3.cpu().numpy()

    fig, ax = plt.subplots(1, 5, figsize=(20, 12), dpi=400)
    #plot 1:
    plt.subplot(1, 2, 1)
    plt.plot(save_ts, energy1[0,...], label="classic 1st order")
    plt.plot(save_ts, energy2[0,...], label="classic 2nd order")
    plt.plot(save_ts, energy3[0,...], label="data-driven solver")   
    plt.xlabel('Time (s)')
    plt.ylabel('Total energy')
    plt.ylim(bottom=energy1[0,0] - 10)
    plt.title('Total Energy comparison')
    plt.legend()

    #plot 2:
    plt.subplot(1, 2, 2)
    plt.plot(save_ts, energy1[1,...], label="classic 1st order")
    plt.plot(save_ts, energy2[1,...], label="classic 2nd order")
    plt.plot(save_ts, energy3[1,...], label="data-driven solver")  
    plt.xlabel('Time (s)')
    plt.ylabel('Potential Enstrophy')
    plt.ylim(bottom=energy1[1,0] - 30)
    plt.title('Potential Enstrophy comparison')
    plt.legend()

    plt.show()
    plt.savefig("./Engergy.png")