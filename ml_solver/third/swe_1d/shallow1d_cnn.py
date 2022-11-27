import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from shallow1d import flux_roe, flux_naive, g, initial_condition, boundary_condition_periodic, fvm_TVD_RK, fvm_2ndorder_space_classic, fvm_1storder
import shallow1d
from cnn import ConvBlock, LightningCnn
from mpl_toolkits.axes_grid1 import make_axes_locatable

shallow1d.boundary_condition = boundary_condition_periodic
shallow1d.fvm = fvm_TVD_RK
shallow1d.fvm_2ndorder_space = fvm_2ndorder_space_classic
shallow1d.flux = flux_roe
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fvm_step(q0, dx, dt, cnn):
    flux_r_pred = cnn(q0)

    flux_l = flux_r_pred[:, :, :-1]
    flux_r = flux_r_pred[:, :, 1:]

    flux_sum = -flux_l + flux_r

    qc = q0.moveaxis(0, 1)                                                                  #(2, nbatch, N)
    q1 = qc - dt/dx * flux_sum

    q1 = q1.moveaxis(1, 0)                                                                  #(nbatch, 2, N)
    return q1

def fvm_cnn(q0, dx, dt, cnn):
    #stage1
    qs = fvm_step(q0, dx, dt, cnn)
    #stage2
    qss = (3/4)*q0 + (1/4)*fvm_step(qs, dx, dt, cnn)
    #stage3
    q1 = (1/3)*q0 + (2/3)*fvm_step(qss, dx, dt, cnn)
    return q1

def integrate_cnn(q_init, q_out, dx, T,  save_ts, cnn):
    nT, _, N = q_out.shape
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
        dt = 1e-3*(128/N)
        q0 = fvm_cnn(q0.unsqueeze(0), dx, dt, cnn).squeeze(0)
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = torch.zeros((2, N), device=device)
            tmp[0, ...] = 0.5 * torch.square(q0[0, ...]) * g                       #0.5* g*h^2 
            tmp[1, ...] = 0.5 * torch.square(q0[1, ...]) / q0[0, ...]     #u^2*h^2 / h = 0.5 * u^2*h
            sum_ = torch.sum(tmp * dx).item()
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

def energy_cal(q, dx):
    potential = 0.5 * torch.square(q[:, 0, ...]) * g
    kinetic = 0.5 * torch.square(q[:, 1,...])  / q[:, 0, ...]
    energy = ((potential + kinetic) * dx).sum(dim=-1)
    potential = (potential * dx).sum(dim=-1)
    Total = torch.stack((energy, potential), dim=0)
    return Total

if __name__ == "__main__":
    from argparse import ArgumentParser
    from shallow1d import integrate
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=2048, type=int)
    parser.add_argument("--scale", default=8, type=int)
    parser.add_argument("--period", default=1.0, type=float)
    args = parser.parse_args()
    
    N = args.resolution_x
    Te = args.period
    downsample_ratio = args.scale

    x_range = (0., 10.)
    dx = (x_range[1]-x_range[0]) / N

    N_low = N//downsample_ratio
    dx_low = dx * downsample_ratio

    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    save_ts = torch.arange(0, Te+0.001, Te/N, device=device)
    save_ts_low = torch.arange(0, Te+0.001, Te/N_low, device=device)

    q_init = initial_condition(N)
    q_init_low = F.avg_pool1d(q_init.unsqueeze(0), kernel_size = downsample_ratio).squeeze(0)
    print(q_init.shape, q_init_low.shape)

    q_out_high = torch.zeros(len(save_ts), 2, N, device=device)             #high order
    q_out_low_1 = torch.zeros(len(save_ts_low), 2, N_low, device=device)    #first order
    q_out_low_2 = torch.zeros_like(q_out_low_1)                             #second order
    q_out_low_3 = torch.zeros_like(q_out_low_1)                             #high to low
    q_out_cnn = torch.zeros_like(q_out_low_1)                               #cnn

    cnn = LightningCnn.load_from_checkpoint(f"./test-{downsample_ratio}.ckpt")
    cnn.eval()
    cnn.to(device)
    q_init = q_init.to(device)
    q_init_low = q_init_low.to(device)

    print("Classic high resolution result:")
    integrate(q_init, q_out_high, cfl_number=0.1, dx=dx, nhalo=3, T=Te, save_ts=save_ts)

    print("Cnn result:")
    with torch.no_grad():
        integrate_cnn(q_init_low, q_out_cnn, dx_low, Te, save_ts_low, cnn)

    print("Classic low resolution 2nd order result:")
    integrate(q_init_low, q_out_low_2, cfl_number=0.1, dx=dx_low, nhalo=3, T=Te, save_ts=save_ts_low)

    print("Classic low resolution 1st order result:")
    shallow1d.fvm = fvm_1storder
    integrate(q_init_low, q_out_low_1, cfl_number=0.1, dx=dx_low, nhalo=3, T=Te, save_ts=save_ts_low)

    energy1 = energy_cal(q_out_low_1, dx_low)
    energy2 = energy_cal(q_out_low_2, dx_low)
    energy3 = energy_cal(q_out_low_3, dx_low)
    energy4 = energy_cal(q_out_cnn, dx_low)

    q_out_low_3 = F.avg_pool1d(q_out_high, kernel_size = downsample_ratio).cpu().numpy()
    q_out_high = q_out_high.cpu().numpy()
    q_out_cnn = q_out_cnn.cpu().numpy()
    q_out_low_1 = q_out_low_1.cpu().numpy()
    q_out_low_2 = q_out_low_2.cpu().numpy()

    fig, ax = plt.subplots(1, 5, figsize=(20, 12), dpi=400)

    ax1 = ax[0].imshow(q_out_cnn[:,0,:], origin="lower", cmap = "viridis", extent=[0, 1, 0, Te])
    ax[0].set_title(f"cnn")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Time(s)")

    ax2 = ax[1].imshow(q_out_low_1[:, 0, :], origin="lower", cmap = "viridis", extent=[0, 1, 0, Te])
    ax[1].set_title(f"classic low 1st")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Time(s)")

    ax3 = ax[2].imshow(q_out_low_2[:, 0, :], origin="lower", cmap = "viridis", extent=[0, 1, 0, Te])
    ax[2].set_title(f"classic low 2nd")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("Time(s)")

    ax4 = ax[3].imshow(q_out_high[:, 0, :], origin="lower", cmap = "viridis", extent=[0, 1, 0, Te])
    ax[3].set_title(f"classic high")
    ax[3].set_xlabel("x")
    ax[3].set_ylabel("Time(s)")

    ax5 = ax[4].imshow(q_out_low_3[::downsample_ratio, 0, :], origin="lower", cmap = "viridis", extent=[0, 1, 0, Te])
    ax[4].set_title(f"high to low")
    ax[4].set_xlabel("x")
    ax[4].set_ylabel("Time(s)")

    plt.show()
    plt.savefig(f"./test-{downsample_ratio}.jpg")
    plt.clf()

    save_ts_low = save_ts_low.cpu().numpy()
    energy1 = energy1.cpu().numpy()
    energy2 = energy2.cpu().numpy()
    energy3 = energy3.cpu().numpy()
    energy4 = energy4.cpu().numpy()

    #plot 1:
    plt.subplot(1, 2, 1)
    plt.plot(save_ts_low, energy1[0,...], label="classic 1st order")
    plt.plot(save_ts_low, energy2[0,...], label="classic 2nd order")
    plt.plot(save_ts_low, energy4[0,...], label="data-driven solver")   
    plt.xlabel('Time (s)')
    plt.ylabel('Total energy')
    plt.ylim(bottom=energy1[0,0]-5)
    plt.title('Total Energy comparison')
    plt.legend()

    #plot 2:
    plt.subplot(1, 2, 2)
    plt.plot(save_ts_low, energy1[1,...], label="classic 1st order")
    plt.plot(save_ts_low, energy2[1,...], label="classic 2nd orderr")
    plt.plot(save_ts_low, energy4[1,...], label="data-driven solver")   
    plt.xlabel('Time (s)')
    plt.ylabel('Potential Enstrophy')
    plt.ylim(bottom=energy1[1,0]-5)
    plt.title('Potential Enstrophy comparison')
    plt.legend()

    plt.show()
    plt.savefig("./Engergy.png")