import numpy as np
import torch.utils.data as Data
import zarr
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb
from random import randint

from SWE_sphere import flux_naiive, flux_roe

OFFSET = 32

class SWEDataset(Dataset):
    def __init__(self, file_path, length=None, load=False, scale_flux=1):
        super().__init__()
        self.ds = zarr.open(file_path, "r")#xr.open_zarr(file_path)
        self.vert_3d = torch.as_tensor(self.ds.vert_3d[...])
        self.cell_center_3d = torch.as_tensor(self.ds.cell_center_3d[...])
        self.cell_area = torch.as_tensor(self.ds.cell_area[...])
        self.length = self.ds.q.shape[0] if (length is None) else length
        self.ds_q = self.ds.q
        self.scale_flux = scale_flux
        self.offset = OFFSET
        self.simulation_steps = 2560
        if self.length % self.simulation_steps != 0:
            self.length -= self.length % self.simulation_steps
        if load:
            #perm = np.random.permutation(self.ds.q.shape[0])[:length]
            self.ds_q = self.ds.q[:length, ...]
 
    def __len__(self):
        return self.length - self.offset

    def __getitem__(self, index):
        if self.simulation_steps - index % self.simulation_steps <= self.offset:
            index -= randint(self.offset, self.simulation_steps - 2*self.offset)
        qs = torch.as_tensor(self.ds_q[index:index+self.offset, ...])
        return qs, self.vert_3d, self.cell_center_3d, self.cell_area

class SWEDataset_Sim(Dataset):
    def __init__(self, length, M, N):
        super().__init__()
        import MeshGrid
        import SWE_sphere
        from SWE_sphere import initial_condition, integrate, fvm_2ndorder_space_classic, fvm_TVD_RK, flux_roe
        SWE_sphere.fvm_2ndorder_space = fvm_2ndorder_space_classic
        SWE_sphere.fvm = fvm_TVD_RK
        SWE_sphere.flux = flux_roe
        SWE_sphere.PRINT = False
        dt = 0.001*(128/max(N, M/2))
        device = "cuda"
        x_range = (-3., 1.)
        y_range = (-1., 1.)
        r = 1.0
        nhalo = 3
        vert_3d, vert_lonlat, cell_area, cell_center_3d = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
        q_init = initial_condition(x_range, y_range, M, N, cell_center_3d)
        q_out = torch.zeros((length, 4, M, N), device=device).double()
        save_ts = torch.arange(0., length*dt-dt/2, dt)
        q_init = q_init.to(device).double()
        vert_3d = vert_3d.to(device).double()
        cell_area = cell_area.to(device).double()
        cell_center_3d = cell_center_3d.to(device).double()
        integrate(q_out, q_init, vert_3d, cell_center_3d, cell_area, r, save_ts, nhalo, None)
        self.q_out = q_out.cpu()
        self.offset = OFFSET
        self.length = length
        self.vert_3d = vert_3d.cpu()
        self.cell_center_3d = cell_center_3d.cpu()
        self.cell_area = cell_area.cpu()

    def __len__(self):
        return self.length - self.offset

    def __getitem__(self, index):
        qs = self.q_out[index:index+self.offset, ...]
        return qs, self.vert_3d, self.cell_center_3d, self.cell_area

class SWEDataModule(pl.LightningDataModule):
    def __init__(self, file_path, length=None, load=False, scale_flux=1, batch_size=32):
        super().__init__()
        self.prepare_data_per_node = False
        self.load = load
        self.batch_size = batch_size
        self.swe = SWEDataset(file_path, length, load, scale_flux)

    def train_dataloader(self):
        num_workers = 1 if self.load else 6
        return DataLoader(
            dataset=self.swe,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros((1, 250, 125))), batch_size=1, shuffle=False)


def boundary_padding(q, halo_x, halo_y, unsqueeze=False):
    """
    q: double[4, M, N]
    return: double[4, M+4, N+4]
    """
    M = q.shape[-2]
    N = q.shape[-1]
    if M > N:
        # periodic boundary condition for dim 1
        q = F.pad(q, (0, 0, halo_x, halo_x), "circular")
        # special boundary condition for dim 2
        q = torch.cat((torch.flip(q[..., 0:halo_y], [-2, -1]), q, torch.flip(q[..., -halo_y:], [-2, -1])), dim=-1)
    elif M < N:
        q = F.pad(q, (halo_y, halo_y, 0, 0), "circular")
        q = torch.cat((torch.flip(q[..., 0:halo_x, :], [-2, -1]), q, torch.flip(q[..., -halo_x:, :], [-2, -1])), dim=-2)
    else:
        assert False # not considering M == N
    return q

class Padding(nn.Module):
    def __init__(self, halo_x, halo_y, unsqueeze=False):
        super(Padding, self).__init__()
        self.halo_x = halo_x
        self.halo_y = halo_y
        self.unsqueeze = unsqueeze

    def forward(self, q):
        return boundary_padding(q, self.halo_x, self.halo_y, self.unsqueeze)


class LightningConstraintLayer2D(pl.LightningModule):
    def __init__(self, nstencil_out: int, ninterp: int):
        super(LightningConstraintLayer2D, self).__init__()
        nconstraint = 1
        assert nstencil_out >= nconstraint

        constraint_mat = torch.ones((ninterp, nstencil_out, nconstraint))
        U, S, Vh = torch.linalg.svd(constraint_mat.transpose(1, 2), full_matrices=True)
        self.affinespace_weight = nn.Parameter(Vh.transpose(1, 2)[:, :, nconstraint:], requires_grad=False) # shape: ninterp, nstencil_out, nstencil_out - nconstraint
        B = torch.zeros(ninterp, nconstraint)
        B[:, 0] = 1.0 # zeroth order (for interpolation)
        S_pi = torch.zeros(ninterp, nstencil_out, nconstraint) # Pseudo inverse of Sigma mat
        for i in range(ninterp):
            S_pi[i, :nconstraint, :] = torch.diag(1.0/S[i, :])
        # https://towardsdatascience.com/underdetermined-least-squares-feea1ac16a9
        sol = Vh.transpose(1, 2) @ S_pi @ (U.transpose(1, 2) @ B.unsqueeze(-1)) # Solving underdetermined linear sys using SVD
        self.affinespace_bias = nn.Parameter(sol.squeeze(2), requires_grad=False) # shape: ninterp, nstencil_out

    def forward(self, x, no_bias = False):
        # x shape: nbatch, ny, nx, ninterp, nstencil_out - nconstraint
        x = (self.affinespace_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0) @ x.unsqueeze(-1)).squeeze(-1) # shape: nbatch, ny, nx, ninterp, nstencil_out
        if not no_bias:
            x += self.affinespace_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return x

class ConvBlock(nn.Module):
    def __init__(self, mid_channels, kernel_size, skip_connection=True, batch_norm=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, self.kernel_size)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, self.kernel_size)
        self.batch_norm = batch_norm
        self.skip_connection = skip_connection
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(mid_channels)

    def forward(self, inputs):
        x = boundary_padding(inputs, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.gelu(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.skip_connection:
            return F.gelu(x) + inputs
        else:
            return F.gelu(x)

class SIRENLayer(nn.Module):
    # expect input around [-1,1]
    def __init__(self, infeatrue, outfeature, omega, kernel_size, is_input_layer=False):
        super().__init__()
        self.conv = nn.Conv2d(infeatrue, outfeature, kernel_size)
        self.omega = omega
        if is_input_layer:
            # Flow implicit network paper
            #nn.init.uniform_(self.conv.weight, -1/outfeature, 1/outfeature)
            # SIREN paper
            nn.init.uniform_(self.conv.weight, -np.sqrt(6/infeatrue), np.sqrt(6/infeatrue))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(6/infeatrue)/omega, np.sqrt(6/infeatrue)/omega)
        nn.init.uniform_(self.conv.bias, -1/np.sqrt(infeatrue), 1/np.sqrt(infeatrue))

    def forward(self, x):
        return torch.sin(self.conv(self.omega*x))

class SIRENResBlock(nn.Module):
    def __init__(self, infeature, midfeature, outfeature, omega, kernel_size, skip_connection = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.skip_connection = skip_connection
        self.conv1 = SIRENLayer(infeature, midfeature, omega, kernel_size, False)
        self.conv2 = SIRENLayer(midfeature, outfeature, omega, kernel_size, False)

    def forward(self, inputs):
        x = boundary_padding(inputs, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv1(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv2(x)
        if self.skip_connection:
            return 0.5*(x + inputs)
        else:
            return x

class LightningCnn(pl.LightningModule):
    def __init__(self, mid_channel: int, input_stencil: int, stencil_x: int, stencil_y: int, learning_rate: float, skip_connection: bool = True, batch_norm: bool = True, batch_size: int = 2):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.mid_channel = mid_channel
        self.nstencil_out = stencil_x * stencil_y
        self.ninterp = 2
        self.kernel_size = 3
        self.input_stencil = input_stencil
        self.stencil_x = stencil_x
        self.stencil_y = stencil_y
        self.automatic_optimization = True
        self.batch_size = batch_size
        
        conv_out = nn.Conv2d(self.mid_channel, 4 * self.nstencil_out, 1)
        nn.init.constant_(conv_out.weight, 0)
        nn.init.constant_(conv_out.bias, 0)
        self.conv = nn.Sequential(
            Padding((self.input_stencil-1)//2, (self.input_stencil-1)//2),
            nn.Conv2d(in_channels=4, out_channels=self.mid_channel, kernel_size=self.input_stencil),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            conv_out,
        )
        """
        omega = 15
        self.conv = nn.Sequential(
            Padding((self.input_stencil-1)//2, (self.input_stencil-1)//2),
            SIRENLayer(4, self.mid_channel, omega, self.kernel_size, is_input_layer=True),
            SIRENResBlock(self.mid_channel, self.mid_channel, self.mid_channel, omega, self.kernel_size, skip_connection),
            SIRENResBlock(self.mid_channel, self.mid_channel, self.mid_channel, omega, self.kernel_size, skip_connection),
            SIRENResBlock(self.mid_channel, self.mid_channel, self.mid_channel, omega, self.kernel_size, skip_connection),
            nn.Conv2d(self.mid_channel, 4*self.ninterp * self.nstencil_out, 1),
        )
        """
        #self.constrain = LightningConstraintLayer2D(self.nstencil_out, 4*self.ninterp)


    def forward_qmid(self, inputs, vert_3d):
        # vert_3d: (batch_size, 3, nx+1, ny+1)
        if len(vert_3d.shape) == 4:
            vert_3d = vert_3d[0, ...]
        (batch_size, _, nx, ny) = inputs.shape                                                  #(nbatch, 4, nx, ny)
        #Normalize inputs data
        inputs_normal = inputs - inputs.mean(dim=(2, 3), keepdim=True)
        inputs_normal = inputs_normal / (inputs_normal.abs().amax(dim=(2, 3), keepdim=True) + 1e-5)
        x = inputs_normal.float()                                                                #(nbatch, 4, nx, ny)
        convout_r = torch.tanh(self.conv(x))                                                     #(nbatch, 4 * nstencil_out, nx, ny)
        convout_u = torch.tanh(torch.transpose(self.conv(torch.transpose(x, -2, -1)), -2, -1))
        x = torch.stack((convout_u, convout_r), dim=1)                                           #(nbatch, ninterp, 4*nstencil_out, nx, ny)
        x = x.reshape(batch_size, self.ninterp, 4, self.nstencil_out, nx, ny).moveaxis(-3,-1)    #(nbatch, ninterp, 4, nx, ny, nstencil_out)
        #Compute alpha
        #alpha = x / (x.sum(dim=-1, keepdim=True) + 1e-5)                                         #(nbatch, ninterp, 4, nx, ny, nstencil_out)
        assert self.nstencil_out == 9
        alpha = x - x.mean(dim=-1, keepdim=True) + torch.tensor([[[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]], [[0, 0, 0], [0, 0.5, 0], [0, 0.5, 0]]],device=self.device).view((1, 2, 1, 1, 1, self.nstencil_out))
        #padding for interpolate
        inputs_pad = boundary_padding(inputs, (self.stencil_x-1)//2, (self.stencil_y-1)//2)
        inputs_unfold = inputs_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 4, nx, ny, stencil_x)
        inputs_unfold = inputs_unfold.unfold(3, self.stencil_y, 1)                               #(nbatch, 4, nx, ny, stencil_x, stencil_y)
        inputs_unfold = inputs_unfold.reshape(batch_size, 4, nx, ny, self.nstencil_out)          #(nbatch, 4, nx, ny, nstencil_out)
        inputs_unfold = inputs_unfold.unsqueeze(1)                                               #(nbatch, 1, 4, nx, ny, nstencil_out)
        #Compute boundary data
        q_mid = (inputs_unfold * alpha).sum(dim=-1)                                              #(nbatch, ninterp, 4, nx, ny)
        q_mid = boundary_padding(q_mid.view((batch_size, 2*4, nx, ny)), 1, 1).view((batch_size, 2, 4, nx+2, ny+2))[:, :, :, :-1, :-1] #(nbatch, ninterp, 4, nx+1, ny+1)
        #inputs_pad = boundary_padding(inputs, 1, 1)
        #q_mid = q_mid.sum()*0 + 0.5*torch.stack((inputs_pad[:, :, 0:-1, 0:-1]+inputs_pad[:, :, 0:-1, 1:], inputs_pad[:, :, 0:-1, 0:-1]+inputs_pad[:, :, 1:, 0:-1]), dim=1)
        q_mid = q_mid.moveaxis(2, 0)                                                             #(4, nbatch, ninterp, nx+1, ny+1)
        q_mid_u = q_mid[:, :, 0, 1:, :]                                                          #(4, nbatch, nx, ny+1)
        q_mid_r = q_mid[:, :, 1, :, 1:]                                                          #(4, nbatch, nx+1, ny)
        vert_ul = vert_3d[:, :-1, :].unsqueeze(1)                                                #(3, 1, nx, ny+1)
        vert_ur_u = vert_3d[:, 1:, :].unsqueeze(1)                                               #(3, 1, nx, ny+1)
        vert_ur_r = vert_3d[:, :, 1:].unsqueeze(1)                                               #(3, 1, nx+1, ny)
        vert_br = vert_3d[:, :, :-1].unsqueeze(1)                                                #(3, 1, nx+1, ny)
        flux_u = flux_naiive(q_mid_u, vert_ul, vert_ur_u, 1.0)                                   #(4, nbatch, nx, ny+1)
        flux_r = flux_naiive(q_mid_r, vert_ur_r, vert_br, 1.0)                                   #(4, nbatch, nx+1, ny)
        return flux_u, flux_r

    def forward_slope(self, inputs, vert_3d):
        # vert_3d: (batch_size, 3, nx+1, ny+1)
        if len(vert_3d.shape) == 4:
            vert_3d = vert_3d[0, ...]
        (batch_size, _, nx, ny) = inputs.shape                                                  #(nbatch, 4, nx, ny)
        #Normalize inputs data
        inputs_normal = inputs - inputs.mean(dim=(2, 3), keepdim=True)
        inputs_normal = inputs_normal / (inputs_normal.abs().amax(dim=(2, 3), keepdim=True) + 1e-5)
        x = inputs_normal.float()                                                                #(nbatch, 4, nx, ny)
        convout_x = torch.tanh(self.conv(x))                                                     #(nbatch, 4 * nstencil_out, nx, ny)
        convout_y = torch.tanh(torch.transpose(self.conv(torch.transpose(x, -2, -1)), -2, -1))
        x = torch.stack((convout_y, convout_x), dim=1)                                           #(nbatch, ninterp, 4*nstencil_out, nx, ny)
        x = x.reshape(batch_size, self.ninterp, 4, self.nstencil_out, nx, ny).moveaxis(-3,-1)    #(nbatch, ninterp, 4, nx, ny, nstencil_out)
        #Compute alpha
        #alpha = x / (x.sum(dim=-1, keepdim=True) + 1e-5)                                        #(nbatch, ninterp, 4, nx, ny, nstencil_out)
        alpha = x - x.mean(dim=-1, keepdim=True) + torch.tensor([[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]],device=self.device).view((1, 2, 1, 1, 1, self.nstencil_out))
        #padding for interpolate
        inputs_pad = boundary_padding(inputs, (self.stencil_x-1)//2, (self.stencil_y-1)//2)
        inputs_unfold = inputs_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 4, nx, ny, stencil_x)
        inputs_unfold = inputs_unfold.unfold(3, self.stencil_y, 1)                               #(nbatch, 4, nx, ny, stencil_x, stencil_y)
        inputs_unfold = inputs_unfold.reshape(batch_size, 4, nx, ny, self.nstencil_out)          #(nbatch, 4, nx, ny, nstencil_out)
        inputs_unfold = inputs_unfold.unsqueeze(1)                                               #(nbatch, 1, 4, nx, ny, nstencil_out)
        #Compute boundary data
        qgrad = (inputs_unfold * alpha).sum(dim=-1)                                              #(nbatch, ninterp, 4, nx, ny)
        slope_y = qgrad[:, 0, :, :, :]
        slope_x = qgrad[:, 1, :, :, :]
        vert_ul = vert_3d[:, :-1, :].unsqueeze(1)                                                #(3, 1, nx, ny+1)
        vert_ur_u = vert_3d[:, 1:, :].unsqueeze(1)                                               #(3, 1, nx, ny+1)
        vert_ur_r = vert_3d[:, :, 1:].unsqueeze(1)                                               #(3, 1, nx+1, ny)
        vert_br = vert_3d[:, :, :-1].unsqueeze(1)                                                #(3, 1, nx+1, ny)
        qr = boundary_padding(inputs + 0.5*slope_x, 1, 1).transpose(0, 1)
        ql = boundary_padding(inputs - 0.5*slope_x, 1, 1).transpose(0, 1)
        qu = boundary_padding(inputs + 0.5*slope_y, 1, 1).transpose(0, 1)
        qb = boundary_padding(inputs - 0.5*slope_y, 1, 1).transpose(0, 1)
        flux_u = flux_roe(qu[:, :, 1:-1, :-1], qb[:, :, 1:-1, 1:], vert_ul, vert_ur_u, 1.0)      #(4, nbatch, nx, ny+1)
        flux_r = flux_roe(qr[:, :, :-1, 1:-1], ql[:, :, 1:, 1:-1], vert_ur_r, vert_br, 1.0)      #(4, nbatch, nx+1, ny)
        return flux_u, flux_r

    def forward(self, inputs, vert_3d):
        return self.forward_slope(inputs, vert_3d)

    def fvm_step(self, q0, vert_3d, cell_center_3d, cell_area):
        from SWE_sphere import coriolis, correction, momentum_on_tan
        M, N = cell_area.shape
        downsample_ratio = 8
        dt = 0.001*(128/max(N, M/2))
        r = 1.0

        flux_u_pred, flux_r_pred = self.forward(q0, vert_3d)

        vertbl = vert_3d[:, :-1, :-1]
        vertul = vert_3d[:, :-1, 1:]
        vertur = vert_3d[:, 1:, 1:]
        vertbr = vert_3d[:, 1:, :-1]

        flux_l = - flux_r_pred[:, :, :-1, :]
        flux_r = flux_r_pred[:, :, 1:, :]
        flux_b = - flux_u_pred[:, :, :, :-1]
        flux_u = flux_u_pred[:, :, :, 1:]

        flux_sum = flux_l + flux_u + flux_r + flux_b

        qc = q0.moveaxis(0, 1) # (4, nbatch, M, N)
        fq = coriolis(qc, cell_center_3d.unsqueeze(1), dt)
        cor = correction(qc, vertbl.unsqueeze(1), vertul.unsqueeze(1), vertur.unsqueeze(1), vertbr.unsqueeze(1), r)

        flux_sum = momentum_on_tan(flux_sum, cell_center_3d.unsqueeze(1))
        cor = momentum_on_tan(cor, cell_center_3d.unsqueeze(1))

        q1c_upd = -dt/cell_area * (flux_sum + cor)

        q1 = qc + q1c_upd + fq

        q1 = q1.moveaxis(1, 0) # (nbatch, 4, M, N)
        return q1

    def fvm_tvd_rk3(self, q0, vert_3d, cell_center_3d, cell_area):
        #stage1
        qs = self.fvm_step(q0, vert_3d, cell_center_3d, cell_area)
        #stage2
        qss = (3/4)*q0 + (1/4)*self.fvm_step(qs, vert_3d, cell_center_3d, cell_area)
        #stage3
        q1 = (1/3)*q0 + (2/3)*self.fvm_step(qss, vert_3d, cell_center_3d, cell_area)
        # print(torch.all(q1[0,:,:]>0))
        return q1

    def fvm_heun(self, q0, vert_3d, cell_center_3d, cell_area):
        #stage1
        qs = self.fvm_step(q0, vert_3d, cell_center_3d, cell_area)
        #stage2
        qss = self.fvm_step(qs, vert_3d, cell_center_3d, cell_area)
        q1 = 0.5*(q0+qss)
        # print(torch.all(q1[0,:,:]>0))
        return q1

    def total_energy(self, q, cell_area):
        assert q.shape[1] == 4
        from SWE_sphere import g
        potential = 0.5 * torch.square(q[:, 0, :, :]) * g
        kinetic = 0.5 * (torch.square(q[:, 1, :, :]) + torch.square(q[:, 2, :, :]) + torch.square(q[:, 3, :, :])) / (F.relu(q[:, 0, :, :]) + 1e-5)
        total_energy = ((potential + kinetic) * cell_area).sum(dim=(-2, -1)) # (nbatch,)
        return total_energy

    def training_step(self, train_batch, batch_idx):
        qs, vert_3d, cell_center_3d, cell_area = train_batch
        if len(vert_3d.shape) > 3:
            vert_3d = vert_3d[0, ...]
            cell_center_3d = cell_center_3d[0, ...]
            cell_area = cell_area[0, ...]
        simulation_len = qs.shape[1]
        assert simulation_len == OFFSET
        qi_pred = qs[:, 0, ...]
        loss = torch.tensor(0.0)
        scale = 1e3
        current_len = 1
        threshold = 1.0
        gamma = 0.15

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad(set_to_none=True)
            #loss_fn = lambda x, y: F.mse_loss(x, y)*scale*scale
            # Question 1: reuse qi_pred subgraph
            # Question 2: gradient accumulation: do not call zero grad
            # Question 3: how to select lr
            # Question 4: how to choose optimizer
            # Question 5: loss fun has to have reduction option
            # Adaptive gradient method
            for i in range(1, simulation_len):
                if i % 2 == 0:
                    qi_pred_dummy = self.gm.setup_model_call(checkpoint, self.fvm_step, qi_pred, vert_3d, cell_center_3d, cell_area)
                else:
                    qi_pred_dummy = self.gm.setup_model_call(self.fvm_step, qi_pred, vert_3d, cell_center_3d, cell_area)
                self.gm.setup_loss_call(F.mse_loss, qi_pred_dummy, qs[:, i, ...])
                qi_pred, loss_i = self.gm.forward()
                loss_i *= scale*scale
                if torch.isnan(loss_i).any() or loss_i.item() > threshold:
                    loss = loss + (simulation_len-i)*threshold
                    break
                loss = loss + loss_i
                current_len = i
            loss = loss / simulation_len
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            opt.step()
        else:
            for i in range(1, simulation_len):
                eT = self.total_energy(qi_pred, cell_area)
                if i % 2 == 0:
                    qi_pred = checkpoint(self.fvm_step, qi_pred, vert_3d, cell_center_3d, cell_area)
                else:
                    qi_pred = self.fvm_step( qi_pred, vert_3d, cell_center_3d, cell_area)
                loss_i = F.mse_loss(qi_pred, qs[:, i, ...])*scale*scale
                if torch.isnan(loss_i).any() or loss_i.item() > threshold:
                    loss = loss + (simulation_len-i)*threshold
                    break
                loss = loss + loss_i
                current_len = i
            loss = loss / simulation_len
            e0 = self.total_energy(qs[:, 0, ...], cell_area)
            scale_e = 1e2
            loss_energy = F.mse_loss(e0*scale_e, eT*scale_e)

            loss = loss + loss_energy
        
        #
        #loss_energy = 0.0#F.mse_loss(scale_e*e1, scale_e*e0) + F.mse_loss(scale_e*e2, scale_e*e0) + F.mse_loss(scale_e*e3, scale_e*e0) + F.mse_loss(scale_e*e4, scale_e*e0)
        
        #loss_total = loss + loss_energy

        self.log('train_loss', loss)
        self.log('current_len', current_len)
        self.log('train_loss_energy', loss_energy)
        #self.log('train_loss_total', loss_total)
        return loss
    """
    def validation_step(self, test_batch, batch_idx):
        inputs, vert_3d, flux_u, flux_r = test_batch
        flux_u_pred, flux_r_pred = self.forward(inputs, vert_3d)
        scale = 1e3
        loss = (F.mse_loss(flux_u_pred*scale, flux_u*scale) + F.mse_loss(flux_r_pred*scale, flux_r*scale))
        self.log('val_loss', loss, sync_dist=True)

    """
    def validation_step(self, test_batch, batch_idx):
        import MeshGrid
        import SWE_sphere
        from SWE_sphere import initial_condition, integrate, fvm_2ndorder_space_cnn, fvm_TVD_RK, fvm_2ndorder_space_midpoint, fvm_heun
        SWE_sphere.fvm_2ndorder_space = fvm_2ndorder_space_cnn
        SWE_sphere.fvm = fvm_heun
        SWE_sphere.PRINT = True
        print("")
        if self.trainer.is_global_zero:
            device = self.device
            M, N = test_batch[0].shape
            x_range = (-3., 1.)
            y_range = (-1., 1.)
            r = 1.0
            nhalo = 3
            vert_3d, vert_lonlat, cell_area, cell_center_3d = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
            q_init = initial_condition(x_range, y_range, M, N, cell_center_3d)
            q_out = torch.zeros((1, 4, M, N), device=device)
            save_ts = np.array([1.0])
            q_init = q_init.to(device)
            vert_3d = vert_3d.to(device)
            cell_area = cell_area.to(device)
            cell_center_3d = cell_center_3d.to(device)
            integrate(q_out, q_init, vert_3d, cell_center_3d, cell_area, r, save_ts, nhalo, self)
        #return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, betas=(0.9, 0.99))
        #optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
        #config = asdl.NaturalGradientConfig(data_size=self.batch_size)
        #self.gm = asdl.KfacGradientMaker(self, config)
        #config = asdl.ShampooGradientConfig()
        #self.gm = asdl.ShampooGradientMaker(self, config)
        #self.gm = asdl.GradientMaker(self)
        return optimizer

def main(args):
    model = LightningModule()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    import time
    import os

    parser = ArgumentParser()
    parser.add_argument("--nx_stencil", default=3, type=int)
    parser.add_argument("--ny_stencil", default=3, type=int)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    stencil_x = args.nx_stencil
    stencil_y = args.ny_stencil
    input_stencil = 3
    mid_channel = 64
    batch_size = 4
    learning_rate = 1e-4
    nepoch = 20

    file_path = "/scratch/lhuang/SWE/training_data/sphere_torch_2000_8x_randvel_iter_heun_downsampleT.zarr"
    #val_file_path = "/scratch/lhuang/SWE/training_data/sphere_torch_2000_8x.zarr"
    #Initialize Dataloader
    #Training
    load = False
    length = 50000
    num_workers = 1 if load else 6

    train_dataloader = DataLoader(
            dataset=SWEDataset(file_path, length, load), #SWEDataset_Sim(1000, 250, 125),#
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

    mock_dataset = TensorDataset(torch.zeros((1, 250, 125)))

    wandb_logger = WandbLogger(project="SWE-cnn", name=f"RUN_ResBlock_iter_lr{learning_rate:.0e}_newdata@{datetime.now()}", save_code=True) # settings=wandb.Settings(start_method='thread')
    current_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"ckpt/",filename='cnn_iter_{epoch}_{train_loss:.4f}', save_top_k=1, monitor="train_loss", every_n_epochs=5)
    accumulator_callback = GradientAccumulationScheduler(scheduling={0:8})#{0: 8, 4: 4, 8: 1}
    avg_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    trainer = pl.Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=nepoch, 
                         strategy="ddp_find_unused_parameters_false", logger=wandb_logger, 
                         max_time="00:23:45:00", callbacks=[checkpoint_callback, accumulator_callback], #accumulator_callback, avg_callback
                         sync_batchnorm=True, check_val_every_n_epoch=1,
                         detect_anomaly=False, gradient_clip_val=0.5)
    #torch.set_num_threads(16)
    #trainer = pl.Trainer(accelerator="cpu", max_epochs=20, logger=wandb_logger)

    #Model
    model = LightningCnn(mid_channel, input_stencil=input_stencil, stencil_x=stencil_x, 
                         stencil_y=stencil_y, learning_rate=learning_rate,
                         skip_connection=False, batch_norm=False, batch_size=batch_size)

    #training
    #trainer.validate(model, mock_dataset)
    trainer.fit(model, train_dataloader, mock_dataset)

    #if trainer.is_global_zero:
        #model = LightningCnn.load_from_checkpoint("ckpt/cnn_epoch=9_train_loss=0.01.ckpt")
    #    trainer.validate(model, dataloaders=validation_dataloader)
    #    trainer.test(model, dataloaders=mock_dataset)
    #trainer.save_checkpoint("cnn.ckpt")
    #print("Model has been saved successfully!")
    
    # model = LightningCnn.load_from_checkpoint("cnn_reconstruct1.ckpt")
    # print("model loaded")
    # trainer.test(model, dataloaders=test_dataloader)

# srun -p amdrtx,amdv100 --gpus 2 --mem=60000 --cpus-per-task=32 -n 1 python cnn.py
# srun -p amdrtx --exclusive --mem=0 --cpus-per-task=32 -n 1 python cnn_iter.py

# TODO: generate data on the fly
# TODO: rotate states to normal vectors of upper & right edges before feeding to neural network
# TODO: get upper & right fluxes through the same cnn
# TODO: use constraint layer to get qdx, qdy => limiter => numerical flux
