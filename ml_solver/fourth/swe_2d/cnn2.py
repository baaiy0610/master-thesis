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

from shallow2d import flux_roe, flux_rusanov
flux = flux_roe

OFFSET = 32
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class SWEDataset(Dataset):
    def __init__(self, file_path, length=None, load=False):
        super().__init__()
        self.ds = zarr.open(file_path, "r")             #xr.open_zarr(file_path)
        self.ds_q = self.ds.q
        self.offset = OFFSET
        self.simulation_steps = self.ds.attrs["nT"]
        self.length = self.ds.q.shape[0] if (length is None) else length
        if self.length % self.simulation_steps != 0:
            self.length -= self.length % self.simulation_steps
        if load:
            self.ds_q = self.ds.q[:length, ...]

 
    def __len__(self):
        return self.length - self.offset

    def __getitem__(self, index):
        if self.simulation_steps - index % self.simulation_steps <= self.offset:
            index -= randint(self.offset, self.simulation_steps - 2*self.offset)
        qs = torch.as_tensor(self.ds_q[index:index+self.offset, ...])
        return qs

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
    def __init__(self, mid_channels, kernel_size, p2d, skip_connection=True, batch_norm=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = p2d
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, self.kernel_size, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, self.kernel_size, padding=1, padding_mode="circular")
        self.batch_norm = batch_norm
        self.skip_connection = skip_connection
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(mid_channels)

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.skip_connection:
            return F.gelu(x) + inputs
        else:
            return F.gelu(x)

class LightningCnn(pl.LightningModule):
    def __init__(self, mid_channel: int, stencil_x: int, stencil_y: int, learning_rate: float, vert: torch.tensor, cell_area: torch.tensor, skip_connection: bool = True, batch_norm: bool = True):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.mid_channel = mid_channel
        self.nstencil_out = stencil_x * stencil_y
        self.ninterp = 2
        self.kernel_size = 3
        self.stencil_x = stencil_x
        self.stencil_y = stencil_y
        self.automatic_optimization = True
        self.vert = nn.Parameter(vert, requires_grad=False)
        self.cell_area = nn.Parameter(cell_area, requires_grad=False)
        
        conv_out = nn.Conv2d(self.mid_channel, 3 * self.nstencil_out, 1)
        nn.init.constant_(conv_out.weight, 0)
        nn.init.constant_(conv_out.bias, 0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.mid_channel, kernel_size=self.kernel_size, padding=1, padding_mode="circular"),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            conv_out,
        )

    def forward_slope(self, inputs):
        (batch_size, _, nx, ny) = inputs.shape                                                   #(nbatch, 3, nx, ny)
        #Normalize inputs data
        inputs_normal = inputs - inputs.mean(dim=(2, 3), keepdim=True)
        inputs_normal = inputs_normal / (inputs_normal.abs().amax(dim=(2, 3), keepdim=True) + 1e-5)
        x = inputs_normal.float()                                                                #(nbatch, 3, nx, ny)
        convout_x = torch.tanh(self.conv(x))                                                     #(nbatch, 3 * nstencil_out, nx, ny)
        convout_y = torch.tanh(torch.transpose(self.conv(torch.transpose(x, -2, -1)), -2, -1))
        x = torch.stack((convout_y, convout_x), dim=1)                                           #(nbatch, ninterp, 3*nstencil_out, nx, ny)
        x = x.reshape(batch_size, self.ninterp, 3, self.nstencil_out, nx, ny).moveaxis(-3,-1)    #(nbatch, ninterp, 3, nx, ny, nstencil_out)
        #Compute alpha
        # alpha = x / (x.sum(dim=-1, keepdim=True) + 1e-5)                                        #(nbatch, ninterp, 3, nx, ny, nstencil_out)
        # alpha = x - x.mean(dim=-1, keepdim=True) + torch.tensor([[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]],device=self.device).view((1, 2, 1, 1, 1, self.nstencil_out))
        alpha = x - x.mean(dim=-1, keepdim=True)
        #padding for interpolate
        p2d = ((self.stencil_x-1)//2, (self.stencil_x-1)//2, (self.stencil_y-1)//2, (self.stencil_y-1)//2)
        inputs_pad = F.pad(inputs, p2d, "circular")
        inputs_unfold = inputs_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 3, nx, ny, stencil_x)
        inputs_unfold = inputs_unfold.unfold(3, self.stencil_y, 1)                               #(nbatch, 3, nx, ny, stencil_x, stencil_y)
        inputs_unfold = inputs_unfold.reshape(batch_size, 3, nx, ny, self.nstencil_out)          #(nbatch, 3, nx, ny, nstencil_out)
        inputs_unfold = inputs_unfold.unsqueeze(1)                                               #(nbatch, 1, 3, nx, ny, nstencil_out)
        #Compute boundary data
        qgrad = (inputs_unfold * alpha).sum(dim=-1)                                              #(nbatch, ninterp, 3, nx, ny)
        slope_y = qgrad[:, 0, :, :, :]
        slope_x = qgrad[:, 1, :, :, :]
        vert_ul = self.vert[:, :-1, :].unsqueeze(1)                                                #(2, 1, nx, ny+1)
        vert_ur_u = self.vert[:, 1:, :].unsqueeze(1)                                               #(2, 1, nx, ny+1)
        vert_ur_r = self.vert[:, :, 1:].unsqueeze(1)                                               #(2, 1, nx+1, ny)
        vert_br = self.vert[:, :, :-1].unsqueeze(1)                                                #(2, 1, nx+1, ny)
        qr = F.pad((inputs + 0.5*slope_x), (1, 1, 1, 1), "circular").transpose(0, 1)
        ql = F.pad((inputs - 0.5*slope_x), (1, 1, 1, 1), "circular").transpose(0, 1)
        qu = F.pad((inputs + 0.5*slope_y), (1, 1, 1, 1), "circular").transpose(0, 1)
        qb = F.pad((inputs - 0.5*slope_y), (1, 1, 1 ,1), "circular").transpose(0, 1)

        flux_u = flux(qu[:, :, 1:-1, :-1], qb[:, :, 1:-1, 1:], vert_ul, vert_ur_u)      #(3, nbatch, nx, ny+1)
        flux_r = flux(qr[:, :, :-1, 1:-1], ql[:, :, 1:, 1:-1], vert_ur_r, vert_br)      #(3, nbatch, nx+1, ny)
        return flux_u, flux_r

    def forward(self, inputs):
        return self.forward_slope(inputs)

    def fvm_step(self, q0):
        M, N = self.cell_area.shape
        dt = 0.001*(128/max(N, M/2))

        flux_u_pred, flux_r_pred = self.forward(q0)

        vertbl = self.vert[:, :-1, :-1]
        vertul = self.vert[:, :-1, 1:]
        vertur = self.vert[:, 1:, 1:]
        vertbr = self.vert[:, 1:, :-1]

        flux_l = - flux_r_pred[:, :, :-1, :]
        flux_r = flux_r_pred[:, :, 1:, :]
        flux_b = - flux_u_pred[:, :, :, :-1]
        flux_u = flux_u_pred[:, :, :, 1:]

        flux_sum = flux_l + flux_u + flux_r + flux_b

        qc = q0.moveaxis(0, 1) # (3, nbatch, M, N)
        q1 = qc + -dt/self.cell_area * flux_sum

        q1 = q1.moveaxis(1, 0) # (nbatch, 3, M, N)
        return q1

    def fvm_tvd_rk3(self, q0):
        #stage1
        qs = self.fvm_step(q0)
        #stage2
        qss = (3/4)*q0 + (1/4)*self.fvm_step(qs)
        #stage3
        q1 = (1/3)*q0 + (2/3)*self.fvm_step(qss)
        return q1

    def total_energy(self, q):
        assert q.shape[1] == 3
        from shallow2d import g
        potential = 0.5 * torch.square(q[:, 0, :, :]) * g
        kinetic = 0.5 * (torch.square(q[:, 1, :, :]) + torch.square(q[:, 2, :, :]) + 1e-5)
        total_energy = ((potential + kinetic) * self.cell_area).sum(dim=(-2, -1)) # (nbatch,)
        return total_energy

    def training_step(self, train_batch, batch_idx):
        qs = train_batch
        simulation_len = qs.shape[1]
        assert simulation_len == OFFSET
        qi_pred = qs[:, 0, ...]
        loss = torch.tensor(0.0, requires_grad=False)
        scale = 1e4
        current_len = 1
        threshold = 1.0
        for i in range(1, simulation_len):
            qi_pred = self.fvm_step(qi_pred)
            loss_i = F.mse_loss(qi_pred, qs[:, i, ...]) * scale
            if torch.isnan(loss_i).any() or loss_i.item() > threshold:
                loss = loss + (simulation_len-i)*threshold
                break
            loss = loss + loss_i
            current_len = i
        loss = loss / simulation_len
        self.log('train_loss', loss)
        self.log('current_len', current_len)
        return loss

    # def training_step(self, train_batch, batch_idx):
    #     qs = train_batch
    #     simulation_len = qs.shape[1]
    #     assert simulation_len == OFFSET
    #     qi_pred = qs[:, 0, ...]
    #     loss = torch.tensor(0.0, requires_grad=False)
    #     scale = 1e5
    #     current_len = 1
    #     threshold = 15
    #     for i in range(1, simulation_len):
    #         eT = self.total_energy(qi_pred)
    #         qi_pred = self.fvm_step(qi_pred)
    #         loss_i = F.mse_loss(qi_pred, qs[:, i, ...])
    #         loss_i = F.mse_loss(qi_pred, qs[:, i, ...]) * scale
    #         if torch.isnan(loss_i).any() or loss_i.item() > threshold:
    #             loss = loss + (simulation_len-i)*threshold
    #             break
    #         loss = loss + loss_i
    #         current_len = i
    #     loss = loss / simulation_len
    #     e0 = self.total_energy(qs[:, 0, ...])
    #     scale_e = 1e-2
    #     loss_energy = F.mse_loss(e0*scale_e, eT*scale_e)

    #     loss = loss + loss_energy

    #     self.log('train_loss', loss)
    #     self.log('current_len', current_len)
    #     self.log('train_loss_energy', loss_energy)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, betas=(0.9, 0.99))
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
    parser.add_argument("--scale", default=8, type=int)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stencil_x = args.nx_stencil
    stencil_y = args.ny_stencil
    downsample_ratio = args.scale
    mid_channel = 64
    batch_size = 32
    learning_rate = 1e-6
    nepoch = 10

    file_path = f"./training_data/quad_torch_1024_1024_{downsample_ratio}x_randvel_iter_TVD_downsampleT_perlin2d_qinit.zarr"
    ds = zarr.open(f"{file_path}", "r")
    print("Total train dataset shape:", ds.q.shape)
    vert = torch.as_tensor(ds.vert[...])
    cell_area = torch.as_tensor(ds.cell_area[...])
    print(vert.shape, cell_area.shape)    #Initialize Dataloader
    #Training
    load = False
    length = None
    num_workers = 1 if load else 6

    train_dataloader = DataLoader(
            dataset=SWEDataset(file_path, length, load), #SWEDataset_Sim(1000, 250, 125),#
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

    if args.gpu:
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

        #Model
        model = LightningCnn(mid_channel=mid_channel, stencil_x=stencil_x, 
                            stencil_y=stencil_y, learning_rate=learning_rate, vert=vert, cell_area=cell_area,
                            skip_connection=False, batch_norm=False)

        #training
        trainer.fit(model, train_dataloader)
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=1, strategy="ddp_find_unused_parameters_false")
        model = LightningCnn(mid_channel=mid_channel, stencil_x=stencil_x, 
                            stencil_y=stencil_y, learning_rate=learning_rate, vert=vert, cell_area=cell_area,
                            skip_connection=False, batch_norm=False)
        trainer.fit(model, train_dataloader)

    trainer.save_checkpoint(f"test2-{downsample_ratio}.ckpt")
    print("Model has been saved successfully!")
