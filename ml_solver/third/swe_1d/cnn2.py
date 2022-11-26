import numpy as np
import torch.utils.data as Data
import zarr
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from random import randint
import wandb
from shallow1d import flux_roe, flux_naive, flux_lf

flux = flux_roe
OFFSET = 32
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class SWEDataset(Dataset):
    def __init__(self, file_path, length=None, load=False):
        super().__init__()
        self.ds = zarr.open(file_path, "r")             #xr.open_zarr(file_path)
        self.ds_q = self.ds.q
        self.offset = OFFSET
        self.simulation_steps = 2560
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

class ConvBlock(nn.Module):
    def __init__(self, mid_channel, kernel_size, skip_connection=True, batch_norm=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1)//2
        self.conv1 = nn.Conv1d(mid_channel, mid_channel, self.kernel_size, padding=self.padding, padding_mode="circular")
        self.conv2 = nn.Conv1d(mid_channel, mid_channel, self.kernel_size, padding=self.padding, padding_mode="circular")
        self.batch_norm = batch_norm
        self.skip_connection = skip_connection
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(mid_channel)
            self.bn2 = nn.BatchNorm1d(mid_channel)

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
    def __init__(self, dx: float, stencil_x: int, learning_rate: float, mid_channel: int, skip_connection: bool = True, batch_norm: bool = True):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.mid_channel= mid_channel        
        self.kernel_size = stencil_x
        self.stencil_x = stencil_x
        self.nstencil_out = self.stencil_x
        self.padding = (self.kernel_size - 1)//2
        self.learning_rate = learning_rate
        self.dx = dx

        conv_out = nn.Conv1d(self.mid_channel, 2 * self.nstencil_out, 1)
        nn.init.constant_(conv_out.weight, 0)
        nn.init.constant_(conv_out.bias, 0)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.mid_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode="circular"),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            conv_out,
        )

    def forward_mid(self, inputs):
        (batch_size, _, nx) = inputs.shape                                                       #(nbatch, 2, nx)
        #Normalize inputs data
        inputs_normal = inputs - inputs.mean(dim=2, keepdim=True)
        inputs_normal = inputs_normal / (inputs_normal.abs().amax(dim=2, keepdim=True) + 1e-5)
        x = inputs_normal.float()                                                                #(nbatch, 2, nx)
        x = torch.tanh(self.conv(x))                                                             #(nbatch, 2 * nstencil_out, nx)
        x = x.reshape(batch_size, 2, self.nstencil_out, nx).moveaxis(-1,-2)                      #(nbatch, 2, nx, nstencil_out)
        #Compute alpha
        assert self.nstencil_out == self.stencil_x
        alpha = x - x.mean(dim=-1, keepdim=True) + torch.tensor([0, 1, 0], device=self.device).view((1, 1, 1, self.nstencil_out))
        #padding for interpolate
        p1d = ((self.nstencil_out-1)//2, (self.nstencil_out-1)//2)
        inputs_pad = F.pad(inputs, p1d, "circular")
        inputs_unfold = inputs_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 2, nx, stencil_x)
        #Compute boundary data
        q_mid = (inputs_unfold * alpha).sum(dim=-1)                                              #(nbatch, 2, nx)
        q_mid = F.pad(q_mid, (1,1), "circular")[:, :, :-1]                                       #(nbatch, 2, nx+1)
        q_mid = q_mid.moveaxis(1, 0)                                                             #(2, nbatch, nx+1)
        flux_r = flux_naive(q_mid)                                                               #(2, nbatch, nx+1)
        return flux_r

    def forward_slope(self, inputs):
        (batch_size, _, nx) = inputs.shape                                                       #(nbatch, 2, nx)
        #Normalize inputs data
        inputs_normal = inputs - inputs.mean(dim=2, keepdim=True)
        inputs_normal = inputs_normal / (inputs_normal.abs().amax(dim=2, keepdim=True) + 1e-5)
        x = inputs_normal.float()                                                                #(nbatch, 2, nx)
        x = torch.tanh(self.conv(x))                                                             #(nbatch, 2 * nstencil_out, nx)
        x = x.reshape(batch_size, 2, self.nstencil_out, nx).moveaxis(-1,-2)                      #(nbatch, 2, nx, nstencil_out)
        #Compute alpha
        assert self.nstencil_out == self.stencil_x
        alpha = x - x.mean(dim=-1, keepdim=True)
        #padding for interpolate
        p1d = ((self.nstencil_out-1)//2, (self.nstencil_out-1)//2)
        inputs_pad = F.pad(inputs, p1d, "circular")
        inputs_unfold = inputs_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 2, nx, stencil_x)
        #Compute boundary data
        slope_x = (inputs_unfold * alpha).sum(dim=-1)                                            #(nbatch, 2, nx)
        ql = F.pad(inputs - 0.5*slope_x, (1,1), "circular").transpose(0,1)                       #(2, nbatch, nx)
        qr = F.pad(inputs + 0.5*slope_x, (1,1), "circular").transpose(0,1)                       #(2, nbatch, nx)
        flux_r = flux(ql[:, :, :-1], qr[:, :, 1:])                                               #(2, nbatch, nx+1)
        return flux_r

    def forward(self, inputs):
        # return self.forward_mid(inputs)
        return self.forward_slope(inputs)

    def fvm_step(self, q0):
        N = q0.shape[-1]
        dt = 1e-3*(128/N)

        flux_r_pred = self.forward(q0)

        flux_l = flux_r_pred[:, :, :-1]
        flux_r = flux_r_pred[:, :, 1:]

        flux_sum = -flux_l + flux_r

        qc = q0.moveaxis(0, 1)                                                                  #(2, nbatch, N)
        q1 = qc - dt/self.dx * flux_sum

        q1 = q1.moveaxis(1, 0)                                                                  #(nbatch, 2, N)
        return q1

    # def training_step(self, train_batch, batch_idx):
    #     qs = train_batch
    #     simulation_len = qs.shape[1]
    #     assert simulation_len == OFFSET
    #     qi_pred = qs[:, 0, ...]
    #     loss = torch.tensor(0.0, requires_grad=False)
    #     # scale = 1e3
    #     for i in range(1, simulation_len):
    #         qi_pred = self.fvm_step(qi_pred)
    #         loss_i = F.mse_loss(qi_pred, qs[:, i, ...])
    #         loss = loss + loss_i
    #     self.log('train_loss', loss)
    #     return loss

    def training_step(self, train_batch, batch_idx):
        qs = train_batch
        simulation_len = qs.shape[1]
        assert simulation_len == OFFSET
        qi_pred = qs[:, 0, ...]
        loss = torch.tensor(0.0, requires_grad=False)
        scale = 1e3
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
    parser.add_argument("--scale", default=8, type=int)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stencil_x = args.nx_stencil
    downsample_ratio = args.scale
    batch_size = 128
    learning_rate = 1e-6
    mid_channel= 128
    nepoch = 20

    file_path = "./training_data/1d_torch_2048_32x_randvel_iter_TVD_downsampleT_gaussian_qinit.zarr"
    ds = zarr.open(f"{file_path}", "r")
    print("Total train dataset shape:", ds.q.shape) 
    dx = ds.attrs["dx"]

    #Initialize Dataloader
    #Training
    load = False
    length = None
    num_workers = 1 if load else 6

    train_dataloader = DataLoader(
            dataset=SWEDataset(file_path, length, load),                #SWEDataset_Sim(nsamples, 256),#
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

    if args.gpu:
        wandb_logger = WandbLogger(project="SWE-cnn", name=f"RUN_1DResBlock_iter_lr{learning_rate:.0e}_newdata@{datetime.now()}", save_code=True) # settings=wandb.Settings(start_method='thread')
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
        model = LightningCnn(dx=dx, stencil_x=stencil_x, learning_rate=learning_rate, mid_channel = mid_channel, skip_connection=False, batch_norm=False)
        #training
        trainer.fit(model, train_dataloader)
        
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=1, strategy="ddp_find_unused_parameters_false")
        model = LightningCnn(dx=dx, stencil_x=stencil_x, learning_rate=learning_rate, mid_channel = mid_channel, skip_connection=False, batch_norm=False)
        trainer.fit(model, train_dataloader)
    trainer.save_checkpoint(f"test2-{downsample_ratio}.ckpt")
    print("Model has been saved successfully!")
