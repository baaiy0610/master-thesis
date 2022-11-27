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
from shallow1d import flux_roe, flux_rusanov

OFFSET = 32
flux = flux_roe
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
    def __init__(self, mid_channel, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1)//2
        self.conv1 = nn.Conv1d(mid_channel, mid_channel, self.kernel_size, padding=self.padding, padding_mode="circular")
        self.conv2 = nn.Conv1d(mid_channel, mid_channel, self.kernel_size, padding=self.padding, padding_mode="circular")

    def forward(self, inputs):
            x = self.conv1(inputs)
            x = F.gelu(x)
            x = self.conv2(x)
            return F.gelu(x)

class LightningCnn(pl.LightningModule):
    def __init__(self, dx: float, stencil_x: int, learning_rate: float, mid_channel: int):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.mid_channel= mid_channel        
        self.kernel_size = stencil_x
        self.nstencil_out = 2
        self.padding = (self.kernel_size - 1)//2
        self.learning_rate = learning_rate
        self.dx = dx

        # conv_out = nn.Conv1d(self.mid_channel, 2, 1)
        # nn.init.constant_(conv_out.weight, 0)
        # nn.init.constant_(conv_out.bias, 0)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.mid_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode="circular"),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size),
            ConvBlock(self.mid_channel, self.kernel_size),
            ConvBlock(self.mid_channel, self.kernel_size),
            nn.Conv1d(self.mid_channel, 2, 1),
            nn.Softplus()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.mid_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode="circular"),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size),
            ConvBlock(self.mid_channel, self.kernel_size),
            ConvBlock(self.mid_channel, self.kernel_size),
            nn.Conv1d(self.mid_channel, 2, 1),
            # conv_out,
        )
        
    def forward(self, inputs):
        (batch_size, _, nx) = inputs.shape                                                       #(nbatch, 2, nx)
        #Normalize inputs data
        inputs_normal = (inputs - inputs.mean(dim=2, keepdim=True)) / inputs.std(dim=2, keepdim=True)
        x = inputs_normal.float()                                                                #(nbatch, 2, nx)
        # h = x[:,0,:].unsqueeze(1)
        # hu = x[:,1,:].unsqueeze(1)
        h = self.conv1(x)                                                                        #(nbatch, 1, nx)
        hu = self.conv2(x)                                                                      #(nbatch, 1, nx)
        h = h.reshape(batch_size, 1, 2, nx)
        hu = hu.reshape(batch_size, 1, 2, nx)
        q = torch.cat((h, hu), dim=1)
        #Compute boundary data
        ql = q[:,:,0,:]
        qr = q[:,:,1,:]
        qlr = F.pad(ql, (1,1), "circular")[:, :, 1:].transpose(0,1)                              #(nbatch, 2, nx+1)
        qll = F.pad(qr, (1,1), "circular")[:, :, :-1].transpose(0,1)                             #(nbatch, 2, nx+1)
        flux_l = flux(qll, qlr)                                                                  #(2, nbatch, nx+1)
        return flux_l

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

    def fvm_tvd_rk3(self, q0):
        #stage1
        qs = self.fvm_step(q0)
        #stage2
        qss = (3/4)*q0 + (1/4)*self.fvm_step(qs)
        #stage3
        q1 = (1/3)*q0 + (2/3)*self.fvm_step(qss)
        return q1

    def training_step(self, train_batch, batch_idx):
        qs = train_batch
        simulation_len = qs.shape[1]
        loss = torch.tensor(0.0, requires_grad=False)
        assert simulation_len == OFFSET
        qi_pred = qs[:, 0, ...]
        for i in range(1, simulation_len):
            qi_pred = self.fvm_tvd_rk3(qi_pred)
            loss_i = F.mse_loss(qi_pred, qs[:, i, ...])
            loss = loss + loss_i
        self.log('train_loss', loss)
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
    parser.add_argument("--scale", default=32, type=int)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stencil_x = args.nx_stencil
    downsample_ratio = args.scale
    batch_size = 64
    learning_rate = 1e-3
    mid_channel= 64
    nepoch = 15

    file_path = f"./training_data/1d_torch_2048_{downsample_ratio}x_randvel_iter_TVD_downsampleT_gaussian_qinit.zarr"
    ds = zarr.open(f"{file_path}", "r")
    print("Total train dataset shape:", ds.q.shape) 
    dx = ds.attrs["dx"]

    #Initialize Dataloader
    #Training
    load = False
    length = None
    num_workers = 1 if load else 6

    train_dataloader = DataLoader(
            dataset=SWEDataset(file_path, length, load),                #SWEDataset_Sim(nsamples, 256)#
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
        model = LightningCnn(dx=dx, stencil_x=stencil_x, learning_rate=learning_rate, mid_channel = mid_channel)
        #training
        trainer.fit(model, train_dataloader)
        
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=1, strategy="ddp_find_unused_parameters_false")
        model = LightningCnn(dx=dx, stencil_x=stencil_x, learning_rate=learning_rate, mid_channel = mid_channel)
        trainer.fit(model, train_dataloader)
    trainer.save_checkpoint(f"test-{downsample_ratio}.ckpt")
    print("Model has been saved successfully!")
