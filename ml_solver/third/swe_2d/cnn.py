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
from shallow2d import flux_roe, flux_rusanov

OFFSET = 16
flux = flux_roe
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

class ConvBlock(nn.Module):
    def __init__(self, mid_channel, kernel_size, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv1 = nn.Conv2d(mid_channel, mid_channel, self.kernel_size, padding=self.padding, padding_mode="circular")
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, self.kernel_size, padding=self.padding, padding_mode="circular")

    def forward(self, inputs):
            x = self.conv1(inputs)
            x = F.gelu(x)
            x = self.conv2(x)
            return F.gelu(x)

class LightningCnn(pl.LightningModule):
    def __init__(self, stencil_x: int, stencil_y: int, learning_rate: float, mid_channel: int, vert: torch.tensor, cell_area: torch.tensor):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.mid_channel= mid_channel        
        self.stencil_x = stencil_x
        self.stencil_y = stencil_y
        self.kernel_size = (self.stencil_x, self.stencil_y)
        self.nstencil_out = 4
        self.padding = ((self.stencil_x - 1)//2, (self.stencil_y - 1)//2)
        self.learning_rate = learning_rate
        self.vert = nn.Parameter(vert, requires_grad=False)
        self.cell_area = nn.Parameter(cell_area, requires_grad=False)

        self.convh = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.mid_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode="circular"),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size, self.padding),
            ConvBlock(self.mid_channel, self.kernel_size, self.padding),
            ConvBlock(self.mid_channel, self.kernel_size, self.padding),
            nn.Conv2d(self.mid_channel, self.nstencil_out, 1),
            nn.Softplus()
        )

        self.convv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.mid_channel, kernel_size=self.kernel_size, padding=self.padding, padding_mode="circular"),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size, self.padding),
            ConvBlock(self.mid_channel, self.kernel_size, self.padding),
            ConvBlock(self.mid_channel, self.kernel_size, self.padding),
            nn.Conv2d(self.mid_channel, 2*self.nstencil_out, 1),
        )

    def forward_slope(self, inputs):
        (batch_size, _, nx, ny) = inputs.shape                                                   #(nbatch, 3, nx, ny)
        #Normalize inputs data
        inputs_normal = inputs - inputs.mean(dim=(2, 3), keepdim=True) / inputs.std(dim=(2,3), keepdim=True)
        # inputs_normal = inputs_normal / (inputs_normal.abs().amax(dim=(2, 3), keepdim=True) + 1e-5)
        x = inputs_normal.float()                                                                #(nbatch, 3, nx, ny)
        h = self.convh(x)                                                                        #(nbatch, 4*1, nx, ny)
        huv = self.convv(x)                                                                      #(nbatch, 4*2, nx, ny)
        h = h.reshape(batch_size, 1, self.nstencil_out, nx, ny)                                  #(nbatch, 1, 4, nx, ny)
        huv = huv.reshape(batch_size, 2, self.nstencil_out, nx, ny)                              #(nbatch, 2, 4, nx, ny)
        q = torch.cat((h, huv), dim=1)                                                           #(nbatch, 3, 4, nx, ny)
        #Compute boundary data
        ql = q[:, :, 0, ...]                                                                     #(nbatch, 3, nx, ny)
        qr = q[:, :, 1, ...]
        qu = q[:, :, 2, ...]
        qb = q[:, :, 3, ...]

        vertbl_b = self.vert[:, :, :-1].unsqueeze(1)                                                   #(2, 1, nx+1, ny)
        vertul = self.vert[:, :, 1:].unsqueeze(1)                                                      #(2, 1, nx+1, ny)
        vertbr = self.vert[:, 1:, :].unsqueeze(1)                                                      #(2, 1, nx, ny+1)
        vertbl_l = self.vert[:, :-1, :].unsqueeze(1)                                                   #(2, 1, nx, ny+1)

        qlm = F.pad(ql, (1,1,1,1), "circular")[:, :, 1:, 1:-1].transpose(0,1)                     #(3, nbatch, nx+1, ny)
        qlp = F.pad(qr, (1,1,1,1), "circular")[:, :, :-1, 1:-1].transpose(0,1)                    #(3, nbatch, nx+1, ny)
        qbm = F.pad(qb, (1,1,1,1), "circular")[:, :, 1:-1, 1:].transpose(0,1)                     #(3, nbatch, nx, ny+1)
        qbp = F.pad(qu, (1,1,1,1), "circular")[:, :, 1:-1, :-1].transpose(0,1)                    #(3, nbatch, nx, ny+1)
        
        flux_l_pred = flux(qlm, qlp, vertbl_b, vertul)                                            #(3, nbatch, nx+1, ny)
        flux_b_pred = flux(qbm, qbp, vertbr, vertbl_l)                                            #(3, nbatch, nx, ny+1)
        return flux_l_pred, flux_b_pred

    def forward(self, inputs):
        return self.forward_slope(inputs)

    def fvm_step(self, q0):
        M, N = self.cell_area.shape
        dt = 1e-3*(128/max(M, N))
        flux_l_pred, flux_b_pred = self.forward(q0)

        vertbl = self.vert[:, :-1, :-1]
        vertul = self.vert[:, :-1, 1:]
        vertur = self.vert[:, 1:, 1:]
        vertbr = self.vert[:, 1:, :-1]

        flux_l = flux_l_pred[:, :, :-1, :]
        flux_r = -flux_l_pred[:, :, 1:, :]
        flux_b = flux_b_pred[:, :, :, :-1]
        flux_u = -flux_b_pred[:, :, :, 1:]

        flux_sum = flux_l + flux_u + flux_r + flux_b

        qc = q0.moveaxis(0, 1)                                                                  #(3, nbatch, M, N)
        q1 = qc + -dt/self.cell_area * flux_sum

        q1 = q1.moveaxis(1, 0)                                                                  #(nbatch, 3, M, N)
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
        assert simulation_len == OFFSET
        qi_pred = qs[:, 0, ...]
        loss = torch.tensor(0.0, requires_grad=False)
        for i in range(1, simulation_len):
            qi_pred = self.fvm_tvd_rk3(qi_pred)
            loss_i = F.mse_loss(qi_pred, qs[:, i, ...])
            loss = loss + loss_i 
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, betas=(0.9, 0.99))
        return optimizer
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "loss"}}
        # return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    import time
    import os
    import MeshGrid

    parser = ArgumentParser()
    parser.add_argument("--nx_stencil", default=3, type=int)
    parser.add_argument("--ny_stencil", default=3, type=int)
    parser.add_argument("--scale", default=16, type=int)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stencil_x = args.nx_stencil
    stencil_y = args.ny_stencil
    downsample_ratio = args.scale
    batch_size = 3
    learning_rate = 1e-3
    mid_channel= 64
    nepoch = 20

    file_path = f"./training_data/quad_torch_1024_1024_{downsample_ratio}x_randvel_iter_TVD_downsampleT_perlin2d_qinit.zarr"
    ds = zarr.open(f"{file_path}", "r")
    print("Total train dataset shape:", ds.q.shape)
    vert = torch.as_tensor(ds.vert[...])
    cell_area = torch.as_tensor(ds.cell_area[...])
    print(vert.shape, cell_area.shape)
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
        model = LightningCnn(stencil_x=stencil_x, stencil_y=stencil_y , learning_rate=learning_rate, mid_channel = mid_channel, vert=vert, cell_area=cell_area)
        #training
        trainer.fit(model, train_dataloader)
        
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=1, strategy="ddp_find_unused_parameters_false")
        model = LightningCnn(stencil_x=stencil_x, stencil_y=stencil_y , learning_rate=learning_rate, mid_channel = mid_channel, vert=vert, cell_area=cell_area)
        trainer.fit(model, train_dataloader)
    trainer.save_checkpoint(f"test-{downsample_ratio}.ckpt")
    print("Model has been saved successfully!")


