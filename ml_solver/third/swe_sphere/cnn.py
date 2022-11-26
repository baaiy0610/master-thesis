import numpy as np
import torch.utils.data as Data
import zarr
from torch.utils.data import Dataset, TensorDataset
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb

from SWE_sphere import flux_naiive


class SWEDataset(Dataset):
    def __init__(self, file_path, length=None, load=False, scale_flux=1):
        super().__init__()
        self.ds = zarr.open(file_path, "r")#xr.open_zarr(file_path)
        self.vert_3d = torch.as_tensor(self.ds.vert_3d[...])
        self.length = self.ds.q.shape[0] if (length is None) else length
        self.ds_q = self.ds.q
        self.ds_flux_u = self.ds.flux_u
        self.ds_flux_r = self.ds.flux_r
        self.scale_flux = scale_flux
        if load:
            #perm = np.random.permutation(self.ds.q.shape[0])[:length]
            self.ds_q = self.ds.q[:length, ...]
            self.ds_flux_u = self.ds.flux_u[:length, ...]
            self.ds_flux_r = self.ds.flux_r[:length, ...]
 
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.as_tensor(self.ds_q[index, ...])
        flux_u = torch.as_tensor(self.ds_flux_u[index, ...])*self.scale_flux
        flux_r = torch.as_tensor(self.ds_flux_r[index, ...])*self.scale_flux
        return inputs, self.vert_3d, flux_u, flux_r

def boundary_padding(q, halo_x, halo_y, unsqueeze=False):
    """
    q: double[4, M, N]
    return: double[4, M+4, N+4]
    """
    # periodic boundary condition for dim 1
    q = F.pad(q, (0, 0, halo_x, halo_x), "circular")
    # special boundary condition for dim 2
    q = torch.cat((torch.flip(q[..., 0:halo_y], [-2, -1]), q, torch.flip(q[..., -halo_y:], [-2, -1])), dim=-1)
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

class LightningCnn(pl.LightningModule):
    def __init__(self, mid_channel: int, input_stencil: int, stencil_x: int, stencil_y: int, learning_rate: float, skip_connection: bool = True, batch_norm: bool = True):
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
        
        self.conv = nn.Sequential(
            Padding((self.input_stencil-1)//2, (self.input_stencil-1)//2),
            nn.Conv2d(in_channels=4, out_channels=self.mid_channel, kernel_size=self.input_stencil),                              
            nn.GELU(),
            ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            #ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            #ConvBlock(self.mid_channel, self.kernel_size, skip_connection, batch_norm),
            nn.Conv2d(self.mid_channel, 4*self.ninterp * (self.nstencil_out - 1), 1)
        )

        self.constrain = LightningConstraintLayer2D(self.nstencil_out, 4*self.ninterp)


    def forward(self, inputs, vert_3d):
        # vert_3d: (batch_size, 3, nx+1, ny+1)
        if len(vert_3d.shape) == 4:
            vert_3d = vert_3d[0, ...]
        (batch_size, _, nx, ny) = inputs.shape                                                  #(nbatch, 4, nx, ny)
        #Normalize inputs data
        inputs_normal = inputs / inputs.abs().amax(dim=(1,2,3), keepdim=True)
        x = inputs_normal.float()                                                                #(nbatch, 4, nx, ny)
        x = self.conv(x)                                                                         #(nbatch, 4*ninterp * (nstencil_out - norder), nx, ny)
        x = x.moveaxis(1,-1)
        x = torch.reshape(x, (batch_size, nx, ny, 4*self.ninterp, self.nstencil_out - 1))        #(nbatch, nx, xy, 4*ninterp, nstencil_out - norder)
        #Compute alpha
        alpha = self.constrain(x)                                                                #(nbatch, nx, ny, 4*ninterp, nstencil_out)
        #padding for interpolate
        inputs_pad = boundary_padding(inputs, (self.stencil_x-1)//2, (self.stencil_y-1)//2)
        inputs_unfold = inputs_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 4, nx, ny, stencil_x)
        inputs_unfold = inputs_unfold.unfold(3, self.stencil_y, 1)                               #(nbatch, 4, nx, ny, stencil_x, stencil_y)
        inputs_unfold = inputs_unfold.reshape(batch_size, 4, nx, ny, self.nstencil_out)          #(nbatch, 4, nx, ny, nstencil_out)
        inputs_unfold = inputs_unfold.unsqueeze(1)                                               #(nbatch, 1, 4, nx, ny, nstencil_out)
        #Compute boundary data
        alpha = torch.reshape(alpha, (batch_size, nx, ny, 4, self.ninterp, self.nstencil_out))
        alpha = torch.permute(alpha, (0, 4, 3, 1, 2, 5))                                         #(nbatch, ninterp, 4, nx, ny, nstencil_out)
        q_mid = (inputs_unfold * alpha).sum(dim=-1)                                              #(nbatch, ninterp, 4, nx, ny)
        q_mid = q_mid.moveaxis(2, 0)                                                             #(4, nbatch, ninterp, nx, ny)
        q_mid_u = q_mid[:, :, 0, :, :]                                                           #(4, nbatch, nx, ny)
        q_mid_r = q_mid[:, :, 1, :, :]                                                           #(4, nbatch, nx, ny)
        vert_ul = vert_3d[:, :-1, 1:].unsqueeze(1)                                               #(3, 1, nx, ny)
        vert_ur = vert_3d[:, 1:, 1:].unsqueeze(1)                                                #(3, 1, nx, ny)
        vert_br = vert_3d[:, 1:, :-1].unsqueeze(1)                                               #(3, 1, nx, ny)
        flux_u = flux_naiive(q_mid_u, vert_ul, vert_ur, 1.0).moveaxis(0, 1)                      #(nbatch, 4, nx, ny)
        flux_r = flux_naiive(q_mid_r, vert_ur, vert_br, 1.0).moveaxis(0, 1)                      #(nbatch, 4, nx, ny)
        return flux_u, flux_r

    def training_step(self, train_batch, batch_idx):
        inputs, vert_3d, flux_u, flux_r = train_batch
        flux_u_pred, flux_r_pred = self.forward(inputs, vert_3d)
        scale = 1e3
        loss = (F.mse_loss(flux_u_pred*scale, flux_u*scale) + F.mse_loss(flux_r_pred*scale, flux_r*scale))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        inputs, vert_3d, flux_u, flux_r = test_batch
        flux_u_pred, flux_r_pred = self.forward(inputs, vert_3d)
        scale = 1e3
        loss = (F.mse_loss(flux_u_pred*scale, flux_u*scale) + F.mse_loss(flux_r_pred*scale, flux_r*scale))
        self.log('val_loss', loss, sync_dist=True)


    def test_step(self, test_batch, batch_idx):
        import MeshGrid
        import SWE_sphere
        from SWE_sphere import initial_condition, integrate, fvm_2ndorder_space_cnn, fvm_TVD_RK
        SWE_sphere.fvm_2ndorder_space = fvm_2ndorder_space_cnn
        SWE_sphere.fvm = fvm_TVD_RK
        if self.trainer.is_global_zero:
            device = self.device
            M, N = test_batch[0].shape
            print(f"M={M}, N={N}")
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
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
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
    input_stencil = 5
    mid_channel = 128
    batch_size = 64
    learning_rate = 3e-4
    nepoch = 20

    file_path = "/scratch/lhuang/SWE/training_data/sphere_torch_2000_8x_randvel.zarr"
    val_file_path = "/scratch/lhuang/SWE/training_data/sphere_torch_2000_8x.zarr"
    #Initialize Dataloader
    #Training
    load = True
    length = 50000
    if load:
        train_dataloader = Data.DataLoader(
            dataset=SWEDataset(file_path, length=length, load=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
    else:
        train_dataloader = Data.DataLoader(
            dataset=SWEDataset(file_path, length=length, load=False),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True
        )
    validation_dataloader = Data.DataLoader(
        dataset=SWEDataset(val_file_path, length=10000, load=True, scale_flux=8),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    mock_dataset = TensorDataset(torch.zeros((1, 250, 125)))


    wandb_logger = WandbLogger(project="SWE-cnn", name=f"RUN_ResBlock_lr3e-4_newdata@{datetime.now()}", save_code=True) # settings=wandb.Settings(start_method='thread')
    current_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"ckpt/",filename='cnn_{epoch}_{train_loss:.4f}', save_top_k=1, monitor="train_loss", every_n_epochs=5)
    trainer = pl.Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=nepoch, 
                         strategy="ddp_find_unused_parameters_false", logger=wandb_logger, 
                         max_time="00:3:45:00", callbacks=[checkpoint_callback],
                         sync_batchnorm=True, check_val_every_n_epoch=1)
    #torch.set_num_threads(16)
    #trainer = pl.Trainer(accelerator="cpu", max_epochs=20, logger=wandb_logger)

    #Model
    model = LightningCnn(mid_channel, input_stencil=input_stencil, stencil_x=stencil_x, 
                         stencil_y=stencil_y, learning_rate=learning_rate,
                         skip_connection=False, batch_norm=False)

    #training
    #trainer.validate(model, validation_dataloader)
    trainer.fit(model, train_dataloader, validation_dataloader)

    if trainer.is_global_zero:
        #model = LightningCnn.load_from_checkpoint("ckpt/cnn_epoch=9_train_loss=0.01.ckpt")
        trainer.validate(model, dataloaders=validation_dataloader)
        trainer.test(model, dataloaders=mock_dataset)
    #trainer.save_checkpoint("cnn.ckpt")
    #print("Model has been saved successfully!")
    
    # model = LightningCnn.load_from_checkpoint("cnn_reconstruct1.ckpt")
    # print("model loaded")
    # trainer.test(model, dataloaders=test_dataloader)

# srun -p amdrtx,amdv100 --gpus 2 --mem=60000 --cpus-per-task=32 python cnn.py
# srun -p amdrtx --exclusive --mem=0 --cpus-per-task=32 python cnn.py