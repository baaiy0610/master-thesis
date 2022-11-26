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

def boundary_padding(q, stencil_x, stencil_y, unsqueeze=False):
    """
    q: double[4, M, N]
    return: double[4, M+4, N+4]
    """
    # periodic boundary condition for dim 1
    q = F.pad(q, (0, 0, stencil_x, stencil_x), "circular")
    # special boundary condition for dim 2
    q = torch.cat((torch.flip(q[..., 0:stencil_y], [-2, -1]), q, torch.flip(q[..., -stencil_y:], [-2, -1])), dim=-1)
    return q

class TrainDataset(Dataset):
    def __init__(self, path):
        self.data_train = torch.as_tensor(zarr.open(f'{path}/Data_Train.zarr', mode="r")[0:5000]).double()
        self.data_true  = torch.as_tensor(zarr.open(f'{path}/Data_True.zarr', mode="r")[0:5000]).double()

    def __getitem__(self, index):
        #Train value
        y_train = self.data_train[index]
        #True value
        y_true = self.data_true[index]
        return y_train, y_true

    def __len__(self):
        return self.data_train.shape[0]

class TestDataset(Dataset):
    def __init__(self, path):
        self.data_test, self.data_test_true = torch.load(path)
        
    def __getitem__(self, index):
        #True value
        y_test_true = self.data_test_true[index]
        y_test_true = torch.as_tensor(y_test_true).double()
        #Train value
        y_test = self.data_test[index]
        y_test = torch.as_tensor(y_test)
        return y_test, y_test_true

    def __len__(self):
        return self.data_test.shape[0]

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

class LightningCnn(pl.LightningModule):
    def __init__(self, mu: torch.Tensor, std: torch.Tensor, mid_channel: int, nstencil_out: int, ninterp: int, stencil_x: int, stencil_y: int):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.mu = nn.Parameter(mu, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.mid_channel = mid_channel
        self.nstencil_out = nstencil_out
        self.ninterp = ninterp
        self.kernel_size = 3
        self.stencil_x = stencil_x
        self.stencil_y = stencil_y
        
        #Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=self.mid_channel, kernel_size=self.kernel_size,
                    padding=0
            ),                              
            nn.ReLU(),                     
        )

        # Layer 2 
        self.conv2 = nn.Sequential(         
            nn.Conv2d(mid_channel, self.mid_channel, self.kernel_size, 
                    padding=0),     
            nn.ReLU(),                     
        )

        # Layer 3
        # self.conv2 = nn.Sequential(         
        #     nn.Conv2d(mid_channel, self.mid_channel, self.kernel_size, 
        #             padding=0),     
        #     nn.ReLU(),                     
        # )

        # Layer 4
        # self.conv2 = nn.Sequential(         
        #     nn.Conv2d(mid_channel, self.mid_channel, self.kernel_size, 
        #             padding=0),     
        #     nn.ReLU(),                     
        # )

        # Layer 5
        # self.conv2 = nn.Sequential(         
        #     nn.Conv2d(mid_channel, self.mid_channel, self.kernel_size, 
        #             padding=0),     
        #     nn.ReLU(),                     
        # )

        #Layer 6
        self.out = nn.Conv2d(self.mid_channel, 4*self.ninterp * (self.nstencil_out - 1), self.kernel_size,
                    padding=0)                                                 #(nbatch, 4*ninterp * (nstencil_out - norder), nx, ny)

        #Layer 7
        self.constrain = LightningConstraintLayer2D(nstencil_out, 4*ninterp)

    def forward(self, input):
        (batch_size, _, nx, ny) = input.shape                                                  #(nbatch, 4, nx, ny)
        #Convert (h, hu, hv, hw) to (h, u, v, w)
        input_ = torch.zeros_like(input)
        input_[:, 0, :, :] = input[:, 0, :, :]
        input_[:, 1, :, :] = input[:, 1, :, :]/input[:, 0, :, :]
        input_[:, 2, :, :] = input[:, 2, :, :]/input[:, 0, :, :]
        input_[:, 3, :, :] = input[:, 3, :, :]/input[:, 0, :, :]
        #Normalize input data
        input_normal = (input_ - self.mu) / self.std
        x = input_normal.float()                                                               #(nbatch, 4, nx, ny)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv1(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.out(x)                                                                        #(nbatch, 4*ninterp * (nstencil_out - norder), nx, ny)
        x = x.moveaxis(1,-1)
        x = torch.reshape(x, (batch_size, nx, ny, 4*self.ninterp, self.nstencil_out - 1))      #(nbatch, nx, xy, 4*ninterp, nstencil_out - norder)
        #Compute alpha
        alpha = self.constrain(x)                                                              #(nbatch, nx, ny, 4*ninterp, nstencil_out)
        #padding for interpolate
        input_pad = boundary_padding(input_, (self.stencil_x-1)//2, (self.stencil_y-1)//2)
        input_unfold = input_pad.unfold(2, self.stencil_x, 1)                                  #(nbatch, 4, nx, ny, stencil_x)
        input_unfold = input_unfold.unfold(3, self.stencil_y, 1)                               #(nbatch, 4, nx, ny, stencil_x, stencil_y)
        input_unfold = input_unfold.reshape(batch_size, 4, nx, ny, self.nstencil_out)          #(nbatch, 4, nx, ny, nstencil_out)
        input_unfold = input_unfold.unsqueeze(4)                                               #(nbatch, 4, nx, ny, 1, nstencil_out)
        #Compute boundary data
        alpha = torch.reshape(alpha, (batch_size, nx, ny, 4, self.ninterp, self.nstencil_out))
        alpha = alpha.moveaxis(3,1)                                                            #(nbatch, 4, nx, ny, ninterp, nstencil_out)
        output = (input_unfold * alpha).sum(dim=-1)                                            #(nbatch, 4, nx, ny, ninterp)
        return output

    def cnn_mse_loss(self, predict, target):
        return F.mse_loss(predict, target)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self.forward(x)
        loss = self.cnn_mse_loss(output, y)
        self.log('train loss', loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        x, y = test_batch
        output = self.forward(x)
        loss = self.cnn_mse_loss(output, y)
        self.log('Val loss', loss, sync_dist=True)
        #return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        output = self.forward(x)
        loss = self.cnn_mse_loss(output, y)
        self.log('TEST loss', loss, sync_dist=True)
        #return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer

def main(args):
    model = LightningModule()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from datetime import datetime
    import time

    parser = ArgumentParser()
    parser.add_argument("--nx_stencil", default=3, type=int)
    parser.add_argument("--ny_stencil", default=3, type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    stencil_x = args.nx_stencil
    stencil_y = args.ny_stencil
    nstencil_out = stencil_x * stencil_y
    mid_channel = 128
    ninterp = 2
    batch_size = 32

    dataset_train = "data_flux_u_r"

    data_train = zarr.open(f'{dataset_train}/Data_Train.zarr', mode="r")
    print("Data train shape:", data_train.shape)

    mu, std = torch.load(f"./mu_std_{dataset_train}.pt")
    mu[:, 0, :, :] = 0.0

    #data
    train_dataset = TrainDataset(f"./{dataset_train}")
    test_dataset = TestDataset("./data_test_flux.pt")

    #Initialize Dataloader
    #Training
    train_dataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    #Testing
    test_dataloader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    wandb_logger = WandbLogger(project="SWE-cnn", name=f"RUN@{datetime.now()}")
    

    if args.gpu:
        trainer = pl.Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=200, strategy="ddp_find_unused_parameters_false", logger=wandb_logger, check_val_every_n_epoch=1)
        if args.retrain:
            trainer = pl.Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=100, strategy="ddp_find_unused_parameters_false", resume_from_checkpoint="test_flux.ckpt", logger=wandb_logger, check_val_every_n_epoch=1)
            print("Model loaded, keep training.........")
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=100, strategy="ddp_find_unused_parameters_false", logger=wandb_logger, check_val_every_n_epoch=1)

    #Model
    model = LightningCnn(mu, std, mid_channel, nstencil_out=nstencil_out, ninterp=ninterp, stencil_x=stencil_x, stencil_y=stencil_y)

    #training
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(dataloaders=test_dataloader)
    trainer.save_checkpoint("test_flux2.ckpt")
    print("Model has been saved successfully!")
    
    # model = LightningCnn.load_from_checkpoint("cnn_reconstruct1.ckpt")
    # print("model loaded")
    # trainer.test(model, dataloaders=test_dataloader)