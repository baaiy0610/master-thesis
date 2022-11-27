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
# torch.set_default_dtype(torch.float64)
#---------------------------------------------------------------------------------------------------------------------------------------------

g = 11489.57219

def transfer_circle(x0, y0, r):
    """
    Generate structure sphere mesh grid
    """
    r1 = r
    d = np.maximum(np.abs(x0), np.abs(y0))
    d = np.maximum(d ,10**-10)
    
    D = r1 * d * (2-d)/np.sqrt(2)
    R = r1
    center = D - np.sqrt(R**2 - D**2)
    xp = D/d * np.abs(x0)
    yp = D/d * np.abs(y0)

    i,j = np.where(np.abs(y0) >= np.abs(x0))
    yp[i,j] = center[i,j] + np.sqrt(R**2 - xp[i,j]**2)
    
    i,j = np.where(np.abs(x0) >= np.abs(y0))
    xp[i,j] = center[i,j] + np.sqrt(R**2 - yp[i,j]**2)

    xp = np.sign(x0) * xp
    yp = np.sign(y0) * yp
    return xp, yp 

def transfer_sphere_surface(vert_3d, vert, r):
    assert vert_3d.shape[0:2] == vert.shape[0:2]
    
    x0 = vert[:,:,0]
    y0 = vert[:,:,1]
    i, j = np.where(x0 < -1)
    x0[i,j] = -2 -x0[i,j]
    xp, yp = transfer_circle(x0, y0, r)
    zp = np.sqrt(r**2 - (xp**2 + yp**2))
    zp[i,j] = -zp[i,j]
    vert_3d[:] = np.stack((xp, yp, zp), axis=-1)

def xyz_to_latlon(vert_3d, r):
    """
    Convert Sphere point from 3d coordinate to lnglat
    vert_sphere: double[M, N, 2], matrix of sphere vertices
    vert_3d: double[M, N, 3], matrix of 3d vertices
    """
    M, N = vert_3d.shape[0], vert_3d.shape[1]
    vert_sphere = np.zeros((M, N, 2))
    vert_sphere[:,:,0] = np.rad2deg(np.arcsin(vert_3d[:,:,2]/r))
    vert_sphere[:,:,1] = np.rad2deg(np.arctan2(vert_3d[:,:,1], vert_3d[:,:,0]))
    return vert_sphere

def sphere_grid(x_range, y_range, M, N, r):
    """
    generate a sphere mesh grid with shape (M, N)
    """
    #initial regular mesh grid
    x = np.linspace(*x_range, M+1)
    y = np.linspace(*y_range, N+1)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    #initial vert_3d
    vert_3d = np.zeros((M+1, N+1, 3))
    #generate sphere grid and yield 3d
    transfer_sphere_surface(vert_3d, vert, r)
    return vert_3d

def sphere_cell_center(x_range, y_range, M, N, r):
    cell_center_3d = np.zeros((M, N, 3))
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = np.linspace(xmin+dx/2, xmax-dx/2, M)
    y = np.linspace(ymin+dy/2, ymax-dy/2, N)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    transfer_sphere_surface(cell_center_3d, vert, r)
    return cell_center_3d

#---------------------------------------------------------------------------------------------------------------------------------------------

def boundary_condition(q, nhalo, unsqueeze=True):
    """
    q: double[4, M, N]
    return: double[4, M+4, N+4]
    """
    if unsqueeze: 
        q = q.unsqueeze(0)
    # periodic boundary condition for dim 1
    q = F.pad(q, (0, 0, nhalo, nhalo), "circular").squeeze()
    # special boundary condition for dim 2
    q = torch.cat((torch.flip(q[..., 0:nhalo], [-2, -1]), q, torch.flip(q[..., -nhalo:], [-2, -1])), dim=-1)
    return q

def d_sphere_3d(point1, point2, r):
    """
    Compute the great circle length crossing point1 and point2
    point1: double[3, ...]
    point2: double[3, ...]
    return: double[...]
    """
    delta = point1 - point2
    distance = torch.sqrt(torch.square(delta).sum(dim=0, keepdim=True))/r
    delta_sigma = 2*torch.arcsin(0.5*distance)
    return r * delta_sigma

def cross(a, b):
    """
    Compute cross product axb
    a: double[3, ...]
    b: double[3, ...]
    return: double[3, ...]
    """
    out = torch.stack((a[1,...]*b[2,...] - a[2,...]*b[1,...],
                       a[2,...]*b[0,...] - a[0,...]*b[2,...],
                       a[0,...]*b[1,...] - a[1,...]*b[0,...]), dim=0)
    return out

def rotate_3d(point1, point2):
    """
    Compute the unit normal vector crossing edge defined by point1 and point2
    point1: double[3, ...]
    point2: double[3, ...]
    return: double[3, ...]
    """
    mid_point = 0.5*(point1 + point2) 
    mid_point = mid_point / torch.sqrt(torch.square(mid_point).sum(dim=0, keepdim=True))
    edge_vector = point2 - point1
    edge_vector = edge_vector / torch.sqrt(torch.square(edge_vector).sum(dim=0, keepdim=True))
    tmp = cross(mid_point, edge_vector)
    out = tmp / torch.sqrt(torch.square(tmp).sum(dim=0, keepdim=True))
    return out

def velocity_tan(q, vector_nor, vector_tan):
    """
    Convert the velocity in q into 2d tangent plane defined by normal and tangent vectors
    q: double[4, ...]
    vector_nor: double[3, ...]
    vector_tan: double[3, ...]
    return: double[3, ...]
    """
    out = torch.stack((q[0, ...],
                       (q[1:,...]*vector_nor).sum(dim=0),
                       (q[1:,...]*vector_tan).sum(dim=0)), dim=0)
    return out

def velocity_xyz(q, vector_nor, vector_tan):
    """
    Convert the velocity in q into 3D Cartesian plane
    q: double[3, ...]
    vector_nor: double[3, ...]
    vector_tan: double[3, ...]
    return: double[4, ...]
    """
    out = torch.cat((q[0, ...].unsqueeze(0),
                     q[1, ...].unsqueeze(0) * vector_nor + q[2, ...].unsqueeze(0) * vector_tan), dim=0)
    return out


def roe_average(ql, qr):
    """
    Compute Roe average of ql and qr
    ql: double[3, ...]
    qr: double[3, ...]
    return: hm, um, vm: double[...]
    """
    hl = ql[0, ...]
    uhl = ql[1, ...]
    vhl = ql[2, ...]

    hr = qr[0, ...]
    uhr = qr[1, ...]
    vhr = qr[2, ...]

    ul = uhl/hl
    vl = vhl/hl

    ur = uhr/hr
    vr = vhr/hr

    sqrthl = torch.sqrt(hl)
    sqrthr = torch.sqrt(hr)

    hm = 0.5*(hl+hr)
    um = (sqrthl*ul + sqrthr*ur)/(sqrthl + sqrthr)
    vm = (sqrthl*vl + sqrthr*vr)/(sqrthl + sqrthr)

    return hm, um, vm

def f(q):
    """
    Flux function for shallow water equation on the direction of x-axis
    q: double[3, ...]
    return: double[3, ...]
    """
    h = q[0, ...]
    uh = q[1, ...]
    vh = q[2, ...]
    u = uh/h
    v = vh/h
    out = torch.stack((uh,
                       u*uh + 0.5*g*h*h,
                       u*vh), dim=0)
    return out


def flux_roe(ql, qr, point1, point2, r):
    """
    Roe's numerical flux for shallow water equation
    ql: double[4,...]
    qr: double[4,...]
    point1: double[3,...]
    point2: double[3,...]
    r: double
    """
    #compute the length of the edge
    edge_length = d_sphere_3d(point1, point2, r)
    #compute tan and normal vector
    edge_vector = point2 - point1
    vector_tan = edge_vector / torch.sqrt(torch.square(edge_vector).sum(dim=0, keepdim=True))
    vector_nor = rotate_3d(point1, point2)
    #compute the velocity in ql and qr aligned with normal and tangent vector
    Tql = velocity_tan(ql, vector_nor, vector_tan)
    Tqr = velocity_tan(qr, vector_nor, vector_tan)

    hm, um, vm = roe_average(Tql, Tqr)
    cm = torch.sqrt(g*hm)

    dq = Tqr - Tql
    dq0 = dq[0, ...]
    dq1 = dq[1, ...]
    dq2 = dq[2, ...]

    # eigval_abs * (eigvecs_inv @ dq)
    k0 = torch.abs(um - cm) * (0.5*(um + cm)/cm * dq0 - 0.5/cm * dq1 + 0.0)
    k1 = torch.abs(um) * (-vm * dq0 + 0.0 + 1.0 * dq2)
    k2 = torch.abs(um + cm) * (-0.5*(um - cm)/cm * dq0 + 0.5/cm * dq1 + 0.0)

    d0 = k0 + 0.0 + k2
    d1 = (um - cm) * k0 + 0.0 + (um + cm) * k2
    d2 = vm * k0 + k1 + vm * k2

    diffq = torch.stack((d0, d1, d2), dim=0)
    fql = f(Tql)
    fqr = f(Tqr)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffq)
    out = velocity_xyz(tmp, vector_nor, vector_tan)
    return out


def minmod(a, b, c = None):
    if c is None:
        return torch.sign(a)*torch.minimum(torch.abs(a), torch.abs(b))*(a*b > 0.0)
    else:
        return torch.sign(a)*torch.minimum(torch.minimum(torch.abs(a), torch.abs(b)), torch.abs(c))*(a*b > 0.0)*(a*c > 0.0)

def maxmod(a, b):
    return torch.sign(a)*torch.maximum(torch.abs(a), torch.abs(b))*(a*b > 0.0)

def limiter_minmod(fd, cd, bd):
    return minmod(fd, cd, bd)

def limiter_superbee(fd, bd):
    return maxmod(minmod(2*bd, fd), minmod(bd, 2*fd))

def limiter_mc(fd, cd, bd):
    return minmod(2*fd, cd, 2*bd)

limiter = limiter_mc
flux = flux_roe

def compute_flux(q0_bound, vert_3d, r):
    """
    Finite volume method for shallow water equation
    q0: double[4, M, N]
    vert_3d: double[3, M+1, N+1]
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """

    nbatch, nx, ny = q0_bound.shape[0], q0_bound.shape[2], q0_bound.shape[3]
    device = q0_bound.device
    nhalo = 2
    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]
    out = torch.zeros((nbatch, 4, nx, ny, 2),device=device)
    for i in range(nbatch):
        #M+1, N+1
        #q0l 0 q0u 1 q0r 2 q0b 3
        q0l = q0_bound[i,...,0]
        q0r = q0_bound[i,...,1]
        q0u = q0_bound[i,...,2]
        q0b = q0_bound[i,...,3]

        q0l = boundary_condition(q0l, nhalo)
        q0r = boundary_condition(q0r, nhalo)
        q0u = boundary_condition(q0u, nhalo)
        q0b = boundary_condition(q0b, nhalo)

        #q0l 0 q0u 1 q0r 2 q0b 3
        qlp = q0r[:,nhalo-1:-nhalo-1, nhalo:-nhalo]
        qlm = q0l[:,nhalo:-nhalo, nhalo:-nhalo]

        # qrp = q0l[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
        # qrm = q0r[:, nhalo:-nhalo, nhalo:-nhalo]

        qbp = q0u[:, nhalo:-nhalo, nhalo-1:-nhalo-1]
        qbm = q0b[:, nhalo:-nhalo, nhalo:-nhalo]
        
        # qup = q0b[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
        # qum = q0u[:, nhalo:-nhalo, nhalo:-nhalo]
        flux_l = flux(qlm, qlp, vertbl, vertul, r)
        # flux_u = flux(qum, qup, vertul, vertur, r)
        # flux_r = flux(qrm, qrp, vertur, vertbr, r)
        flux_b = flux(qbm, qbp, vertbr, vertbl, r)
        out_ = torch.stack((flux_b, flux_l), axis=-1)
        out[i,...] = out_[...]
    return out

#---------------------------------------------------------------------------------------------------------------------------------------------

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
        self.data_train = torch.as_tensor(zarr.open(f'{path}/Data_Train.zarr', mode="r")[0:]).float()
        self.data_true  = torch.as_tensor(zarr.open(f'{path}/Data_True.zarr', mode="r")[0:]).float() 

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
        y_test_true = torch.as_tensor(y_test_true).float()
        #Train value
        y_test = self.data_test[index]
        y_test = torch.as_tensor(y_test).float() 
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
    def __init__(self, mu: torch.Tensor, std: torch.Tensor, mid_channel: int, nstencil_out: int, ninterp: int, stencil_x: int, stencil_y: int, vert_3d: torch.Tensor, r: float):
        super(LightningCnn, self).__init__()
        self.save_hyperparameters()
        self.mu = nn.Parameter(mu, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.vert_3d = nn.Parameter(vert_3d, requires_grad=False)
        self.r = r
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
            nn.Conv2d(self.mid_channel, self.mid_channel, self.kernel_size, 
                    padding=0),     
            nn.ReLU(),                     
        )

        # Layer 3
        self.conv3 = nn.Sequential(         
            nn.Conv2d(mid_channel, self.mid_channel, self.kernel_size, 
                    padding=0),     
            nn.ReLU(),                     
        )

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
        self.out = nn.Conv2d(self.mid_channel, 4*2, self.kernel_size,
                    padding=0)                                                 #(nbatch, 4*ninterp * (nstencil_out - norder), nx, ny)

        #Layer 7
        self.constrain = LightningConstraintLayer2D(nstencil_out, 4*ninterp)

    def forward(self, input):
        (batch_size, _, nx, ny) = input.shape                                                  #(nbatch, 4, nx, ny)
        #Normalize input data
        input_normal = (input - self.mu) / self.std
        x = input_normal.float()                                                               #(nbatch, 4, nx, ny)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv1(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv2(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = boundary_padding(x, (self.kernel_size-1)//2, (self.kernel_size-1)//2)
        x = self.out(x)                                                                        #(nbatch, 4*ninterp * (nstencil_out - norder), nx, ny)
        x = torch.reshape(x, (batch_size, 4, 2, nx, ny))
        output = x.moveaxis(2,-1)
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
    ninterp = 4
    batch_size = 32

    dataset_train = "data_flux_ur"

    data_train = zarr.open(f'{dataset_train}/Data_Train.zarr', mode="r")
    print("Data train shape:", data_train.shape)

    mu, std = torch.load(f"./mu_std_{dataset_train}.pt")
    mu[:, 0, :, :] = 0.0

    #data
    train_dataset = TrainDataset(f"./{dataset_train}")
    test_dataset = TestDataset("./data_test_flux2.pt")

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
    
    xmin, xmax = -3., 1.
    ymin, ymax = -1., 1.
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)
    r = 1.0
    M, N = 100, 50
    vert_3d = sphere_grid(x_range, y_range, M, N, r)
    vert_3d = torch.from_numpy(vert_3d)
    vert_3d = vert_3d.moveaxis(-1,0)

    if args.gpu:
        trainer = pl.Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=200, strategy="ddp_find_unused_parameters_false", logger=wandb_logger, check_val_every_n_epoch=1)
        if args.retrain:
            trainer = pl.Trainer(accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=100, strategy="ddp_find_unused_parameters_false", resume_from_checkpoint="test_flux3.ckpt", logger=wandb_logger, check_val_every_n_epoch=1)
            print("Model loaded, keep training.........")
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=100, strategy="ddp_find_unused_parameters_false", logger=wandb_logger, check_val_every_n_epoch=1)

    #Model
    model = LightningCnn(mu, std, mid_channel, nstencil_out=nstencil_out, ninterp=ninterp, stencil_x=stencil_x, stencil_y=stencil_y, vert_3d=vert_3d, r=r)

    #training
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(dataloaders=test_dataloader)
    trainer.save_checkpoint("test_flux3.ckpt")
    print("Model has been saved successfully!")
    
    # model = LightningCnn.load_from_checkpoint("cnn_reconstruct1.ckpt")
    # print("model loaded")
    # trainer.test(model, dataloaders=test_dataloader)