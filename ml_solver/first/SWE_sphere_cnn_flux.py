import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.utils.data as Data
import zarr
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
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
    nhalo = 3
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
#---------------------------------------------------------------------------------------------------------------------------------------------

g = 11489.57219

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

def max_eigenvalue(q):
    """
    Maximum eigenvalue
    q: double, state vector
    """
    h = q[0, ...]
    uh = q[1, ...]
    vh = q[2, ...]
    u = uh / h
    v = vh / h
    res = torch.sqrt(torch.abs(g*h)) + torch.abs(u)
    return res

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

def fn(q, n):
    """
    Flux function for shallow water equation on the direction of n
    q: double[4, ...]
    n: double[3, ...]
    return: double[3, ...]
    """
    h = q[0, ...]
    uh = q[1, ...]
    vh = q[2, ...]
    wh = q[3, ...]
    u = uh/h
    v = vh/h
    w = wh/h
    n0 = n[0, ...]
    n1 = n[1, ...]
    n2 = n[2, ...]
    out0 = uh*n0 + vh*n1 + wh*n2
    out1 = (uh*u + 0.5*g*h*h)*n0 + uh*v*n1 + uh*w*n2
    out2 = uh*v*n0 + (vh*v + 0.5*g*h*h)*n1 + vh*w*n2
    out3 = uh*w*n0 + vh*w*n1 + (wh*w+0.5*g*h*h)*n2
    return torch.stack((out0, out1, out2, out3), dim=0)

def flux_rusanov(ql, qr, point1, point2, r):
    """
    Rusanov flux scheme
    out: double[4,...], accumulate result with the out array
    ql: double[4,...], left state
    qr: double[4,...], right state
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
    #compute the velocity in ql and qr along latitude and longtitude
    Tql = velocity_tan(ql, vector_nor, vector_tan)
    Tqr = velocity_tan(qr, vector_nor, vector_tan)
    #compute the numerical flux and rotate it back to xyz coordinate system
    diffq = torch.maximum(max_eigenvalue(Tql), max_eigenvalue(Tqr)) * (Tqr - Tql)
    fql = f(Tql)
    fqr = f(Tqr)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffq)
    out = velocity_xyz(tmp, vector_nor, vector_tan)
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

def flux_hll(ql, qr, point1, point2, r):
    # HLL's flux scheme
    #compute the length of the edge
    edge_length = d_sphere_3d(point1, point2, r)
    #compute tan and normal vector
    edge_vector = point2 - point1
    vector_tan = edge_vector / torch.sqrt(torch.square(edge_vector).sum(dim=0, keepdim=True))
    vector_nor = rotate_3d(point1, point2)
    #compute the velocity in ql and qr aligned with normal and tangent vector
    Tql = velocity_tan(ql, vector_nor, vector_tan)
    Tqr = velocity_tan(qr, vector_nor, vector_tan)

    fql = f(Tql)
    fqr = f(Tqr)

    eigvalsl0 = Tql[1,...]/Tql[0,...] - torch.sqrt(torch.abs(g*Tql[0,...]))
    eigvalsl1 = Tql[1,...]/Tql[0,...]
    eigvalsl2 = Tql[1,...]/Tql[0,...] + torch.sqrt(torch.abs(g*Tql[0,...]))

    eigvalsr0 = Tqr[1,...]/Tqr[0,...] - torch.sqrt(torch.abs(g*Tqr[0,...]))
    eigvalsr1 = Tqr[1,...]/Tqr[0,...]
    eigvalsr2 = Tqr[1,...]/Tqr[0,...] + torch.sqrt(torch.abs(g*Tqr[0,...]))

    eigval = torch.stack((eigvalsl0, eigvalsl1, eigvalsl2, eigvalsr0, eigvalsr1, eigvalsr2), dim=0)
    sl = torch.min(eigval, dim=0, keepdim=False, out=None)[0]
    sr = torch.max(eigval, dim=0, keepdim=False, out=None)[0]

    tmp = torch.where(sl>=0.0, fql, 0.0)
    tmp = torch.where(sr<=0.0, fqr, tmp)
    tmp = torch.where(tmp==0.0, (sl*sr*(Tqr - Tql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    out = velocity_xyz(tmp * edge_length, vector_nor, vector_tan)
    return out

def flux_hlle(ql, qr, point1, point2, r):
    # HLLE flux scheme
    # https://www.sciencedirect.com/science/article/pii/S0898122199002965
    # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics

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

    fql = f(Tql)
    fqr = f(Tqr)

    vl = Tql[1,...] / Tql[0,...]
    vr = Tqr[1,...] / Tqr[0,...]
    cl = torch.sqrt(torch.abs(g*Tql[0,...]))
    cr = torch.sqrt(torch.abs(g*Tqr[0,...]))
    sl = torch.minimum(vm - cm, vl - cl)
    sr = torch.maximum(vm + cm, vr + cr)
    
    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    tmp = torch.where(tmp==0.0, (sl*sr*(Tqr - Tql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    out = velocity_xyz(tmp * edge_length, vector_nor, vector_tan)
    return out


def flux_hllc(ql, qr, point1, point2, r):
    # HLLC flux scheme
    # https://link.springer.com/content/pdf/10.1007%2F978-3-662-03490-3_10.pdf
    # p9 p301

    # HLLE speeds
    #compute the length of the edge
    edge_length = d_sphere_3d(point1, point2, r)
    #compute tan and normal vector
    edge_vector = point2 - point1
    vector_tan = edge_vector / torch.sqrt(torch.square(edge_vector).sum(dim=0, keepdim=True))
    vector_nor = rotate_3d(point1, point2)
    #compute the velocity in ql and qr aligned with normal and tangent vector
    Tql = velocity_tan(ql, vector_nor, vector_tan)
    Tqr = velocity_tan(qr, vector_nor, vector_tan)

    fql = f(Tql)
    fqr = f(Tqr)

    hl = Tql[0,...]
    hr = Tqr[0,...]
    
    vl = Tql[1,...] / hl
    vr = Tqr[1,...] / hr
    
    wl = Tql[2,...] / hl
    wr = Tqr[2,...] / hr
    
    cl = torch.sqrt(torch.abs(g*hl))
    cr = torch.sqrt(torch.abs(g*hr))
    
    vm = 0.5 * (vl+vr) + cl - cr
    #hm = ((0.5*(np.sqrt(g*hl)+np.sqrt(g*hr))+0.25*(vl-vr))**2)/g
    cm = 0.25 * (vl-vr) + 0.5 * (cl+cr)

    sl = torch.where(hl>0.0, torch.minimum(vm - cm, vl - cl), vr - 2.0*torch.sqrt(torch.abs(g*hr)))        
    sr = torch.where(hr>0.0, torch.maximum(vm + cm, vr + cr), vl + 2.0*torch.sqrt(torch.abs(g*hl)))

    # middle state estimate
    hlm = (sl - vl)/(sl - vm) * hl
    hrm = (sr - vr)/(sr - vm) * hr

    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    q_cons1 = torch.where(vm<0.0, 0.0, torch.stack((hlm, hlm * vm, hlm * wl), dim=0))
    tmp = torch.where(vm<0.0, tmp, (fql + sl*(q_cons1 - Tql)))
    q_cons2 = torch.where(tmp==0.0, torch.stack((hrm, hrm * vm, hrm * wr), dim=0), 0.0)
    tmp = torch.where(tmp==0.0, (fqr + sr*(q_cons2 - Tqr)) , tmp)
    out = velocity_xyz(tmp * edge_length, vector_nor, vector_tan)
    return out

def correction(q, vertbl, vertul, vertur, vertbr, r):
    """
    Correction term for conservation on spherical domain
    q: double[4, ...]
    vertbl: double[3, ...]
    vertul: double[3, ...]
    vertur: double[3, ...]
    vertbr: double[3, ...]
    r: double
    """
    n_l = rotate_3d(vertbl, vertul)
    n_u = rotate_3d(vertul, vertur)
    n_r = rotate_3d(vertur, vertbr)
    n_b = rotate_3d(vertbr, vertbl)

    h_l = d_sphere_3d(vertbl, vertul, r)
    h_u = d_sphere_3d(vertul, vertur, r)
    h_r = d_sphere_3d(vertur, vertbr, r)
    h_b = d_sphere_3d(vertbr, vertbl, r)

    cor = fn(q, -h_l*n_l - h_r*n_r - h_b*n_b - h_u*n_u)
    cor[0, ...] = 0.0

    return cor

def momentum_on_tan(q, cell_center):
    """
    Project momentum part of q onto tangent plane crossing cell center
    q: double[4, ...]
    cell_center: double[3, ...]
    """
    cor = (q[1:, ...] * cell_center).sum(dim=0, keepdim=True)
    out = torch.cat((q[0, ...].unsqueeze(0), q[1:, ...] - cor*cell_center), dim=0)
    return out

def coriolis(q, cell_center, dt):
    # The source term models the Coriolis force using a 4-stage RK method
    df = 12.600576
    M = q.shape[1]
    N = q.shape[2]
    fq = torch.zeros_like(q)
    device = q.device
    RK = torch.zeros((4, 3, M, N), device=device)

    #compute the cell center's sphere tangential vector along latitude and longtitude
    cell_center = cell_center / torch.sqrt(cell_center[0,...]**2 + cell_center[1,...]**2 + cell_center[2,...]**2)
    erx = cell_center[0,...]
    ery = cell_center[1,...]
    erz = cell_center[2,...]
    # calculate Coriolis term
    fcor = df * erz

    #stage 1
    hu_1 = q[1,...]
    hv_1 = q[2,...]
    hw_1 = q[3,...]
                   
    RK[0,0,...] = fcor*dt*(erz*hv_1-ery*hw_1)
    RK[0,1,...] = fcor*dt*(erx*hw_1-erz*hu_1)
    RK[0,2,...] = fcor*dt*(ery*hu_1-erx*hv_1)
    #stage 2
    hu_2 = q[1,...] + 0.5*RK[0,0,...]
    hv_2 = q[2,...] + 0.5*RK[0,1,...]
    hw_2 = q[3,...] + 0.5*RK[0,2,...]

    RK[1,0,...] = fcor*dt*(erz*hv_2-ery*hw_2)
    RK[1,1,...] = fcor*dt*(erx*hw_2-erz*hu_2)
    RK[1,2,...] = fcor*dt*(ery*hu_2-erx*hv_2)
    #stage 3
    hu_3 = q[1,...] + 0.5*RK[1,0,...]
    hv_3 = q[2,...] + 0.5*RK[1,1,...]
    hw_3 = q[3,...] + 0.5*RK[1,2,...]

    RK[2,0,...] = fcor*dt*(erz*hv_3-ery*hw_3)
    RK[2,1,...] = fcor*dt*(erx*hw_3-erz*hu_3)
    RK[2,2,...] = fcor*dt*(ery*hu_3-erx*hv_3)
    #stage 4
    hu_4 = q[1,...] + 0.5*RK[2,0,...]
    hv_4 = q[2,...] + 0.5*RK[2,1,...]
    hw_4 = q[3,...] + 0.5*RK[2,2,...]

    RK[3,0,...] = fcor*dt*(erz*hv_4-ery*hw_4)
    RK[3,1,...] = fcor*dt*(erx*hw_4-erz*hu_4)
    RK[3,2,...] = fcor*dt*(ery*hu_4-erx*hv_4)

    for i in range(1,4):
        fq[i,...] = (RK[0,i-1,...] + 2.0*RK[1,i-1,...] + 2.0*RK[2,i-1,...] + RK[3,i-1,...]) / 6.0
    fq[0,...] = 0.0 
    return fq

def initial_condition(x_range, y_range, M, N, cell_center):
    #Rossby-Haurwitz wave
    a = 6.37122e6
    K = 7.848e-6
    G = 9.8
    t0 = 86400.
    h0 = 8.e3
    r = 4.0
    omega = 7.292e-5
    q_init = torch.zeros((4, M, N))
    vert_3d = torch.zeros((3, M, N))
    vert_sphere = torch.zeros((2, M, N))

    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = torch.linspace(xmin+dx/2, xmax-dx/2, M)
    y = torch.linspace(ymin+dy/2, ymax-dy/2, N)
    vert = torch.stack(torch.meshgrid(x, y, indexing="ij"), axis=0)
    #initial vert_sphere, vert_3d
    MeshGrid.transfer_sphere_surface(vert_3d, vert, r)
    MeshGrid.xyz_to_latlon(vert_sphere, vert_3d, r)

    xp = vert_sphere[1, :,:] * torch.pi/ 180
    yp = vert_sphere[0, :,:] * torch.pi/ 180

    A = 0.5 * K * (2.0*omega + K) * torch.cos(yp)**2.0 + \
                0.25 * K * K * torch.cos(yp)**(2.0*r)*( \
                (1.0*r+1.0) * torch.cos(yp)**2.0 \
                +(2.0*r*r - 1.0*r - 2.0) \
                - 2.0*r*r*(torch.cos(yp))**(-2.0))

    B = (2.0*(omega+K) * K) / ((1.0*r+1.0)*(1.0*r+2.0)) \
                *torch.cos(yp)**r*( (1.0*r*r + 2.0*r + 2.0) \
                - (1.0*r+1.0)**(2)*torch.cos(yp)**2)

    C = 0.25*K*K*torch.cos(yp)**(2*r)*( (1.0*r + 1.0)* \
                torch.cos(yp)**2 - (1.0*r + 2.0))

    x_angular = (K*torch.cos(yp)+K*torch.cos(yp)**(r-1.)*(r*torch.sin(yp)**2.0 \
                - torch.cos(yp)**2.0 )*torch.cos(r*xp))*t0

    y_angular = (-K*r*torch.cos(yp)**(r-1.0)*torch.sin(yp)*torch.sin(r*xp))*t0

    z_angular = 0.0

    u = (-torch.sin(xp)*x_angular-torch.sin(yp)*torch.cos(xp)*y_angular)
    v = (torch.cos(xp)*x_angular-torch.sin(yp)*torch.sin(xp)*y_angular)
    w = torch.cos(yp)*y_angular

    # q = -np.sin(xp)*np.cos(yp)*x_angular - np.cos(xp)*np.sin(yp)*y_angular
    # v = np.cos(xp)*np.cos(yp)*x_angular - np.sin(xp)*np.sin(yp)*y_angular
    # w = np.cos(yp)*y_angular

    q_init[0,:,:] =  h0/a + (a/G)*( A + B*torch.cos(r*xp) \
                + C*torch.cos(2.0*r*xp))
    q_init[1,...] = q_init[0,...] * u
    q_init[2,...] = q_init[0,...] * v
    q_init[3,...] = q_init[0,...] * w

    # project momentum components of q onto tangent plane:
    tmp = q_init[1,...]*cell_center[0,...] + q_init[2,...]*cell_center[1,...] + q_init[3,...]*cell_center[2,...]
    q_init[1,...] -= tmp*cell_center[0,...]
    q_init[2,...] -= tmp*cell_center[1,...]
    q_init[3,...] -= tmp*cell_center[2,...]
    return q_init

def fvm_1storder(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn):
    """
    Finite volume method for shallow water equation
    q0: double[4, M+nhalo*2, N+nhalo*2]
    vert_3d: double[3, M+1, N+1]
    cell_center: double[3, M, N]
    cell_area: double[M, N]
    dt: double
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 1 > 0

    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]

    qc = q0[:, nhalo:-nhalo, nhalo:-nhalo]
    ql = q0[:, nhalo-1:-nhalo-1, nhalo:-nhalo]
    qu = q0[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qr = q0[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qb = q0[:, nhalo:-nhalo, nhalo-1:-nhalo-1]

    flux_l = flux(qc, ql, vertbl, vertul, r)
    flux_u = flux(qc, qu, vertul, vertur, r)
    flux_r = flux(qc, qr, vertur, vertbr, r)
    flux_b = flux(qc, qb, vertbr, vertbl, r)

    flux_sum = flux_l + flux_u + flux_r + flux_b

    fq = coriolis(qc, cell_center, dt)
    cor = correction(qc, vertbl, vertul, vertur, vertbr, r)

    flux_sum = momentum_on_tan(flux_sum, cell_center)
    cor = momentum_on_tan(cor, cell_center)

    q1c_upd = -dt/cell_area * (flux_sum + cor)

    q1c = qc + q1c_upd +fq
    q1 = boundary_condition(q1c, nhalo)
    return q1

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

def limiter_minmod2(r):
    tmp = torch.where(r<1, r, 1)
    return torch.where(tmp>0.0, tmp, 0)
    # return torch.max(0.0, torch.min(1, r))

def limiter_van_leer(r):
    return (r + torch.abs(r))/(1+torch.abs(r))

def limiter_koren(r):
    tmp = torch.where((1+2*r)/3<2, (1+2*r)/3, 2)
    tmp = torch.where(tmp<(2*r), tmp, 2*r)
    return torch.where(tmp>0, tmp, 0)

limiter = limiter_mc
limiter_cnn = limiter_minmod2

def fvm_2ndorder_space_cnn(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn):
    """
    Finite volume method for shallow water equation
    q0: double[4, M+nhalo*2, N+nhalo*2]
    vert_3d: double[3, M+1, N+1]
    cell_center: double[3, M, N]
    cell_area: double[M, N]
    dt: double
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 2 > 0
    output = cnn(q0[:, nhalo:-nhalo, nhalo:-nhalo].unsqueeze(0))
    output = output.squeeze(0)
    output = torch.moveaxis(output, -1, 0).double() # [4, M, N，2] -> [2, 4, M, N]

    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]
    #M+1, N+1
    qc = q0[:, nhalo:-nhalo, nhalo:-nhalo]

    flux_v = boundary_condition(output[0,...], nhalo)
    flux_h = boundary_condition(output[1,...], nhalo)

    flux_r = -flux_h[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    flux_b = flux_v[:, nhalo:-nhalo, nhalo:-nhalo]
    flux_l = flux_h[:, nhalo:-nhalo, nhalo:-nhalo]
    flux_u = -flux_v[:, nhalo:-nhalo, nhalo+1:-nhalo+1]

    flux_sum = flux_l + flux_u + flux_r + flux_b

    fq = coriolis(qc, cell_center, dt)
    cor = correction(qc, vertbl, vertul, vertur, vertbr, r)

    flux_sum = momentum_on_tan(flux_sum, cell_center)
    cor = momentum_on_tan(cor, cell_center)

    q1c_upd = -dt/cell_area * (flux_sum + cor)

    q1c = qc + q1c_upd + fq
    q1 = boundary_condition(q1c, nhalo)
    return q1

def fvm_2ndorder_space_cnn_limiter(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn):
    """
    Finite volume method for shallow water equation
    q0: double[4, M+nhalo*2, N+nhalo*2]
    vert_3d: double[3, M+1, N+1]
    cell_center: double[3, M, N]
    cell_area: double[M, N]
    dt: double
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 2 > 0
    output = cnn(q0[:, nhalo:-nhalo, nhalo:-nhalo].unsqueeze(0))
    output = output.squeeze(0)
    output = torch.moveaxis(output, -1, 0).double() # [4, M, N，2] -> [2, 4, M, N]

    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]
    #compute x-direction slope
    fdx1 = q0[:, nhalo-1:-nhalo, nhalo:-nhalo] - q0[:, nhalo-2:-nhalo-1, nhalo:-nhalo]
    fdx2 = q0[:, nhalo:-nhalo+1, nhalo:-nhalo] - q0[:, nhalo-1:-nhalo, nhalo:-nhalo]
    rx = fdx1 / (fdx2 + 1e-10)
    xphi = limiter_cnn(rx)
    # #compute y-direction slope
    fdy1 = q0[:, nhalo:-nhalo, nhalo-1:-nhalo] - q0[:, nhalo:-nhalo, nhalo-2:-nhalo-1]
    fdy2 = q0[:, nhalo:-nhalo, nhalo:-nhalo+1] - q0[:, nhalo:-nhalo, nhalo-1:-nhalo]

    ry = fdy1 / (fdy2 + 1e-10)
    yphi = limiter_cnn(ry)

    #M+1, N+1
    qc = q0[:, nhalo:-nhalo, nhalo:-nhalo]
    ql = q0[:, nhalo-1:-nhalo-1, nhalo:-nhalo]
    qu = q0[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qr = q0[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qb = q0[:, nhalo:-nhalo, nhalo-1:-nhalo-1]

    flux_l_low = flux(qc, ql, vertbl, vertul, r)
    flux_u_low = flux(qc, qu, vertul, vertur, r)
    flux_r_low = flux(qc, qr, vertur, vertbr, r)
    flux_b_low = flux(qc, qb, vertbr, vertbl, r)

    flux_v = boundary_condition(output[0,...], nhalo)
    flux_h = boundary_condition(output[1,...], nhalo)

    flux_r_high = -flux_h[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    flux_b_high = flux_v[:, nhalo:-nhalo, nhalo:-nhalo]
    flux_l_high = flux_h[:, nhalo:-nhalo, nhalo:-nhalo]
    flux_u_high = -flux_v[:, nhalo:-nhalo, nhalo+1:-nhalo+1]

    flux_l = flux_l_low - xphi[:, :-1, :]*(flux_l_low - flux_l_high)
    flux_u = flux_u_low - yphi[:, :, 1:]*(flux_u_low - flux_u_high)
    flux_r = flux_r_low - xphi[:, 1:, :]*(flux_r_low - flux_r_high)
    flux_b = flux_b_low - yphi[:, :, :-1]*(flux_b_low - flux_b_high)

    flux_sum = flux_l + flux_u + flux_r + flux_b

    fq = coriolis(qc, cell_center, dt)
    cor = correction(qc, vertbl, vertul, vertur, vertbr, r)

    flux_sum = momentum_on_tan(flux_sum, cell_center)
    cor = momentum_on_tan(cor, cell_center)

    q1c_upd = -dt/cell_area * (flux_sum + cor)

    q1c = qc + q1c_upd + fq
    q1 = boundary_condition(q1c, nhalo)
    return q1

def fvm_2ndorder_space_classic(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn):
    """
    Finite volume method for shallow water equation
    q0: double[4, M+nhalo*2, N+nhalo*2]
    vert_3d: double[3, M+1, N+1]
    cell_center: double[3, M, N]
    cell_area: double[M, N]
    dt: double
    r: double
    return: double[4, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 2 > 0

    #q0l 0 q0u 1 q0r 2 q0b 3
    #compute x-direction slope
    fdx = q0[:, nhalo:-nhalo+2, nhalo-1:-nhalo+1] - q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1]
    bdx = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - q0[:, nhalo-2:-nhalo, nhalo-1:-nhalo+1]
    cdx = 0.5 * (q0[:, nhalo:-nhalo+2, nhalo-1:-nhalo+1] - q0[:, nhalo-2:-nhalo, nhalo-1:-nhalo+1])
    slope_x = limiter(fdx, cdx, bdx)
    #compute y-direction slope
    fdy = q0[:, nhalo-1:-nhalo+1, nhalo:-nhalo+2] - q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1]
    bdy = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - q0[:, nhalo-1:-nhalo+1, nhalo-2:-nhalo]
    cdy = 0.5 * (q0[:, nhalo-1:-nhalo+1, nhalo:-nhalo+2] - q0[:, nhalo-1:-nhalo+1, nhalo-2:-nhalo])
    slope_y = limiter(fdy, cdy, bdy)

    vertbl = vert_3d[:, :-1, :-1]
    vertul = vert_3d[:, :-1, 1:]
    vertur = vert_3d[:, 1:, 1:]
    vertbr = vert_3d[:, 1:, :-1]
    #M+1, N+1
    q0l = torch.zeros_like(q0)
    q0r = torch.zeros_like(q0)
    q0u = torch.zeros_like(q0)
    q0b = torch.zeros_like(q0)

    q0l[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - 0.5*slope_x
    q0r[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] + 0.5*slope_x
    q0u[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] + 0.5*slope_y
    q0b[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1, nhalo-1:-nhalo+1] - 0.5*slope_y

    #q0l 0 q0u 1 q0r 2 q0b 3
    qc = q0[:, nhalo:-nhalo, nhalo:-nhalo]
    qlp = q0r[:,nhalo-1:-nhalo-1, nhalo:-nhalo]
    qlm = q0l[:,nhalo:-nhalo, nhalo:-nhalo]

    qrp = q0l[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qrm = q0r[:, nhalo:-nhalo, nhalo:-nhalo]

    qbp = q0u[:, nhalo:-nhalo, nhalo-1:-nhalo-1]
    qbm = q0b[:, nhalo:-nhalo, nhalo:-nhalo]
    
    qup = q0b[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qum = q0u[:, nhalo:-nhalo, nhalo:-nhalo]

    flux_l = flux(qlm, qlp, vertbl, vertul, r)
    flux_u = flux(qum, qup, vertul, vertur, r)
    flux_r = flux(qrm, qrp, vertur, vertbr, r)
    flux_b = flux(qbm, qbp, vertbr, vertbl, r)

    flux_sum = flux_l + flux_u + flux_r + flux_b

    fq = coriolis(qc, cell_center, dt)
    cor = correction(qc, vertbl, vertul, vertur, vertbr, r)

    flux_sum = momentum_on_tan(flux_sum, cell_center)
    cor = momentum_on_tan(cor, cell_center)

    q1c_upd = -dt/cell_area * (flux_sum + cor)

    q1c = qc + q1c_upd + fq
    q1 = boundary_condition(q1c, nhalo)
    return q1

def fvm_heun(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn):
    """
    q0: double[3,M+nhalo*2,N+nhalo*2], state matrix of current timestep
    q1: double[3,M+nhalo*2,N+nhalo*2], state matrix of next timestep
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    us = fvm_2ndorder_space(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
    uss = fvm_2ndorder_space(us, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
    q1 = 0.5 * (q0 + uss)
    return q1

def fvm_TVD_RK(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn):
    """
    q0: double[3,M+nhalo*2,N+nhalo*2], state matrix of current timestep
    q1: double[3,M+nhalo*2,N+nhalo*2], state matrix of next timestep
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    #stage1
    qs = fvm_2ndorder_space(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
    #stage2
    qss = (3/4)*q0 + (1/4)*fvm_2ndorder_space(qs, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
    #stage3
    q1 = (1/3)*q0 + (2/3)*fvm_2ndorder_space(qss, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
    # print(torch.all(q1[0,:,:]>0))
    return q1

def integrate(q_out, q_init, vert_3d, cell_center, cell_area, r, save_ts, nhalo, cnn):
    """
    Solve shallow water equation on sphere
    q_out: double[nT, 4, M, N]
    q_init: double[4, M, N], initial condition
    vert_3d: double[3, M+1, N+1], matrix of vertices
    cell_center: double[3, M, N], matrix of cell area 
    cell_area: double[M, N], matrix of cell area 
    save_ts: double[nT], array of timesteps that need to be saved
    """
    device = q_out.device
    nT, _, M, N = q_out.shape
    assert nT == len(save_ts)
    q0 = boundary_condition(q_init, nhalo)

    T = 0.0
    it = 0
    save_it = 0
    dt_ = 0.001*(128/max(N, M/2))
    dt = min(dt_, float(save_ts[save_it] - T))

    # q0 = fvm(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
    # q_out[0, ...] = q0[:, 2:M+2, 2:N+2]

    while save_it < nT:
        if save_ts[save_it] - T < 1e-10:
            q_out[save_it, ...] = q0[:, nhalo:-nhalo, nhalo:-nhalo]
            save_it += 1
            if save_it == nT:
                break
        dt_ = 0.001*(128/max(N, M/2))
        dt = min(dt_, float(save_ts[save_it] - T))
        q0 = fvm(q0, vert_3d, cell_center, cell_area, dt, r, nhalo, cnn)
        it += 1
        T += dt
        if it % 10 == 0:
            tmp = torch.zeros((2, M, N), device=device)
            tmp[0, ...] = 0.5 * torch.square(q0[0, nhalo:-nhalo, nhalo:-nhalo]) * g
            tmp[1, ...] = 0.5 * (torch.square(q0[1, nhalo:-nhalo, nhalo:-nhalo]) + torch.square(q0[2, nhalo:-nhalo, nhalo:-nhalo]) + torch.square(q0[3, nhalo:-nhalo, nhalo:-nhalo])) / q0[0, nhalo:-nhalo, nhalo:-nhalo]
            sum_ = torch.sum(tmp * cell_area[None, ...]).item()
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

def energy_cal(q, cell_area):
    potential = 0.5 * torch.square(q[:, 0, ...]) * g
    kinetic = 0.5 * (torch.square(q[:, 1,...]) + torch.square(q[:, 2, ...])) / q[:, 0, ...]
    energy = ((potential + kinetic) * cell_area).sum(dim=(-2, -1))
    potential = (potential * cell_area).sum(dim=(-2, -1))
    Total = torch.stack((energy, potential), dim=0)
    return Total

if __name__=="__main__":
    from argparse import ArgumentParser
    import MeshGrid
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=100, type=int)
    parser.add_argument("--resolution_y", default=50, type=int)
    parser.add_argument("--order", default="2", type=int)
    parser.add_argument("--period", default=1.0, type=float)
    parser.add_argument("--flux", default="roe", type=str)
    parser.add_argument("--solver", default="cnn", type=str)
    parser.add_argument("--test_energy", default="store_true", type=str)

    args = parser.parse_args()
    
    M = args.resolution_x
    N = args.resolution_y
    Te = args.period
    flux = globals()["flux_" + args.flux]
    fvm_2ndorder_space = globals()["fvm_2ndorder_space_" + args.solver]
    fvm = fvm_TVD_RK
    if args.order == 1:
        fvm = fvm_1storder

    x_range = (-3., 1.)
    y_range = (-1., 1.)
    r = 1.0
    nhalo = 3
    vert_3d, vert_lonlat, cell_area, cell_center_3d = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
    q_init = initial_condition(x_range, y_range, M, N, cell_center_3d)

    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    q_init = q_init.to(device)
    vert_3d = vert_3d.to(device)
    cell_area = cell_area.to(device)
    cell_center_3d = cell_center_3d.to(device)

    if args.solver =="classic":
        cnn = None
    else:
        cnn = LightningCnn.load_from_checkpoint("./test_flux.ckpt")

    print(f"Sphere mesh grid {M}-{N} successfully generate!")

    save_ts = torch.arange(0, args.period+0.01, args.period/10., device=device)
    q_out = torch.zeros((len(save_ts), 4, M, N), device=device, requires_grad=False)

    print("Start computing------------------------")
    start_time = time.time()
    integrate(q_out, q_init, vert_3d, cell_center_3d, cell_area, r, save_ts, nhalo, cnn)
    end_time = time.time()
    print(f"End computing------------------------\nComputing time:",end_time - start_time,"s")

    path = f"{os.path.dirname(os.path.realpath(__file__))}/../data/torch/{Te}/{args.flux}"
    if not os.path.exists(path):
        os.makedirs(path)

    save_ts = save_ts.cpu().numpy()
    q_out = q_out.cpu().detach().numpy()
    vert_3d = vert_3d.cpu()
    MeshGrid.save_sphere_grid_xdmf(f"{path}/sphere-fvm-{M}-{N}-{args.flux}-{args.solver}.xdmf", zip(save_ts, q_out), vert_3d)

    if args.test_energy:
        save_ts = torch.arange(0, args.period+0.01, args.period/5., device=device)
        q_out1 = torch.zeros((len(save_ts), 4, M, N), device=device, requires_grad=False)   #1st order
        q_out2 = torch.zeros_like(q_out1)                                                   #2nd order
        q_out3 = torch.zeros_like(q_out1)                                                   #cnn

        q_init = q_init.to(device)
        vert_3d = vert_3d.to(device)
        cell_area = cell_area.to(device)
        cell_center_3d = cell_center_3d.to(device)

        print("Classic low result:")
        fvm_2ndorder_space = fvm_2ndorder_space_classic
        fvm = fvm_1storder
        integrate(q_out1, q_init, vert_3d, cell_center_3d, cell_area, r, save_ts, nhalo, cnn=None)

        print("Classic high result:")
        fvm_2ndorder_space = fvm_2ndorder_space_classic
        fvm = fvm_TVD_RK
        integrate(q_out2, q_init, vert_3d, cell_center_3d, cell_area, r, save_ts, nhalo, cnn=None)

        print("Cnn result:")
        with torch.no_grad():
            fvm_2ndorder_space = fvm_2ndorder_space_cnn
            cnn = LightningCnn.load_from_checkpoint("./test_flux.ckpt")
            fvm = fvm_TVD_RK
            integrate(q_out3, q_init, vert_3d, cell_center_3d, cell_area, r, save_ts, nhalo, cnn=cnn)

        energy1 = energy_cal(q_out1, cell_area)
        energy2 = energy_cal(q_out2, cell_area)
        energy3 = energy_cal(q_out3, cell_area)

        q_out1 = q_out1.cpu().numpy()                 
        q_out2 = q_out2.cpu().numpy()
        q_out3 = q_out3.cpu().numpy()

        save_ts = save_ts.cpu().numpy()
        dataset = np.array([q_out1, q_out2, q_out3])
        fname = ["classic 1st", "classic 2nd", "cnn"]

        fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 20), dpi=400)
        for i in range(len(fname)):
            for j in range(5):
                pcm = axs[i,j].imshow(dataset[i, j, 0, ...], cmap='viridis', origin="lower")
                if j == 0:
                    axs[i,j].set_title(f"{fname[i].capitalize()} t=0 day")
                else:
                    axs[i,j].set_title("t={time} days".format(time = '%.1f'%save_ts[j]))
                # divider = make_axes_locatable(axs[i,j])
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # fig.colorbar(pcm, cax=cax)
                axs[i,j].set_xlabel('x')
                axs[i,j].set_ylabel('y')
                # axs[i,j].axis("off")
                axs[i,j].xaxis.set_ticks([])
                axs[i,j].yaxis.set_ticks([])

        plt.tight_layout()
        plt.savefig(f"./test-cnn-sphere.png")
        plt.show()

        energy1 = energy1.cpu().numpy()
        energy2 = energy2.cpu().numpy()
        energy3 = energy3.cpu().numpy()

        plt.figure(figsize=(20, 8), dpi=400)
        #plot 1:
        plt.subplot(1, 2, 1)
        plt.plot(save_ts, energy1[0,...], label="classic 1st order")
        plt.plot(save_ts, energy2[0,...], label="classic 2nd order")
        plt.plot(save_ts, energy3[0,...], label="data-driven solver")   
        plt.xlabel('Time (days)')
        plt.ylabel('Total energy')
        plt.ylim(bottom=energy1[0,0] - 0.01)
        plt.title('Total Energy comparison')
        plt.legend()

        #plot 2:
        plt.subplot(1, 2, 2)
        plt.plot(save_ts, energy1[1,...], label="classic 1st order")
        plt.plot(save_ts, energy2[1,...], label="classic 2nd order")
        plt.plot(save_ts, energy3[1,...], label="data-driven solver")  
        plt.xlabel('Time (days)')
        plt.ylabel('Potential Enstrophy')
        plt.ylim(bottom=energy1[0,0] - 0.01)
        plt.title('Potential Enstrophy comparison')
        plt.legend()

        plt.show()
        plt.savefig("./Engergy.png")
