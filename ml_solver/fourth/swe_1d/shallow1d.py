import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import scipy.spatial

g = 9.8
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the exponentiated quadratic 
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with sigma=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def gaussian_process(xmin, xmax, nb_of_samples, number_of_functions):
    # Sample from the Gaussian process distribution

    # Independent variable samples
    X = np.expand_dims(np.linspace(xmin, xmax, nb_of_samples), 1)
    Σ = exponentiated_quadratic(X, X)  # Kernel of data points

    # Draw samples from the prior at our data points.
    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=Σ, 
        size=number_of_functions)
    return ys
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def minmaxscaler(data):
    min_data = torch.min(data)
    max_data = torch.max(data)
    return (data - min_data) / (max_data - min_data)

def initial_condition(N):
    q0 = torch.zeros(2, N)
    h, u = torch.from_numpy(gaussian_process(0.0, 10.0, N, 2))
    q0[0, ...] = minmaxscaler(h) + (0.15-0.05) * np.random.random() + 0.05
    q0[1, ...] = ((0.5-(-0.5)) * minmaxscaler(u) - 0.5)
    return q0

def initial_condition_naive(N):
    q0 = torch.zeros(2, N)
    q0[0, 0:N // 2] = 1
    q0[0, N // 2:N] = 0.35
    return q0

def f(q):
    h = q[0,...]
    uh = q[1,...]
    u = uh / h
    return torch.stack((
                        uh,
                        u*uh + 0.5*g*h*h),dim=0)

def max_eigenvalue(q):
    h = q[0,...]
    uh = q[1,...]
    u = uh / h
    return torch.sqrt(torch.abs(g*h)) + u

def roe_average(ql, qr):
    """
    Compute Roe average of ql and qr
    ql: double[3, ...]
    qr: double[3, ...]
    return: hm, um, vm: double[...]
    """
    hl = ql[0, ...]
    uhl = ql[1, ...]

    hr = qr[0, ...]
    uhr = qr[1, ...]

    ul = uhl/hl
    ur = uhr/hr

    sqrthl = torch.sqrt(hl)
    sqrthr = torch.sqrt(hr)

    hm = 0.5*(hl+hr)
    um = (sqrthl*ul + sqrthl*ur)/(sqrthl + sqrthr)

    return hm, um

def flux_naive(q):
    h = q[0,...]
    uh = q[1,...]
    u = uh / h
    return torch.stack((
                        uh,
                        u*uh + 0.5*g*h*h),dim=0)

def flux_lf(ql, qr):
    # N = ql.shape[1]
    # dx = (1.0 - 0.0) / N
    # dt = 1e-3*(128/N)
    # return 0.5*(f(ql) + f(qr)) - 0.5 * (dx / dt)* (qr - ql)
    return 0.5*(f(ql) + f(qr)) - (qr - ql)

# def flux_lf(ql, qr):
#     alpha = 2
#     return 0.5*(f(ql) + f(qr)) - 0.5 * alpha * (qr - ql)

def flux_rusanov(ql, qr):
    # Rusanov's flux scheme
    diffu = 0.5 * torch.maximum(max_eigenvalue(ql), max_eigenvalue(qr)) * (qr - ql)
    return 0.5 * (f(ql) + f(qr)) - diffu

def flux_roe(ql, qr):
    """
    Roe's numerical flux for shallow water equation
    ql: double[2,...]
    qr: double[2,...]
    """
    hm, um = roe_average(ql, qr)
    cm = torch.sqrt(g*hm)

    dq = qr - ql

    dq0 = dq[0, ...]
    dq1 = dq[1, ...]

    # eigval_abs * (eigvecs_inv @ dq)
    k0 = torch.abs(um - cm) * (0.5*(um + cm)/cm * dq0 - 0.5/cm * dq1 + 0.0)
    k1 = torch.abs(um + cm) * (-0.5*(um - cm)/cm * dq0 + 0.5/cm * dq1 + 0.0)

    d0 = k0 + 0.0 + k1
    d1 = (um - cm) * k0 + 0.0 + (um + cm) * k1

    diffq = torch.stack((d0, d1), dim=0)
    fql = f(ql)
    fqr = f(qr)
    out = 0.5  * ((fql + fqr) - diffq)
    return out

def flux_hll(ql, qr):
    """
    HLL's numerical flux for shallow water equation
    ql: double[2,...]
    qr: double[2,...]
    """
    # HLL's flux scheme
    fql = f(ql)
    fqr = f(qr)

    eigvalsl0 = ql[1,...]/ql[0,...] - torch.sqrt(torch.abs(g*ql[0,...]))
    eigvalsl1 = ql[1,...]/ql[0,...] + torch.sqrt(torch.abs(g*ql[0,...]))

    eigvalsr0 = qr[1,...]/qr[0,...] - torch.sqrt(torch.abs(g*qr[0,...]))
    eigvalsr1 = qr[1,...]/qr[0,...] + torch.sqrt(torch.abs(g*qr[0,...]))

    eigval = torch.stack((eigvalsl0, eigvalsl1, eigvalsr0, eigvalsr1), dim=0)
    sl = torch.min(eigval, dim=0, keepdim=False, out=None)[0]
    sr = torch.max(eigval, dim=0, keepdim=False, out=None)[0]

    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    out = torch.where(tmp==0.0, (sl*sr*(qr - ql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    return out

def flux_hlle(ql, qr):
    """
    HLLE's numerical flux for shallow water equation
    ql: double[2,...]
    qr: double[2,...]
    """
    # HLLE flux scheme
    # https://www.sciencedirect.com/science/article/pii/S0898122199002965
    # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics

    hm, um = roe_average(ql, qr)
    cm = torch.sqrt(g*hm)

    fql = f(ql)
    fqr = f(qr)

    ul = ql[1,...] / ql[0,...]
    ur = qr[1,...] / qr[0,...]
    cl = torch.sqrt(torch.abs(g*ql[0,...]))
    cr = torch.sqrt(torch.abs(g*qr[0,...]))
    sl = torch.minimum(um - cm, ul - cl)
    sr = torch.maximum(um + cm, ur + cr)
    
    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    out = torch.where(tmp==0.0, (sl*sr*(qr - ql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    return out

def flux_hllc(ql, qr):
    """
    HLLC's numerical flux for shallow water equation
    ql: double[2,...]
    qr: double[2,...]
    """
    # HLLC flux scheme
    # https://link.springer.com/content/pdf/10.1007%2F978-3-662-03490-3_10.pdf
    # p9 p301

    # HLLE speeds
    fql = f(ql)
    fqr = f(qr)

    hl = ql[0,...]
    hr = qr[0,...]
    
    ul = ql[1,...] / hl
    ur = qr[1,...] / hr
    
    cl = torch.sqrt(torch.abs(g*hl))
    cr = torch.sqrt(torch.abs(g*hr))
    
    um = 0.5 * (ul+ur) + cl - cr
    #hm = ((0.5*(np.sqrt(g*hl)+np.sqrt(g*hr))+0.25*(vl-vr))**2)/g
    cm = 0.25 * (ul-ur) + 0.5 * (cl+cr)

    sl = torch.where(hl>0.0, torch.minimum(um - cm, ul - cl), ur - 2.0*torch.sqrt(torch.abs(g*hr)))        
    sr = torch.where(hr>0.0, torch.maximum(um + cm, ur + cr), ul + 2.0*torch.sqrt(torch.abs(g*hl)))

    # middle state estimate
    hlm = (sl - ul)/(sl - um) * hl
    hrm = (sr - ur)/(sr - um) * hr

    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    q_cons1 = torch.where(um<0.0, 0.0, torch.stack((hlm, hlm * um), dim=0))
    tmp = torch.where(um<0.0, tmp, (fql + sl*(q_cons1 - ql)))
    q_cons2 = torch.where(tmp==0.0, torch.stack((hrm, hrm * um), dim=0), 0.0)
    out = torch.where(tmp==0.0, (fqr + sr*(q_cons2 - qr)) , tmp)
    return out

def fvm_1storder(q0, dx, dt, nhalo):
    assert nhalo > 0 and nhalo - 1 > 0

    qc = q0[:, nhalo:-nhalo]
    ql = q0[:, nhalo-1:-nhalo-1]
    qr = q0[:, nhalo+1:-nhalo+1]
    flux_l = flux(ql, qc)
    flux_r = flux(qc, qr)
    flux_sum = -flux_l + flux_r 
    q1c = qc - dt / dx * flux_sum
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

limiter = limiter_mc
def fvm_2ndorder_space_classic(q0, dx, dt, nhalo):
    assert nhalo > 0 and nhalo - 2 > 0

    fdx = q0[:, nhalo:-nhalo+2] - q0[:, nhalo-1:-nhalo+1]
    bdx = q0[:, nhalo-1:-nhalo+1] - q0[:, nhalo-2:-nhalo]
    cdx = 0.5 * (q0[:, nhalo:-nhalo+2] - q0[:, nhalo-2:-nhalo])
    slope_x = limiter(fdx, cdx, bdx)

    q0l = torch.zeros_like(q0)
    q0r = torch.zeros_like(q0)

    q0l[:, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1] - 0.5*slope_x
    q0r[:, nhalo-1:-nhalo+1] = q0[:, nhalo-1:-nhalo+1] + 0.5*slope_x
   
    qc = q0[:, nhalo:-nhalo]
    qlp = q0r[:, nhalo-1:-nhalo-1]
    qlm = q0l[:, nhalo:-nhalo]

    qrp = q0l[:, nhalo+1:-nhalo+1]
    qrm = q0r[:, nhalo:-nhalo]

    flux_l = flux(qlp, qlm)
    flux_r = flux(qrm, qrp)

    flux_sum = -flux_l + flux_r 
    q1c = qc - dt / dx * flux_sum
    q1 = boundary_condition(q1c, nhalo)
    return q1

def fvm_heun(q0, dx, dt, nhalo):
    """
    q0: double[2, N+nhalo*2], state matrix of current timestep
    q1: double[2, N+nhalo*2], state matrix of next timestep
    dt: double
    nhalo: int
    return: double[3, N+nhalo*2]
    """
    qs = fvm_2ndorder_space(q0, dx, dt, nhalo)
    qss = fvm_2ndorder_space(qs, dx, dt, nhalo)
    q1 = 0.5 * (q0 + qss)
    return q1


def fvm_TVD_RK(q0, dx, dt, nhalo):
    """
    q0: double[2, N+nhalo*2], state matrix of current timestep
    q1: double[2, N+nhalo*2], state matrix of next timestep
    dt: double
    nhalo: int
    return: double[3, N+nhalo*2]
    """
    #stage1
    qs = fvm_2ndorder_space(q0, dx, dt, nhalo)
    #stage2
    qss = (3/4)*q0 + (1/4)*fvm_2ndorder_space(qs, dx, dt, nhalo)
    #stage3
    q1 = (1/3)*q0 + (2/3)*fvm_2ndorder_space(qss, dx, dt, nhalo)
    return q1

# fixed boundary 
def boundary_condition_reflect(q, nhalo, unsqueeze=True):
    if unsqueeze:
        q = q.unsqueeze(0)
    q = F.pad(q, (nhalo, nhalo), "replicate").squeeze(0)
    q[1, 0:nhalo] *= -1
    q[1, -nhalo:] *= -1
    return q

# fixed boundary 
def boundary_condition_repeat(q, nhalo, unsqueeze=True):
    if unsqueeze:
        q = q.unsqueeze(0)
    q = F.pad(q, (nhalo, nhalo), "replicate").squeeze(0)
    return q

# periodic boundary 
def boundary_condition_periodic(q, nhalo, unsqueeze=True):
    if unsqueeze:
        q = q.unsqueeze(0)
    q = F.pad(q, (nhalo, nhalo), "circular").squeeze()
    return q

def calc_dt(q0, nhalo, cfl_number, dx):
    maxeig = torch.max(max_eigenvalue(q0[:, nhalo:-nhalo]))
    return cfl_number * dx / maxeig

def integrate(q_init, q_out, cfl_number, dx, nhalo, T, save_ts):
    nT, _, N = q_out.shape
    assert nT == len(save_ts)
    device = q_out.device

    q0 = boundary_condition(q_init, nhalo)

    T = 0.0
    it = 0
    save_it = 0

    while save_it < nT:
        if save_ts[save_it] - T < 1e-10:
            q_out[save_it, ...] = q0[:, nhalo:-nhalo]
            save_it += 1
            if save_it == nT:
                break
        dt = 1e-3*(128/N)
        q0 = fvm(q0, dx, dt, nhalo)
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = torch.zeros((2, N), device=device)
            tmp[0, ...] = 0.5 * torch.square(q0[0, nhalo:-nhalo]) * g                       #0.5* g*h^2 
            tmp[1, ...] = 0.5 * torch.square(q0[1, nhalo:-nhalo]) / q0[0, nhalo:-nhalo]     #u^2*h^2 / h = 0.5 * u^2*h
            sum_ = torch.sum(tmp * dx).item()
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)
            
# def integrate(q_init, q_out, cfl_number, dx, nhalo, T, save_ts):
#     nT, _, N = q_out.shape
#     assert nT == len(save_ts)
#     device = q_out.device

#     q0 = boundary_condition(q_init, nhalo)

#     T = 0.0
#     it = 0
#     save_it = 0

#     while save_it < nT:
#         if save_ts[save_it] - T < 1e-10:
#             q_out[save_it, ...] = q0[:, nhalo:-nhalo]
#             save_it += 1
#             if save_it == nT:
#                 break
#         dt_ = calc_dt(q0, nhalo, cfl_number, dx)
#         dt = float(min(dt_, float(save_ts[save_it]) - T))
#         q0 = fvm(q0, dx, dt, nhalo)
#         it += 1
#         T += dt
#         if it % 100 == 0:
#             tmp = torch.zeros((2, N), device=device)
#             tmp[0, ...] = 0.5 * torch.square(q0[0, nhalo:-nhalo]) * g                       #0.5* g*h^2 
#             tmp[1, ...] = 0.5 * torch.square(q0[1, nhalo:-nhalo]) / q0[0, nhalo:-nhalo]     #u^2*h^2 / h = 0.5 * u^2*h
#             sum_ = torch.sum(tmp * dx).item()
#             # u = q0[1, nhalo:-nhalo] / q0[0, nhalo:-nhalo]
#             # print("u:",u.max(), u.min())
#             # print("hu:",q0[1, nhalo:-nhalo])
#             print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=512, type=int)
    parser.add_argument("--order", default="1", type=int)
    parser.add_argument("--period", default=1.0, type=float)
    parser.add_argument("--flux", default="roe", type=str)
    parser.add_argument("--solver", default="classic", type=str)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    order = args.order
    fvm_2ndorder_space = globals()["fvm_2ndorder_space_" + args.solver]
    fvm = fvm_TVD_RK
    if order == 1:
        fvm = fvm_1storder

    nhalo = 3
    N = args.resolution_x
    Te = args.period
    flux = globals()["flux_" + args.flux]
    x_range = (0., 10.)
    xmin, xmax = x_range
    dx = (xmax-xmin) / N
    cfl_number = 0.1
    save_ts = torch.arange(0, Te+0.001, Te/N)

    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    flux = globals()["flux_" + args.flux]
    initial_condition = initial_condition_naive
    # initial_condition = initial_condition
    boundary_condition = boundary_condition_periodic

    q_init = initial_condition(N)
    q_out = torch.zeros(len(save_ts), 2, N, device=device)
    integrate(q_init, q_out, cfl_number, dx, nhalo, Te, save_ts)
    q_out = q_out.cpu().numpy()

    if args.plot:
        plt.figure()
        plt.imshow(q_out[:,0,:], origin="lower", cmap = "viridis")
        plt.colorbar()
        plt.show()
        plt.savefig(f"test-{N}-{order}-classic")
