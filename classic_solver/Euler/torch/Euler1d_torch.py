# Numerical solver for 1D Euler equation under Torch framework
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import Image
# torch.set_default_dtype(torch.float64)

gamma = 7/5
cfl_number = 0.3

def f(q):
    rho = q[0,...]
    rhov = q[1,...]
    E = q[2,...]
    v = rhov/rho
    p = (gamma-1)*(E - 0.5*rhov*v)
    return torch.stack((rhov,
                        rhov*v + p,
                        (E+p)*v),dim=0)

def max_eigenvalue(q):
    rho = q[0,...]
    rhov = q[1,...]
    E = q[2,...]
    v = rhov/rho
    p = (gamma-1)*(E - 0.5*rhov*v)
    c = torch.sqrt(gamma*p/rho)
    return torch.abs(v) + c

def eigenvalues(q):
    rho = q[0,...]
    rhov = q[1,...]
    E = q[2,...]
    v = rhov/rho
    p = (gamma-1)*(E - 0.5*rhov*v)
    c = torch.sqrt(gamma*p/rho)
    return torch.stack((v - c,
                        v,
                        v + c), dim=0)

def roe_average(ql, qr):
    """
    Compute Roe average of ql and qr
    ql: double[2, ...]
    qr: double[2, ...]
    return: hm, um, vm: double[...]
    """
    sqrt_rhol = torch.sqrt(ql[0,...])
    sqrt_rhor = torch.sqrt(qr[0,...])
    vl = ql[1,...] / ql[0,...]
    vr = qr[1,...] / qr[0,...]
    vm = (sqrt_rhol * vl + sqrt_rhor * vr) / (sqrt_rhol + sqrt_rhor)
    pl = (ql[2,...] - 0.5 * ql[1,...] * vl) * (gamma - 1)
    pr = (qr[2,...] - 0.5 * qr[1,...] * vr) * (gamma - 1)
    hl = (ql[2,...] + pl) / ql[0,...]
    hr = (qr[2,...] + pr) / qr[0,...]
    hm = (sqrt_rhol * hl + sqrt_rhor * hr) / (sqrt_rhol + sqrt_rhor)
    return vm, hm

def flux_roe(ql, qr):
    """
    Roe's numerical flux for euler equation
    ql: double[3,...]
    qr: double[3,...]
    """
    
    fql = f(ql)
    fqr = f(qr)

    vm, hm = roe_average(ql, qr)
    cm = torch.sqrt((gamma - 1) * (hm - 0.5 * vm * vm))

    dq = qr - ql
    eigval_abs = torch.stack((vm - cm, vm, vm + cm), dim=0)
    eigval_abs = torch.abs(eigval_abs)

    eigvecs1 = torch.ones_like(dq)
    eigvecs2 = torch.stack((vm - cm, vm, vm + cm), dim=0)
    eigvecs3 = torch.stack((hm - vm*cm, 0.5*vm*vm, hm + vm*cm), dim=0)
    eigvecs = torch.stack((eigvecs1, eigvecs2, eigvecs3),dim=0)

    diffu = torch.zeros_like(ql)
    for i in range(eigvecs.shape[-1]):
        diffu[:, i] = 0.5 * eigvecs[..., i] @ (eigval_abs[:, i] * (torch.linalg.inv(eigvecs[..., i]) @ dq[:, i]))
    return 0.5 * (fql + fqr) - diffu

# def flux_lf(ul, ur, dt):
#     # Lax & Friedrich's flux scheme
#     return 0.5 * ((f(ul) + f(ur)) - dx / dt * (ur - ul))

def flux_rusanov(ql, qr):
    # Rusanov's flux scheme
    diffu = 0.5 * torch.maximum(max_eigenvalue(ql), max_eigenvalue(qr)) * (qr - ql)
    return 0.5 * (f(ql) + f(qr)) - diffu

def flux_hll(ql, qr):
    """
    HLL's numerical flux for Euler equation
    ql: double[3,...]
    qr: double[3,...]
    """
    # HLL's flux scheme
    fql = f(ql)
    fqr = f(qr)

    eigvalsl = eigenvalues(ql)
    eigvalsr = eigenvalues(qr)
    eigval = torch.cat((eigvalsl, 
                          eigvalsr), dim=0)
    sl = torch.min(eigval, dim=0, keepdim=False, out=None)[0]
    sr = torch.max(eigval, dim=0, keepdim=False, out=None)[0]

    tmp = torch.where(sl>=0.0, fql, 0.0)
    tmp = torch.where(sr<=0.0, fqr, tmp)
    out = torch.where(tmp==0.0, (sl*sr*(qr - ql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    return out

def flux_hlle(ql, qr):
    """
    HLLE's numerical flux for euler equation
    ql: double[3,...]
    qr: double[3,...]
    """
    # HLLE flux scheme
    # https://www.sciencedirect.com/science/article/pii/S0898122199002965
    # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics

    vm, hm = roe_average(ql, qr)
    cm = torch.sqrt((gamma-1)*(hm - 0.5 * vm * vm))

    fql = f(ql)
    fqr = f(qr)

    vl = ql[1,...] / ql[0,...]
    vr = qr[1,...] / qr[0,...]
    pl = (ql[2,...] - 0.5 * ql[1,...] * vl) * (gamma -1)
    pr = (qr[2,...] - 0.5 * qr[1,...] * vr) * (gamma -1)
    cl = torch.sqrt(gamma * pl) / ql[0,...]
    cr = torch.sqrt(gamma * pr) / qr[0,...]

    sl = torch.minimum(vm - cm, vl - cl)
    sr = torch.maximum(vm + cm, vr + cr)
    
    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    out = torch.where(tmp==0.0, (sl*sr*(qr - ql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    return out

def flux_hllc(ql, qr):
    """
    HLLC's numerical flux for shallow water equation
    ql: double[3,...]
    qr: double[3,...]
    point1: double[2,...]
    point2: double[2,...]
    """
    # HLLC flux scheme
    # https://link.springer.com/content/pdf/10.1007%2F978-3-662-03490-3_10.pdf
    # p9 p301

    # HLLE speeds
    fql = f(ql)
    fqr = f(qr)

    vm, hm = roe_average(ql, qr)
    cm = torch.sqrt((gamma-1)*(hm - 0.5 * vm * vm))

    vl = ql[1,...] / ql[0,...]
    vr = qr[1,...] / qr[0,...]
    pl = (ql[2,...] - 0.5 * ql[1,...] * vl) * (gamma -1)
    pr = (qr[2,...] - 0.5 * qr[1,...] * vr) * (gamma -1)
    cl = torch.sqrt(gamma * pl) / ql[0,...]
    cr = torch.sqrt(gamma * pr) / qr[0,...]

    sl = torch.minimum(vm - cm, vl - cl)
    sr = torch.maximum(vm + cm, vr + cr)

    # middle state estimate
    rhol = ql[0, ...]
    rhor = qr[0, ...]
    vm = (qr[1,...]*(sr - vr) - ql[1,...]*(sl - vl) - (rhor - rhol)) / (rhor*(sr - vr) - rhol*(sl - vl))
    pl = (ql[2,...] - 0.5 * ql[1,...] * vl) * (gamma - 1)
    pm = pl + rhol * (vl - vm)*(vl - sl)
    rhoml = (sl - vl)/(sl - vm)*rhol
    rhomr = (sr - vr)/(sr - vm)*rhor

    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    q_cons1 = torch.where(vm<0.0, 0.0, torch.stack((rhoml, rhoml*vm, pm/(gamma - 1) + 1/2 * rhoml*vm*vm), dim=0))
    tmp = torch.where(vm<0.0, tmp, (fql + sl*(q_cons1 - ql)))
    q_cons2 = torch.where(tmp==0.0, torch.stack((rhomr, rhomr*vm, pm/(gamma - 1) + 1/2 * rhomr*vm*vm), dim=0), 0.0)
    out = torch.where(tmp==0.0, (fqr + sr*(q_cons2 - qr)) , tmp)
    return out

def initial_condition_sod(N):
    q0 = torch.zeros(3, N, dtype=torch.double)
    q0[0, 0:N // 2] = 1.0
    q0[0, N // 2:N] = 0.125
    q0[2, 0:N // 2] = 1  / (gamma-1)
    q0[2, N // 2:N] = 0.1/ (gamma-1)
    return q0

def fvm_1storder(q0, dx, dt, nhalo):
    qc = q0[:, nhalo:-nhalo]
    ql = q0[:, nhalo-1:-nhalo-1]
    qr = q0[:, nhalo+1:-nhalo+1]
    flux_l = flux(ql, qc)
    flux_r = flux(qc, qr)
    flux_sum = - flux_l + flux_r
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

def fvm_2ndorder_space(q0, dx, dt, nhalo):
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

# reflect boundary 
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
    q = F.pad(q, (nhalo, nhalo), "circular").squeeze(0)
    return q

def calc_dt(q0, nhalo, dx):
    maxeig = torch.max(max_eigenvalue(q0[:, nhalo:-nhalo]))
    return cfl_number * dx / maxeig

def integrate(q_init, q_out, cfl_number, dx, nhalo, T, save_ts, cal_dt=False):
    nT, _, N = q_out.shape
    assert nT == len(save_ts)
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
        if cal_dt:
            dt_ = calc_dt(q0, nhalo, dx)
        else:
            dt_ = 1e-3*(128/N)
        dt = float(min(dt_, float(save_ts[save_it]) - T))
        q0 = fvm(q0, dx, dt, nhalo)
        it += 1
        T += dt
        if it % 100 == 0:
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", torch.sum(q0[2,...]).item())

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=128, type=int)
    parser.add_argument("--order", default="2", type=int)
    parser.add_argument("--period", default=1.0, type=float)
    parser.add_argument("--flux", default="roe", type=str)
    parser.add_argument("--limiter", default="minmod", type=str)
    parser.add_argument("--boundary", default="periodic", type=str)
    parser.add_argument("--time_int", default="TVD_RK", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_gif", action="store_true")
    args = parser.parse_args()
    
    N = args.resolution_x
    Te = args.period

    flux = globals()["flux_" + args.flux]
    limiter = globals()["limiter_" + args.limiter]
    boundary_condition = globals()["boundary_condition_" + args.boundary]
    fvm = globals()["fvm_" + args.time_int]
    initial_condition = initial_condition_sod

    if args.order == 1:
        fvm = fvm_1storder

    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    x_range = (0,1)
    q_init = initial_condition(N)
    dx = (x_range[1]-x_range[0]) / N
    nhalo = 3

    path = f"{os.path.dirname(os.path.realpath(__file__))}/../data/torch/{args.order}order/{Te}/"
    if not os.path.exists(path):
        os.makedirs(path)

    save_ts = torch.arange(0, Te+0.001, Te/100.)
    q_out = torch.zeros(len(save_ts), 3, N, device=device, dtype=torch.double)
    integrate(q_init, q_out, cfl_number, dx, nhalo, Te, save_ts)

    if args.plot:
        plt.figure()
        plt.imshow(q_out[:,0,:], origin="lower", cmap = "viridis", extent=[0., 1., 0., Te])
        plt.colorbar()
        plt.show()
        plt.savefig(f"{path}/Euler1d-{N}-{args.flux}")

    if args.plot_gif:
        filenames = []
        x = torch.linspace(x_range[0], x_range[1], N, dtype=torch.double)
        for step in range(q_out.shape[0]):
            plt.plot(x, q_out[step, 0, :])
            plt.savefig(f"{path}/{step}.png")
            filenames.append(f"{path}/{step}.png")
            plt.close()

        with imageio.get_writer(f"{path}/Euler1d-{N}.gif", mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)