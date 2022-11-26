# Numerical solver for 2D shallow water equation on rectangular mesh under Torch framework
import torch
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt

g = 9.8
cfl_number = 0.3

def f(q):
    """
    Flux function for 2D shallow water equation on the direction of x-axis
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

def max_eigenvalue(q, all_direction = False):
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
    if all_direction:
        res = torch.max(res, torch.sqrt(torch.abs(g*h)) + torch.abs(v))
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
    um = (sqrthl*ul + sqrthl*ur)/(sqrthl + sqrthr)
    vm = (sqrthl*vl + sqrthl*vr)/(sqrthl + sqrthr)

    return hm, um, vm

def rotate(normal_vec, q, direction = 1.0):
    Tu = torch.stack((q[0, ...],
                      normal_vec[0, ...]*q[1, ...] + normal_vec[1, ...]*q[2, ...]*direction,
                      -normal_vec[1, ...]*q[1, ...]*direction + normal_vec[0, ...]*q[2, ...]),dim=0)
    return Tu

def flux_rusanov(ql, qr, point1, point2):
    """
    Rusanov flux scheme
    out: double[3,...], accumulate result with the out array
    ql: double[3,...], left state
    qr: double[3,...], right state
    point1: double[2,...]
    point2: double[2,...]
    """
    edge_vector = point2 - point1
    normal_vector = torch.stack((-edge_vector[1, ...],
                                  edge_vector[0, ...]), dim=0)
    edge_length = torch.sqrt(edge_vector[0, ...]*edge_vector[0, ...] + edge_vector[1, ...]*edge_vector[1, ...])
    n = normal_vector / edge_length #normalized
    Tql = rotate(n, ql, direction=1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction=1.0)#Tn @ qr
    diffu = torch.maximum(max_eigenvalue(Tql, all_direction = False), max_eigenvalue(Tqr, all_direction = False)) * (Tqr - Tql)
    fql = f(Tql)
    fqr = f(Tqr)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffu)
    out = rotate(n, tmp, direction=-1.0)
    return out

def flux_roe(ql, qr, point1, point2):
    """
    Roe's numerical flux for shallow water equation
    ql: double[3,...]
    qr: double[3,...]
    point1: double[2,...]
    point2: double[2,...]
    """
    edge_vector = point2 - point1
    normal_vector = torch.stack((-edge_vector[1, ...],
                                  edge_vector[0, ...]), dim=0)
    edge_length = torch.sqrt(edge_vector[0, ...]*edge_vector[0, ...] + edge_vector[1, ...]*edge_vector[1, ...])
    n = normal_vector / edge_length #normalized
    Tql = rotate(n, ql, direction=1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction=1.0)#Tn @ qr
    
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
    out = rotate(n, tmp, direction=-1.0)
    return out

def flux_hll(ql, qr, point1, point2):
    """
    HLL's numerical flux for shallow water equation
    ql: double[3,...]
    qr: double[3,...]
    point1: double[2,...]
    point2: double[2,...]
    """
    # HLL's flux scheme
    edge_vector = point2 - point1
    normal_vector = torch.stack((-edge_vector[1, ...],
                                  edge_vector[0, ...]), dim=0)
    edge_length = torch.sqrt(edge_vector[0, ...]*edge_vector[0, ...] + edge_vector[1, ...]*edge_vector[1, ...])
    n = normal_vector / edge_length #normalized
    Tql = rotate(n, ql, direction=1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction=1.0)#Tn @ qr

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

    tmp = torch.where(sl<0.0, 0.0, fql)
    tmp = torch.where(sr>0.0, tmp, fqr)
    tmp = torch.where(tmp==0.0, (sl*sr*(Tqr - Tql) + fql*sr - fqr*sl)/(sr-sl) , tmp)
    out = rotate(n, tmp*edge_length, direction=-1.0)
    return out

def flux_hlle(ql, qr, point1, point2):
    """
    HLLE's numerical flux for shallow water equation
    ql: double[3,...]
    qr: double[3,...]
    point1: double[2,...]
    point2: double[2,...]
    """
    # HLLE flux scheme
    # https://www.sciencedirect.com/science/article/pii/S0898122199002965
    # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics

    edge_vector = point2 - point1
    normal_vector = torch.stack((-edge_vector[1, ...],
                                  edge_vector[0, ...]), dim=0)
    edge_length = torch.sqrt(edge_vector[0, ...]*edge_vector[0, ...] + edge_vector[1, ...]*edge_vector[1, ...])
    n = normal_vector / edge_length #normalized
    Tql = rotate(n, ql, direction=1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction=1.0)#Tn @ qr

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
    out = rotate(n, tmp*edge_length, direction=-1.0)
    return out

def flux_hllc(ql, qr, point1, point2):
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
    edge_vector = point2 - point1
    normal_vector = torch.stack((-edge_vector[1, ...],
                                  edge_vector[0, ...]), dim=0)
    edge_length = torch.sqrt(edge_vector[0, ...]*edge_vector[0, ...] + edge_vector[1, ...]*edge_vector[1, ...])
    n = normal_vector / edge_length #normalized
    Tql = rotate(n, ql, direction=1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction=1.0)#Tn @ qr

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
    out = rotate(n, tmp*edge_length, direction=-1.0)
    return out

def fvm_1storder(q0, vert, cell_area, dt, nhalo):
    """
    Finite volume method for shallow water equation
    q0: double[3, M+nhalo*2, N+nhalo*2]
    vert: double[2, M+1, N+1]
    cell_area: double[M, N]
    dt: double
    nhalo: int
    return: double[3, M+nhalo*2, N+nhalo*2]
    """
    assert nhalo > 0 and nhalo - 1 > 0

    vertbl = vert[:, :-1, :-1]
    vertul = vert[:, :-1, 1:]
    vertur = vert[:, 1:, 1:]
    vertbr = vert[:, 1:, :-1]

    qc = q0[:, nhalo:-nhalo, nhalo:-nhalo]
    ql = q0[:, nhalo-1:-nhalo-1, nhalo:-nhalo]
    qu = q0[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qr = q0[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qb = q0[:, nhalo:-nhalo, nhalo-1:-nhalo-1]

    flux_l = flux(qc, ql, vertbl, vertul)
    flux_u = flux(qc, qu, vertul, vertur)
    flux_r = flux(qc, qr, vertur, vertbr)
    flux_b = flux(qc, qb, vertbr, vertbl)

    flux_sum = flux_l + flux_u + flux_r + flux_b

    q1c = qc - dt/cell_area * flux_sum

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

def fvm_2ndorder_space(q0, vert, cell_area, dt, nhalo):
    """
    Finite volume method for shallow water equation
    q0: double[3, M+nhalo*2, N+nhalo*2]
    vert: double[2, M+1, N+1]
    cell_area: double[M, N]
    dt: double
    nhalo: int
    return: double[3, M+nhalo*2, N+nhalo*2]
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

    vertbl = vert[:, :-1, :-1]
    vertul = vert[:, :-1, 1:]
    vertur = vert[:, 1:, 1:]
    vertbr = vert[:, 1:, :-1]
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
    qlp = q0r[:, nhalo-1:-nhalo-1, nhalo:-nhalo]
    qlm = q0l[:, nhalo:-nhalo, nhalo:-nhalo]

    qrp = q0l[:, nhalo+1:-nhalo+1, nhalo:-nhalo]
    qrm = q0r[:, nhalo:-nhalo, nhalo:-nhalo]

    qbp = q0u[:, nhalo:-nhalo, nhalo-1:-nhalo-1]
    qbm = q0b[:, nhalo:-nhalo, nhalo:-nhalo]
    
    qup = q0b[:, nhalo:-nhalo, nhalo+1:-nhalo+1]
    qum = q0u[:, nhalo:-nhalo, nhalo:-nhalo]

    flux_l = flux(qlm, qlp, vertbl, vertul)
    flux_u = flux(qum, qup, vertul, vertur)
    flux_r = flux(qrm, qrp, vertur, vertbr)
    flux_b = flux(qbm, qbp, vertbr, vertbl)

    flux_sum = flux_l + flux_u + flux_r + flux_b

    q1c = qc + -dt/cell_area * flux_sum

    q1 = boundary_condition(q1c, nhalo)
    return q1

def fvm_heun(q0, vert, cell_area, dt, nhalo):
    """
    q0: double[3, M+nhalo*2,N+nhalo*2], state matrix of current timestep
    q1: double[3, M+nhalo*2,N+nhalo*2], state matrix of next timestep
    vert: double[2, M+1, N+1], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    dt: double
    nhalo: int
    return: double[3, M+nhalo*2, N+nhalo*2]
    """
    qs = fvm_2ndorder_space(q0, vert, cell_area, dt, nhalo)
    qss = fvm_2ndorder_space(qs, vert, cell_area, dt, nhalo)
    q1 = 0.5 * (q0 + qss)
    return q1

def fvm_TVD_RK(q0, vert, cell_area, dt, nhalo):
    """
    q0: double[3, M+nhalo*2,N+nhalo*2], state matrix of current timestep
    q1: double[3, M+nhalo*2,N+nhalo*2], state matrix of next timestep
    vert: double[2, M+1, N+1], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    dt: double
    nhalo: int
    return: double[3, M+nhalo*2, N+nhalo*2]
    """
    #stage1
    qs = fvm_2ndorder_space(q0, vert, cell_area, dt, nhalo)
    #stage2
    qss = (3/4)*q0 + (1/4)*fvm_2ndorder_space(qs, vert, cell_area, dt, nhalo)
    #stage3
    q1 = (1/3)*q0 + (2/3)*fvm_2ndorder_space(qss, vert, cell_area, dt, nhalo)
    # print(torch.all(q1[0,:,:]>0))
    return q1

def initial_condition(x_range, y_range, M, N):
    hin, hout = 1.0, 0.5
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = torch.linspace(xmin+dx/2, xmax-dx/2, M, dtype=torch.double)
    y = torch.linspace(ymin+dy/2, ymax-dy/2, N, dtype=torch.double)
    xv, yv = torch.meshgrid(x, y, indexing = 'ij')
    sigma = 0.5
    zz = hin*torch.exp((-(xv - 0.5)**2 - (yv - 0.5)**2)/sigma**2) + hout
    u_init = torch.zeros(3, M, N, dtype=torch.double)
    u_init[0, ...] = zz
    return u_init

def boundary_condition_reflect(q, nhalo, unsqueeze=True):
    """
    Periodic boundary condition
    q: double[3, M+nhalo*2, N+nhalo*2], state matrix with halo of width nhalo
    """
    if unsqueeze:
        q = q.unsqueeze(0)
    q = F.pad(q, (nhalo, nhalo, nhalo, nhalo), "replicate").squeeze(0)
    q[1:, 0:nhalo, :] *= -1.
    q[1:, -nhalo:, :] *= -1.
    q[1:, :, 0:nhalo] *= -1.
    q[1:, :, -nhalo:] *= -1.
    return q

def boundary_condition_periodic(q, nhalo, unsqueeze=True):
    """
    Periodic boundary condition
    q: double[3, M+nhalo*2, N+nhalo*2], state matrix with halo of width nhalo
    """
    if unsqueeze:
        q = q.unsqueeze(0)
    q = F.pad(q, (nhalo, nhalo, nhalo, nhalo), "circular").squeeze()
    return q

def calc_dt(q, vert, cell_area, nhalo):
    """
    Calculate dt according to CFL condition
    q: double[3, M+2, N+2], state matrix
    vert: double[2, M+1, N+1], vertices matrix
    min_dist: global minimum of the inner circle radius of all the quad cells
    """ 
    max_eig = max_eigenvalue(q[:, nhalo:-nhalo, nhalo:-nhalo], all_direction = True)
    min_dt = cfl_number * torch.min(torch.sqrt(cell_area)) /torch.max(max_eig)
    return min_dt

def integrate(q_out, q_init, vert, cell_area, Te, save_ts, nhalo, cal_dt=False):
    """
    Solve shallow water equation on Planar
    q_out: double[nT, 3, M, N]
    q_init: double[3, M, N], initial condition
    vert: double[2, M+1, N+1], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    save_ts: double[nT], array of timesteps that need to be saved
    """
    nT, _, M, N = q_out.shape
    assert nT == len(save_ts)
    q0 = boundary_condition(q_init, nhalo)

    T = 0.0
    it = 0
    save_it = 0

    while save_it < nT:
        if save_ts[save_it] - T < 1e-10:
            q_out[save_it, ...] = q0[:, nhalo:-nhalo, nhalo:-nhalo]
            save_it += 1
            if save_it == nT:
                break
        if cal_dt:
            dt_ = calc_dt(q0, vert, cell_area, nhalo).item()
        else:
            dt_ = 5e-4*(128/max(N, M))
        dt = min(dt_, float(save_ts[save_it]) - T)
        q0 = fvm(q0, vert, cell_area, dt, nhalo)
        it += 1
        T += dt
        if it % 100 == 0:
            potential = 0.5 * torch.square(q0[0, nhalo:-nhalo, nhalo:-nhalo]) * g
            kinetic = 0.5 * (torch.square(q0[1, nhalo:-nhalo, nhalo:-nhalo] + torch.square(q0[2, nhalo:-nhalo, nhalo:-nhalo]))) / q0[0, nhalo:-nhalo, nhalo:-nhalo]
            total_energy = ((potential + kinetic) * cell_area).sum(dim=(-2, -1)).item()
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", total_energy)

def integrate_end(q_out, q_init, vert, cell_area, Te, nhalo, cal_dt=False):
    """
    Solve shallow water equation on planar
    q_out: double[3, M, N]
    q_init: double[3, M, N], initial condition
    vert: double[2, M+1, N+1], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    _, M, N = q_out.shape
    q0 = boundary_condition(q_init, nhalo)

    T = 0.0
    it = 0

    while T < Te:
        if cal_dt:
            dt_ = calc_dt(q0, vert, cell_area, nhalo).item()
        else:
            dt_ = 5e-4*(128/max(N, M))
        dt = min(dt_, Te - T)
        q0 = fvm(q0, vert, cell_area, dt, nhalo)
        it += 1
        T += dt
        if it % 100 == 0:
            potential = 0.5 * torch.square(q0[0, nhalo:-nhalo, nhalo:-nhalo]) * g
            kinetic = 0.5 * (torch.square(q0[1, nhalo:-nhalo, nhalo:-nhalo] + torch.square(q0[2, nhalo:-nhalo, nhalo:-nhalo]))) / q0[0, nhalo:-nhalo, nhalo:-nhalo]
            total_energy = ((potential + kinetic) * cell_area).sum(dim=(-2, -1)).item()
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", total_energy)
    q_out[...] = q0[:, nhalo:-nhalo, nhalo:-nhalo]

if __name__=="__main__":
    from argparse import ArgumentParser
    import MeshGrid

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=128, type=int)
    parser.add_argument("--resolution_y", default=128, type=int)
    parser.add_argument("--order", default="2", type=int)
    parser.add_argument("--period", default=1.5, type=float)
    parser.add_argument("--flux", default="roe", type=str)
    parser.add_argument("--limiter", default="minmod", type=str)
    parser.add_argument("--boundary", default="periodic", type=str)
    parser.add_argument("--time_int", default="TVD_RK", type=str)
    parser.add_argument("--save_xdmf", action="store_true")
    parser.add_argument("--save_vtk", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    M = args.resolution_x
    N = args.resolution_y
    Te = args.period
    flux = globals()["flux_" + args.flux]
    limiter = globals()["limiter_" + args.limiter]
    boundary_condition = globals()["boundary_condition_" + args.boundary]
    fvm = globals()["fvm_" + args.time_int]

    if args.order == 1:
        fvm = fvm_1storder

    use_gpu = torch.cuda.is_available()
    print("GPU is avalilable:", use_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
    print(device)

    x_range = (0., 1.)
    y_range = (0., 1.)
    q_init = initial_condition(x_range, y_range, M, N)
    vert, cell_area = MeshGrid.regular_grid(x_range, y_range, M, N)
    q_init = q_init.to(device)
    vert = vert.to(device)
    cell_area = cell_area.to(device)

    nhalo = 3
    print(f"quad mesh grid {M}-{N} successfully generate!")

    path = f"{os.path.dirname(os.path.realpath(__file__))}/../data/torch/{args.order}order/{Te}/"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.save_xdmf:
        save_ts = torch.arange(0, args.period+0.01, args.period/10., device=device, dtype=torch.double)
        q_out = torch.zeros((len(save_ts), 3, M, N), device=device, requires_grad=False, dtype=torch.double)

        print("Start computing------------------------")
        start_time = time.time()
        integrate(q_out, q_init, vert, cell_area, Te, save_ts, nhalo)
        end_time = time.time()
        print(f"End computing-------------------------\nComputing time:",end_time - start_time,"s")
        save_ts = save_ts.cpu().numpy()
        q_out = q_out.cpu().numpy()
        vert = vert.cpu()
        MeshGrid.save_quad_grid_xdmf(f"{path}/quad-SWE-{M}-{N}-{args.flux}.xdmf", zip(save_ts, q_out), vert)

    else:
        q_out = torch.zeros_like(q_init)
        print("Start computing------------------------")
        start_time = time.time()
        integrate_end(q_out, q_init, vert, cell_area, Te, nhalo)
        end_time = time.time()
        print(f"End computing-------------------------\nComputing time:",end_time - start_time,"s")
        q_out = q_out.cpu()
        vert = vert.cpu()
        if args.save_vtk:
            MeshGrid.save_quad_grid_vtk(f"{path}/quad-SWE-{M}-{N}-{args.flux}", q_out, vert)

        if args.plot:
            plt.figure(figsize=(10,8))
            plt.imshow(q_out[0,...], extent=[*x_range, *y_range], cmap='viridis', origin="lower")
            plt.colorbar()
            plt.savefig(f"{path}/quad-SWE-{M}-{N}-{args.flux}.png")