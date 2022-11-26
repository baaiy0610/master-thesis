# Numerical solver for 2D shallow water equation on rectangular mesh under dace framework
import numpy as np
import dace
import os
import time
import matplotlib.pyplot as plt
from dace.sdfg.utils import load_precompiled_sdfg

g = 9.8
M = dace.symbol("M")
N = dace.symbol("N")
nT = dace.symbol("nT")

@dace.program
def f(u: dace.float64[3]) -> dace.float64[3]:
    """
    Flux function for 2D shallow water equation on the direction of x-axis
    """
    out = np.zeros((3,))
    h = u[0]
    vh = u[1]
    wh = u[2]
    v = vh/h
    w = wh/h
    out[0] = vh
    out[1] = v*v*h + 0.5*g*h*h
    out[2] = v*w*h
    return out

@dace.program
def max_eigenvalue(u: dace.float64[3], all_direction: dace.bool = False) -> dace.float64:
    """
    Maximum eigenvalue
    u: double, state vector
    """
    h = u[0]
    vh = u[1]
    wh = u[2]
    v = vh / h
    w = wh / h
    res = np.sqrt(np.abs(g*h)) + np.abs(v)
    if all_direction:
        res = max(res, np.sqrt(np.abs(g*h)) + np.abs(w))
    return res

@dace.program
def roe_average(ul: dace.float64[3], ur: dace.float64[3]) -> dace.float64[3]:
    out = np.zeros((3,))
    hl = ul[0]
    vhl = ul[1]
    whl = ul[2]

    hr = ur[0]
    vhr = ur[1]
    whr = ur[2]

    vl = vhl/hl
    wl = whl/hl

    vr = vhr/hr
    wr = whr/hr

    hm = 0.5*(hl+hr)
    vm = (np.sqrt(hl)*vl + np.sqrt(hr)*vr)/(np.sqrt(hl)+np.sqrt(hr))
    wm = (np.sqrt(hl)*wl + np.sqrt(hr)*wr)/(np.sqrt(hl)+np.sqrt(hr))

    out[0] = hm
    out[1] = vm
    out[2] = wm
    return out

@dace.program
def rotate(normal_vec: dace.float64[2], u: dace.float64[3], direction: dace.float64 = 1.0) -> dace.float64[3]:
    Tu = np.zeros((3,))
    
    Tu[0] = u[0]
    Tu[1] = normal_vec[0]*u[1] + normal_vec[1]*u[2]*direction
    Tu[2] = - normal_vec[1]*u[1]*direction + normal_vec[0]*u[2]
    return Tu

@dace.program
def flux_rusanov(ul: dace.float64[3], ur: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
    """
    Rusanov flux scheme
    out: double[3], out = out + flux, accumulate result on the out array
    ul: double[3], left state
    ur: double[3], right state
    edge_vec: double[2], vector of the clockwise edge
    """
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]
    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tul = rotate(n, ul, direction=1.0)#Tn @ ul
    Tur = rotate(n, ur, direction=1.0)#Tn @ ur
    diffu = max(max_eigenvalue(Tul, all_direction = False), max_eigenvalue(Tur, all_direction = False)) * (Tur - Tul)
    ful = f(Tul)
    fur = f(Tur)
    tmp = 0.5 * edge_length * ((ful + fur) - diffu)
    out = rotate(n, tmp, direction=-1.0)
    return out
   
@dace.program
def flux_roe(ul: dace.float64[3], ur: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]

    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tul = rotate(n, ul, direction = 1.0)#Tn @ ul
    Tur = rotate(n, ur, direction = 1.0)#Tn @ ur
    
    roe_ave = roe_average(Tul, Tur)

    hm = roe_ave[0]
    vm = roe_ave[1]
    wm = roe_ave[2]

    cm = np.sqrt(g*hm)
    eigval_abs = np.zeros((3,))
    eigval_abs[0] = abs(vm - cm)
    eigval_abs[1] = abs(vm)
    eigval_abs[2] = abs(vm + cm)

    du = Tur - Tul

    eigvecs = np.zeros((3,3))
    eigvecs[0,0] = 1.
    eigvecs[0,1] = 0.
    eigvecs[0,2] = 1.

    eigvecs[1,0] = vm - cm
    eigvecs[1,1] = 0.
    eigvecs[1,2] = vm + cm

    eigvecs[2,0] = wm
    eigvecs[2,1] = 1.
    eigvecs[2,2] = wm

    eigvecs_inv = np.zeros((3,3))
    eigvecs_inv[0,0] = (vm+cm)/(2.*cm)
    eigvecs_inv[0,1] = -0.5 * (1./cm)
    eigvecs_inv[0,2] = 0.

    eigvecs_inv[1,0] = -wm
    eigvecs_inv[1,1] = 0.
    eigvecs_inv[1,2] = 1.

    eigvecs_inv[2,0] = -(vm-cm)/(2.*cm)
    eigvecs_inv[2,1] = 0.5 * (1./cm)
    eigvecs_inv[2,2] = 0.

    diffu = eigvecs @ (eigval_abs * (eigvecs_inv @ du))
    
    ful = f(Tul)
    fur = f(Tur)
    tmp = 0.5 * edge_length * ((ful + fur) - diffu)
    out = rotate(n, tmp, direction = -1.0)
    return out

@dace.program
def flux_hll(ul: dace.float64[3], ur: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
    # HLL's flux scheme
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]

    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tul = rotate(n, ul, direction = 1.0)#Tn @ ul
    Tur = rotate(n, ur, direction = 1.0)#Tn @ ur

    ful = f(Tul)
    fur = f(Tur)

    eigvalsl = np.zeros((3,))
    eigvalsl[0] = Tul[1]/Tul[0] - np.sqrt(np.abs(g*Tul[0]))
    eigvalsl[1] = Tul[1]/Tul[0]
    eigvalsl[2] = Tul[1]/Tul[0] + np.sqrt(np.abs(g*Tul[0]))
    # eigvalsl = eigenvalues(Tul)
    # eigvalsr = eigenvalues(Tur)

    eigvalsr = np.zeros((3,))
    eigvalsr[0] = Tur[1]/Tur[0] - np.sqrt(np.abs(g*Tur[0]))
    eigvalsr[1] = Tur[1]/Tur[0]
    eigvalsr[2] = Tur[1]/Tur[0] + np.sqrt(np.abs(g*Tur[0]))

    # sl = np.min(eigvalsl)
    # sr = np.max(eigvalsr)
    sl = min(eigvalsl[0], eigvalsl[1], eigvalsl[2], eigvalsr[0], eigvalsr[1], eigvalsr[2])
    sr = max(eigvalsl[0], eigvalsl[1], eigvalsl[2], eigvalsr[0], eigvalsr[1], eigvalsr[2])

    if sl >= 0.0:
        tmp = edge_length * ful
        out = rotate(n, tmp, direction=-1.0)
        return out
    elif sr <= 0.0:
        tmp = edge_length * fur 
        out = rotate(n, tmp, direction=-1.0)
        return out
    else:
        tmp = edge_length * (sl*sr*(Tur - Tul) + ful*sr - fur*sl)/(sr-sl)
        out = rotate(n, tmp, direction=-1.0)
        return out

@dace.program
def flux_hlle(ul: dace.float64[3], ur: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
    # HLLE flux scheme
    # https://www.sciencedirect.com/science/article/pii/S0898122199002965
    # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]

    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tul = rotate(n, ul, direction = 1.0)#Tn @ ul
    Tur = rotate(n, ur, direction = 1.0)#Tn @ ur

    ful = f(Tul)
    fur = f(Tur)
    
    roe_ave = roe_average(Tul, Tur)

    hm = roe_ave[0]
    vm = roe_ave[1]
    wm = roe_ave[2]

    cm = np.sqrt(g*hm)

    vl = Tul[1] / Tul[0]
    vr = Tur[1] / Tur[0]
    cl = np.sqrt(np.abs(g*Tul[0]))
    cr = np.sqrt(np.abs(g*Tur[0]))
    sl = min(vm - cm, vl - cl)
    sr = max(vm + cm, vr + cr)
    
    if sl >= 0.0:
        tmp = ful * edge_length
        out = rotate(n, tmp, direction= -1.0)
        return out
    elif sr <= 0.0:
        tmp = fur * edge_length
        out = rotate(n, tmp, direction= -1.0) 
        return out        
    else:
        tmp = ((sl*sr*(Tur - Tul) + ful*sr - fur*sl)/(sr-sl)) * edge_length
        out = rotate(n, tmp, direction= -1.0) 
        return out

@dace.program
def flux_hllc(ul: dace.float64[3], ur: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
    # HLLC flux scheme
    # https://link.springer.com/content/pdf/10.1007%2F978-3-662-03490-3_10.pdf
    # p9 p301

    # HLLE speeds
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]

    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tul = rotate(n, ul, direction = 1.0)#Tn @ ul
    Tur = rotate(n, ur, direction = 1.0)#Tn @ ur

    ful = f(Tul)
    fur = f(Tur)

    hl = Tul[0]
    hr = Tur[0]
    
    vl = Tul[1] / hl
    vr = Tur[1] / hr
    
    wl = Tul[2] / hl
    wr = Tur[2] / hr
    
    cl = np.sqrt(np.abs(g*hl))
    cr = np.sqrt(np.abs(g*hr))
    
    vm = 0.5 * (vl+vr) + cl - cr
    # hm = ((0.5*(np.sqrt(g*hl)+np.sqrt(g*hr))+0.25*(vl-vr))**2)/g
    cm = 0.25 * (vl-vr) + 0.5 * (cl+cr)

    if hl<=0.0:
        sl = vr - 2.0 * np.sqrt(g*hr)
    else:
        sl = min(vm - cm, vl - cl)
        
    if hr<=0.0:
        sr = vl + 2.0 * np.sqrt(g*hl)
    else:
        sr = max(vm + cm, vr + cr)
    
    # middle state estimate
    hlm = (sl - vl)/(sl - vm) * hl
    hrm = (sr - vr)/(sr - vm) * hr
    if sl >= 0.0:
        tmp = ful * edge_length
        out = rotate(n, tmp, direction= -1.0)        
        return out
    elif sr <= 0.0:
        tmp = fur * edge_length
        out = rotate(n, tmp, direction= -1.0)         
        return out
    elif vm >= 0.0:
        u_cons = np.zeros((3,))
        u_cons[0] = hlm
        u_cons[1] = hlm * vm
        u_cons[2] = hlm * wl
        tmp = (ful + sl*(u_cons - Tul)) * edge_length
        out = rotate(n, tmp, direction= -1.0)         
        return out
    else: # vm <= 0
        u_cons = np.zeros((3,))
        u_cons[0] = hrm
        u_cons[1] = hrm*vm
        u_cons[2] = hrm*wr
        tmp = (fur + sr*(u_cons - Tur)) * edge_length
        out = rotate(n, tmp, direction= -1.0)     
        return out

flux = flux_roe

@dace.program
def fvm_1storder(u0: dace.float64[M+4, N+4, 3], u1: dace.float64[M+4, N+4, 3], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], dt: dace.float64):
    """
    u0: double[M+4,N+4,3], state matrix of current timestep
    u1: double[M+4,N+4,3], state matrix of next timestep
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    # internal region
    for j in range(2, N+2): # iterate over y axis
        for i in range(2, M+2): # iterate over x axis
            vertbl = vert[i-2, j-2, :]
            vertul = vert[i-2, j-1, :]
            vertur = vert[i-1, j-1, :]
            vertbr = vert[i-1, j-2, :]
            
            uu = u0[i, j+1, :]
            ub = u0[i, j-1, :]
            ul = u0[i-1, j, :]
            ur = u0[i+1, j, :]
            uc = u0[i, j, :]
            
            area = cell_area[i-2, j-2]
            flux_sum = np.zeros((3,))
            flux_sum = flux(uc, ul, vertbl, vertul) +\
                       flux(uc, uu, vertul, vertur) +\
                       flux(uc, ur, vertur, vertbr) +\
                       flux(uc, ub, vertbr, vertbl)
            u1[i, j, :] = uc - dt/area*flux_sum   
    boundary_condition(u1)

@dace.program
def minmod(a, b, c = None):
    if c is None:
        return np.sign(a)*np.minimum(np.abs(a), np.abs(b))*(a*b > 0.0)
    else:
        return np.sign(a)*np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))*(a*b > 0.0)*(a*c > 0.0)

@dace.program
def maxmod(a, b):
    return np.sign(a)*np.maximum(np.abs(a), np.abs(b))*(a*b > 0.0)

@dace.program
def limiter_minmod(fd, cd, bd):
    return minmod(fd, bd)

@dace.program
def limiter_superbee(fd, cd, bd):
    return maxmod(minmod(2*bd, fd), minmod(bd, 2*fd))

@dace.program
def limiter_mc(fd, cd, bd):
    return minmod(2*fd, cd, 2*bd)

limiter = limiter_mc

@dace.program
def fvm_2ndorder_space(u0: dace.float64[M+4, N+4, 3], u1: dace.float64[M+4, N+4, 3], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], dt: dace.float64):
    # 2nd order fvm scheme using limiters for conservative variables in space
    #compute x-direction slope
    fdx = u0[2:M+4, 1:N+4-1 ,:] - u0[1:M+4-1, 1:N+4-1, :]
    bdx = u0[1:M+4-1, 1:N+4-1, :] - u0[0:M+4-2, 1:N+4-1, :]
    cdx = 0.5 * (u0[2:M+4, 1:N+4-1 ,:] - u0[0:M+4-2, 1:N+4-1, :])
    slope_x = limiter(fdx, cdx, bdx)
    #compute y-direction slope
    fdy = u0[1:M+4-1, 2:N+4 ,:] - u0[1:M+4-1, 1:N+4-1, :]
    bdy = u0[1:M+4-1, 1:N+4-1, :] - u0[1:M+4-1, 0:N+4-2, :]
    cdy = 0.5 * (u0[1:M+4-1, 2:N+4 ,:] - u0[1:M+4-1, 0:N+4-2, :])
    slope_y = limiter(fdy, cdy, bdy)

    u0l = np.zeros_like(u0)
    u0r = np.zeros_like(u0)
    u0u = np.zeros_like(u0)
    u0b = np.zeros_like(u0)

    u0l[1:M+4-1, 1:N+4-1, :] = u0[1:M+4-1, 1:N+4-1, :] - 0.5*slope_x
    u0r[1:M+4-1, 1:N+4-1, :] = u0[1:M+4-1, 1:N+4-1, :] + 0.5*slope_x
    u0u[1:M+4-1, 1:N+4-1, :] = u0[1:M+4-1, 1:N+4-1, :] + 0.5*slope_y
    u0b[1:M+4-1, 1:N+4-1, :] = u0[1:M+4-1, 1:N+4-1, :] - 0.5*slope_y

    # cell_center_3d = cell_add_halo[2:-2, 2:-2, :]
    for i in range(2, M+2):
        for j in range(2, N+2):
            vertbl = vert[i-2, j-2, :]
            vertul = vert[i-2, j-1, :]
            vertur = vert[i-1, j-1, :]
            vertbr = vert[i-1, j-2, :]

            uc = u0[i, j, :]

            ulp = u0r[i-1, j, :]
            ulm = u0l[i, j, :]

            urp = u0l[i+1, j, :]
            urm = u0r[i, j, :]

            ubp = u0u[i, j-1, :]
            ubm = u0b[i, j, :]
            
            uup = u0b[i, j+1, :]
            uum = u0u[i, j, :]

            area = cell_area[i-2, j-2]

            #compute flux term
            flux_sum = flux(ulm, ulp, vertbl, vertul) +\
                       flux(uum, uup, vertul, vertur) +\
                       flux(urm, urp, vertur, vertbr) +\
                       flux(ubm, ubp, vertbr, vertbl)

            u1[i, j, :] = uc - dt/area*flux_sum   
    boundary_condition(u1)

@dace.program
def fvm_2ndorder(u0: dace.float64[M+4, N+4, 3], u1: dace.float64[M+4, N+4, 3], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], dt: dace.float64):
    """
    u0: double[M+4,N+4,3], state matrix of current timestep
    u1: double[M+4,N+4,3], state matrix of next timestep
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    s = np.zeros_like(u0)
    uss = np.zeros_like(u0)

    fvm_2ndorder_space(u0, us, vert, cell_area, dt)
    fvm_2ndorder_space(us, uss, vert, cell_area,dt)
    u1[...] = 0.5 * (u0 + uss)

def initial_condition(x_range, y_range, M, N):
    hin, hout = 1.0, 0.5
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = np.linspace(xmin+dx/2, xmax-dx/2, M)
    y = np.linspace(ymin+dy/2, ymax-dy/2, N)
    xv, yv = np.meshgrid(x,y,indexing = 'ij')
    sigma = 0.5
    zz = hin*np.exp((-(xv - 0.5)**2 - (yv - 0.5)**2)/sigma**2) + hout
    u_init = np.zeros((M, N, 3))
    u_init[..., 0] = zz
    return u_init

@dace.program
def boundary_condition(u: dace.float64[M+4, N+4, 3]):
    """
    Periodic boundary condition
    u: double[M+4, N+4, 3], state matrix with halo of width 2
    """
    # upper boundary
    u[:, M+4-1, :] = u[:, 3, :]
    u[:, M+4-2, :] = u[:, 2, :]
    # lower boundary
    u[:, 0, :] = u[:, M+4-4, :]
    u[:, 1, :] = u[:, M+4-3, :]
    # left boundary
    u[0, :, :] = u[M+4-4, :, :]
    u[1, :, :] = u[M+4-3, :, :]
    # right boundary
    u[M+4-1, :, :] = u[3, :, :]
    u[M+4-2, :, :] = u[2, :, :]

fvm = fvm_2ndorder
@dace.program(auto_optimize=True)
def integrate(u_out: dace.float64[M, N, 3], u_init: dace.float64[M, N, 3], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], Te: dace.float64):
    """
    Solve shallow water equation
    u_init: double[M, N, 3], initial condition
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    u0 = np.zeros((M+4, N+4, 3))
    u1 = np.zeros_like(u0)
    u0[2:M+2, 2:N+2, :] = u_init[...]
    boundary_condition(u0)
    T: dace.float64 = 0.0
    it = 0

    while T < Te:
        dt_ = 0.001*(128/max(N, M))
        dt = min(dt_, Te-T)
        fvm(u0, u1, vert, cell_area, dt)
        u0[...] = u1[...]
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = np.zeros((M, N, 2))
            tmp[..., 0] = 0.5 * np.square(u0[2:M+2, 2:N+2, 0]) * g
            tmp[..., 1] = 0.5 * (np.square(u0[2:M+2, 2:N+2, 1]) + np.square(u0[2:M+2, 2:N+2, 2])) / u0[2:M+2, 2:N+2, 0]
            sum_ = np.sum(tmp * cell_area[:,:, np.newaxis])
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

    u_out[...] = u0[2:M+2, 2:N+2, :]

@dace.program(auto_optimize=True)
def integrate_slow(u_out: dace.float64[nT ,M, N, 3], u_init: dace.float64[M, N, 3], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], Te: dace.float64, save_ts: dace.float64[nT]):
    """
    Solve shallow water equation
    u_init: double[M, N, 3], initial condition
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    save_ts: double[:], array of timesteps that need to be saved
    """
    u0 = np.zeros((M+4, N+4, 3))
    u1 = np.zeros_like(u0)
    u0[2:M+2, 2:N+2, :] = u_init[...]
    boundary_condition(u0)
    T: dace.float64 = 0.0
    it = 0
    save_it = 0

    while save_it < nT:
        if save_ts[save_it] - T < 1e-10:
            u_out[save_it, ...] = u0[2:M+2, 2:N+2, :]
            save_it += 1
            if save_it == nT:
                break
        dt_ = 0.001*(128/max(N, M/2))
        dt = min(dt_, save_ts[save_it] - T)
        fvm(u0, u1, vert, cell_area, dt)
        u0[...] = u1[...]
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = np.zeros((M, N, 2))
            tmp[..., 0] = 0.5 * np.square(u0[2:M+2, 2:N+2, 0]) * g
            tmp[..., 1] = 0.5 * (np.square(u0[2:M+2, 2:N+2, 1]) + np.square(u0[2:M+2, 2:N+2, 2])) / u0[2:M+2, 2:N+2, 0]
            sum_ = np.sum(tmp * cell_area[:,:, np.newaxis])
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

if __name__=="__main__":
    from argparse import ArgumentParser
    import MeshGrid

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=100, type=int)
    parser.add_argument("--resolution_y", default=100, type=int)
    parser.add_argument("--period", default=1.5, type=float)
    parser.add_argument("--save_xdmf", action="store_true")
    parser.add_argument("--save_vtk", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()

    M = args.resolution_x
    N = args.resolution_y
    Te = args.period
    x_range = (0., 1.)
    y_range = (0., 1.)
    u_init = initial_condition(x_range, y_range, M, N)
    vert, cell_area = MeshGrid.regular_grid(x_range, y_range, M, N)
    print(f"quad mesh grid {M}-{N} successfully generate!")

    path = f"{os.path.dirname(os.path.realpath(__file__))}/../data/dace/{fvm.name[4:]}/{Te}/"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.save_xdmf:
        save_ts = np.arange(0, args.period+0.01, args.period/10.)
        u_out = np.zeros((len(save_ts), M, N, 3))
        if args.use_cache:
            sdfg, _ = integrate_slow.load_sdfg("_dacegraphs/program.sdfg")
            integrate_slow_compiled = load_precompiled_sdfg(sdfg.build_folder)
        else:
            integrate_slow_compiled = integrate_slow.compile(simplify=True, save=True)
        print("Start computing------------------------")
        start_time = time.time()
        integrate_slow_compiled(u_out=u_out, u_init=u_init, vert=vert, cell_area=cell_area, Te=Te, save_ts=save_ts,  M=M, N=N, nT=len(save_ts), print=print)
        end_time = time.time()
        print(f"End computing-------------------------\nComputing time:",end_time - start_time,"s")
        MeshGrid.save_quad_grid_xdmf(f"{path}/quad-SWE-{M}-{N}-{flux.name[5:]}.xdmf", zip(save_ts, u_out), vert)

    else:
        u_out = np.zeros_like(u_init)
        if args.use_cache:
            sdfg, _ = integrate.load_sdfg("_dacegraphs/program.sdfg")
            integrate_compiled = load_precompiled_sdfg(sdfg.build_folder)
        else :
            integrate_compiled = integrate.compile(simplify=True, save=True)

        print("Start computing------------------------")
        start_time = time.time()
        integrate_compiled(u_out=u_out, u_init=u_init, vert=vert, cell_area=cell_area, Te=Te, M=M, N=N, print=print)
        end_time = time.time()
        print(f"End computing-------------------------\nComputing time:",end_time - start_time,"s")

        if args.save_vtk:
            MeshGrid.save_quad_grid_vtk(f"{path}/quad-SWE-{M}-{N}-{flux.name[5:]}", q_out, vert)

        if args.plot:
            plt.figure(figsize=(10,8))
            plt.imshow(u_out[...,0], extent=[*x_range, *y_range], cmap='viridis', origin="lower")
            plt.colorbar()
            plt.savefig(f"{path}/quad-SWE-{M}-{N}-{flux.name[5:]}.png")
