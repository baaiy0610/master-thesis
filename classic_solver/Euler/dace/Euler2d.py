# Numerical solver for 2D Euler equation on rectangular mesh under dace framework
import numpy as np
import dace
import os
import time
import matplotlib.pyplot as plt
from dace.sdfg.utils import load_precompiled_sdfg

gamma = 7/5
M = dace.symbol("M")
N = dace.symbol("N")
nT = dace.symbol("nT")

@dace.program
def f(q: dace.float64[4]) -> dace.float64[4]:
    """
    Flux function for 2D Euler equation on the direction of x-axis
    """
    out = np.zeros((4,))
    rho = q[0]
    rhou = q[1]
    rhov = q[2]
    E = q[3]
    u = rhou/rho
    v = rhov/rho
    p = (gamma-1)*(E - 0.5*rho*(u*u + v*v))
    out[0] = rhou
    out[1] = rhou*u + p
    out[2] = rhou*v
    out[3] = (E+p)*u
    return out

@dace.program
def max_eigenvalue(q: dace.float64[4], all_direction: dace.bool = False) -> dace.float64:
    """
    Maximum eigenvalue
    q: double, state vector
    """
    rho = q[0]
    rhou = q[1]
    rhov = q[2]
    E = q[3]
    u = rhou/rho
    v = rhov/rho
    p = (gamma-1)*(E - 0.5*rho*(u*u + v*v))
    c = np.sqrt(gamma*p/rho)
    res = np.abs(u) + c
    if all_direction:
        res = max(res, np.abs(v) + c)
    return res

@dace.program
def roe_average(ql: dace.float64[4], qr: dace.float64[4]) -> dace.float64[3]:
    """
    Compute Roe average of ql and qr
    ql: double[4]
    qr: double[4]
    return: hm, um, vm: double[3]
    """
    out = np.zeros((3,))
    sqrt_rhol = np.sqrt(ql[0])
    sqrt_rhor = np.sqrt(qr[0])
    ul = ql[1] / ql[0]
    ur = qr[1] / qr[0]
    um = (sqrt_rhol * ul + sqrt_rhor * ur) / (sqrt_rhol + sqrt_rhor)

    vl = ql[2] / ql[0]
    vr = qr[2] / qr[0]
    vm = (sqrt_rhol * vl + sqrt_rhor * vr) / (sqrt_rhol + sqrt_rhor)

    pl = (ql[3] - 0.5 * ql[0] * (ul * ul + vl * vl)) * (gamma - 1)
    pr = (qr[3] - 0.5 * qr[0] * (ur * ur + vr * vr)) * (gamma - 1)
    hl = (ql[3] + pl) / ql[0]
    hr = (qr[3] + pr) / qr[0]
    hm = (sqrt_rhol * hl + sqrt_rhor * hr) / (sqrt_rhol + sqrt_rhor)

    out[0] = hm
    out[1] = um
    out[2] = vm
    return out

@dace.program
def rotate(normal_vec: dace.float64[2], q: dace.float64[4], direction: dace.float64 = 1.0) -> dace.float64[4]:
    Tq = np.zeros((4,))
    
    Tq[0] = q[0]
    Tq[1] = normal_vec[0]*q[1] + normal_vec[1]*q[2]*direction
    Tq[2] = -normal_vec[1]*q[1]*direction + normal_vec[0]*q[2]
    Tq[3] = q[3]
    return Tq

@dace.program
def flux_rusanov(ql: dace.float64[4], qr: dace.float64[4], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[4]:
    """
    Rusanov flux scheme
    out: double[4], out = out + flux, accumulate result on the out array
    ql: double[4], left state
    qr: double[4], right state
    edge_vec: double[2], vector of the clockwise edge
    """
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]
    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tql = rotate(n, ql, direction=1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction=1.0)#Tn @ qr
    diffu = max(max_eigenvalue(Tql, all_direction = False), max_eigenvalue(Tqr, all_direction = False)) * (Tqr - Tql)
    fql = f(Tql)
    fqr = f(Tqr)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffu)
    out = rotate(n, tmp, direction=-1.0)
    return out
   
@dace.program
def flux_roe(ql: dace.float64[4], qr: dace.float64[4], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[4]:
    normal_vec = np.zeros((2,))

    edge_vec = point2 - point1
    normal_vec[0] = -edge_vec[1]
    normal_vec[1] = edge_vec[0]

    edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
    n = normal_vec / edge_length #normalized
    
    Tql = rotate(n, ql, direction = 1.0)#Tn @ ql
    Tqr = rotate(n, qr, direction = 1.0)#Tn @ qr
    
    roe_ave = roe_average(Tql, Tqr)

    hm = roe_ave[0]
    um = roe_ave[1]
    vm = roe_ave[2]

    cm = np.sqrt((gamma - 1) * (hm - 0.5 * (um * um + vm * vm)))

    #left state
    rhol = Tql[0]
    ul = Tql[1] / rhol
    vl = Tql[2] / rhol
    pl = (gamma-1)*(Tql[3] - 0.5*rhol*(ul*ul + vl*vl))

    #left state
    rhor = Tqr[0]
    ur = Tqr[1] / rhor
    vr = Tqr[2] / rhor
    pr = (gamma-1)*(Tqr[3] - 0.5*rhor*(ur*ur + vr*vr))

    dp = pr - pl
    drho = rhor - rhol
    du = ur - ul
    dv = vr - vl
    rhom = np.sqrt(Tqr[0]*Tql[0])

    alpha1 = (dp - cm*rhom*du)/(2*cm*cm)
    alpha2 = rhom * dv/cm
    alpha3 = drho - dp/(cm*cm)
    alpha4 = (dp + cm*rhom*du)/(2*cm*cm)

    # eigval_abs * (eigvecs_inv @ dq)
    k0 = np.abs(um - cm) * alpha1
    k1 = np.abs(um) * alpha2
    k2 = np.abs(um) * alpha3
    k3 = np.abs(um + cm) * alpha4

    d0 = k0 + 0.0 + k2 + k3
    d1 = (um-cm)*k0 + 0 + um*k2 + (um+cm)*k3
    d2 = um*k0 + cm*k1 + um*k2 + um*k3
    d3 = (hm-um*cm)*k0 + vm*cm*k1 + 0.5*(um*um + vm*vm)*k2 + (hm+um*cm)*k3

    diffq = np.zeros((4,))
    diffq[0] = d0
    diffq[1] = d1
    diffq[2] = d2
    diffq[3] = d3

    fql = f(Tql)
    fqr = f(Tqr)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffq)
    out = rotate(n, tmp, direction = -1.0)
    return out

# @dace.program
# def flux_hll(ql: dace.float64[3], qr: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
#     # HLL's flux scheme
#     normal_vec = np.zeros((2,))
#     g = 9.8

#     edge_vec = point2 - point1
#     normal_vec[0] = -edge_vec[1]
#     normal_vec[1] = edge_vec[0]

#     edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
#     n = normal_vec / edge_length #normalized
    
#     Tql = rotate(n, ql, direction = 1.0)#Tn @ ql
#     Tqr = rotate(n, qr, direction = 1.0)#Tn @ qr

#     fql = f(Tql)
#     fqr = f(Tqr)

#     eigvalsl = np.zeros((3,))
#     eigvalsl[0] = Tql[1]/Tql[0] - np.sqrt(np.abs(g*Tql[0]))
#     eigvalsl[1] = Tql[1]/Tql[0]
#     eigvalsl[2] = Tql[1]/Tql[0] + np.sqrt(np.abs(g*Tql[0]))
#     # eigvalsl = eigenvalues(Tql)
#     # eigvalsr = eigenvalues(Tqr)

#     eigvalsr = np.zeros((3,))
#     eigvalsr[0] = Tqr[1]/Tqr[0] - np.sqrt(np.abs(g*Tqr[0]))
#     eigvalsr[1] = Tqr[1]/Tqr[0]
#     eigvalsr[2] = Tqr[1]/Tqr[0] + np.sqrt(np.abs(g*Tqr[0]))

#     # sl = np.min(eigvalsl)
#     # sr = np.max(eigvalsr)
#     sl = min(eigvalsl[0], eigvalsl[1], eigvalsl[2], eigvalsr[0], eigvalsr[1], eigvalsr[2])
#     sr = max(eigvalsl[0], eigvalsl[1], eigvalsl[2], eigvalsr[0], eigvalsr[1], eigvalsr[2])

#     if sl >= 0.0:
#         tmp = edge_length * fql
#         out = rotate(n, tmp, direction=-1.0)
#         return out
#     elif sr <= 0.0:
#         tmp = edge_length * fqr 
#         out = rotate(n, tmp, direction=-1.0)
#         return out
#     else:
#         tmp = edge_length * (sl*sr*(Tqr - Tql) + fql*sr - fqr*sl)/(sr-sl)
#         out = rotate(n, tmp, direction=-1.0)
#         return out

# @dace.program
# def flux_hlle(ql: dace.float64[3], qr: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
#     # HLLE flux scheme
#     # https://www.sciencedirect.com/science/article/pii/S0898122199002965
#     # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics
#     normal_vec = np.zeros((2,))
#     g = 9.8

#     edge_vec = point2 - point1
#     normal_vec[0] = -edge_vec[1]
#     normal_vec[1] = edge_vec[0]

#     edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
#     n = normal_vec / edge_length #normalized
    
#     Tql = rotate(n, ql, direction = 1.0)#Tn @ ql
#     Tqr = rotate(n, qr, direction = 1.0)#Tn @ qr

#     fql = f(Tql)
#     fqr = f(Tqr)
    
#     roe_ave = roe_average(Tql, Tqr)

#     hm = roe_ave[0]
#     vm = roe_ave[1]
#     wm = roe_ave[2]

#     cm = np.sqrt(g*hm)

#     vl = Tql[1] / Tql[0]
#     vr = Tqr[1] / Tqr[0]
#     cl = np.sqrt(np.abs(g*Tql[0]))
#     cr = np.sqrt(np.abs(g*Tqr[0]))
#     sl = min(vm - cm, vl - cl)
#     sr = max(vm + cm, vr + cr)
    
#     if sl >= 0.0:
#         tmp = fql * edge_length
#         out = rotate(n, tmp, direction= -1.0)
#         return out
#     elif sr <= 0.0:
#         tmp = fqr * edge_length
#         out = rotate(n, tmp, direction= -1.0) 
#         return out        
#     else:
#         tmp = ((sl*sr*(Tqr - Tql) + fql*sr - fqr*sl)/(sr-sl)) * edge_length
#         out = rotate(n, tmp, direction= -1.0) 
#         return out

# @dace.program
# def flux_hllc(ql: dace.float64[3], qr: dace.float64[3], point1: dace.float64[2], point2: dace.float64[2]) -> dace.float64[3]:
#     # HLLC flux scheme
#     # https://link.springer.com/content/pdf/10.1007%2F978-3-662-03490-3_10.pdf
#     # p9 p301

#     # HLLE speeds
#     normal_vec = np.zeros((2,))
#     g = 9.8

#     edge_vec = point2 - point1
#     normal_vec[0] = -edge_vec[1]
#     normal_vec[1] = edge_vec[0]

#     edge_length = np.sqrt(edge_vec[0]*edge_vec[0] + edge_vec[1]*edge_vec[1])
#     n = normal_vec / edge_length #normalized
    
#     Tql = rotate(n, ql, direction = 1.0)#Tn @ ql
#     Tqr = rotate(n, qr, direction = 1.0)#Tn @ qr

#     fql = f(Tql)
#     fqr = f(Tqr)

#     hl = Tql[0]
#     hr = Tqr[0]
    
#     vl = Tql[1] / hl
#     vr = Tqr[1] / hr
    
#     wl = Tql[2] / hl
#     wr = Tqr[2] / hr
    
#     cl = np.sqrt(np.abs(g*hl))
#     cr = np.sqrt(np.abs(g*hr))
    
#     vm = 0.5 * (vl+vr) + cl - cr
#     # hm = ((0.5*(np.sqrt(g*hl)+np.sqrt(g*hr))+0.25*(vl-vr))**2)/g
#     cm = 0.25 * (vl-vr) + 0.5 * (cl+cr)

#     if hl<=0.0:
#         sl = vr - 2.0 * np.sqrt(g*hr)
#     else:
#         sl = min(vm - cm, vl - cl)
        
#     if hr<=0.0:
#         sr = vl + 2.0 * np.sqrt(g*hl)
#     else:
#         sr = max(vm + cm, vr + cr)
    
#     # middle state estimate
#     hlm = (sl - vl)/(sl - vm) * hl
#     hrm = (sr - vr)/(sr - vm) * hr
#     if sl >= 0.0:
#         tmp = fql * edge_length
#         out = rotate(n, tmp, direction= -1.0)        
#         return out
#     elif sr <= 0.0:
#         tmp = fqr * edge_length
#         out = rotate(n, tmp, direction= -1.0)         
#         return out
#     elif vm >= 0.0:
#         u_cons = np.zeros((3,))
#         u_cons[0] = hlm
#         u_cons[1] = hlm * vm
#         u_cons[2] = hlm * wl
#         tmp = (fql + sl*(u_cons - Tql)) * edge_length
#         out = rotate(n, tmp, direction= -1.0)         
#         return out
#     else: # vm <= 0
#         u_cons = np.zeros((3,))
#         u_cons[0] = hrm
#         u_cons[1] = hrm*vm
#         u_cons[2] = hrm*wr
#         tmp = (fqr + sr*(u_cons - Tqr)) * edge_length
#         out = rotate(n, tmp, direction= -1.0)     
#         return out

flux = flux_roe

@dace.program
def fvm_1storder(q0: dace.float64[M+4, N+4, 4], q1: dace.float64[M+4, N+4, 4], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], dt: dace.float64):
    """
    q0: double[M+4,N+4,4], state matrix of current timestep
    q1: double[M+4,N+4,4], state matrix of next timestep
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
            
            uu = q0[i, j+1, :]
            ub = q0[i, j-1, :]
            ql = q0[i-1, j, :]
            qr = q0[i+1, j, :]
            uc = q0[i, j, :]
            
            area = cell_area[i-2, j-2]
            # flux_sum = np.zeros((4,))
            flux_sum = flux(uc, ql, vertbl, vertul) +\
                       flux(uc, uu, vertul, vertur) +\
                       flux(uc, qr, vertur, vertbr) +\
                       flux(uc, ub, vertbr, vertbl)
            q1[i, j, :] = uc - dt/area*flux_sum   
    boundary_condition(q1)

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
def fvm_2ndorder_space(q0: dace.float64[M+4, N+4, 4], q1: dace.float64[M+4, N+4, 4], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], dt: dace.float64):
    # 2nd order fvm scheme using limiters for conservative variables in space
    #compute x-direction slope
    fdx = q0[2:M+4, 1:N+4-1 ,:] - q0[1:M+4-1, 1:N+4-1, :]
    bdx = q0[1:M+4-1, 1:N+4-1, :] - q0[0:M+4-2, 1:N+4-1, :]
    cdx = 0.5 * (q0[2:M+4, 1:N+4-1 ,:] - q0[0:M+4-2, 1:N+4-1, :])
    slope_x = limiter(fdx, cdx, bdx)
    #compute y-direction slope
    fdy = q0[1:M+4-1, 2:N+4 ,:] - q0[1:M+4-1, 1:N+4-1, :]
    bdy = q0[1:M+4-1, 1:N+4-1, :] - q0[1:M+4-1, 0:N+4-2, :]
    cdy = 0.5 * (q0[1:M+4-1, 2:N+4 ,:] - q0[1:M+4-1, 0:N+4-2, :])
    slope_y = limiter(fdy, cdy, bdy)

    u0l = np.zeros_like(q0)
    u0r = np.zeros_like(q0)
    u0u = np.zeros_like(q0)
    u0b = np.zeros_like(q0)

    u0l[1:M+4-1, 1:N+4-1, :] = q0[1:M+4-1, 1:N+4-1, :] - 0.5*slope_x
    u0r[1:M+4-1, 1:N+4-1, :] = q0[1:M+4-1, 1:N+4-1, :] + 0.5*slope_x
    u0u[1:M+4-1, 1:N+4-1, :] = q0[1:M+4-1, 1:N+4-1, :] + 0.5*slope_y
    u0b[1:M+4-1, 1:N+4-1, :] = q0[1:M+4-1, 1:N+4-1, :] - 0.5*slope_y

    # cell_center_3d = cell_add_halo[2:-2, 2:-2, :]
    for i in range(2, M+2):
        for j in range(2, N+2):
            vertbl = vert[i-2, j-2, :]
            vertul = vert[i-2, j-1, :]
            vertur = vert[i-1, j-1, :]
            vertbr = vert[i-1, j-2, :]

            uc = q0[i, j, :]

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

            q1[i, j, :] = uc - dt/area*flux_sum   
    boundary_condition(q1)

@dace.program
def fvm_2ndorder(q0: dace.float64[M+4, N+4, 4], q1: dace.float64[M+4, N+4, 4], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], dt: dace.float64):
    """
    q0: double[M+4,N+4,3], state matrix of current timestep
    q1: double[M+4,N+4,3], state matrix of next timestep
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    qs = np.zeros_like(q0)
    qss = np.zeros_like(q0)

    fvm_2ndorder_space(q0, qs, vert, cell_area, dt)
    fvm_2ndorder_space(qs, qss, vert, cell_area,dt)
    q1[...] = 0.5 * (q0 + qss)

def initial_condition(x_range, y_range, M, N):
    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = np.linspace(xmin+dx/2, xmax-dx/2, M)
    y = np.linspace(ymin+dy/2, ymax-dy/2, N)
    xv, yv = np.meshgrid(x,y,indexing = 'ij')

    q_init = np.zeros((M, N, 4))
    l = np.where(xv<0.5, 1, 0.0)
    r = np.where(xv>=0.5, 1, 0.0)
    t = np.where(yv>=0.5, 1, 0.0)
    b = np.where(yv<0.5, 1, 0.0)
    q_init[...,0] = 2.0*l*t + 1.0*l*b + 1.0*r*t +3.0*r*b
    q_init[...,1] = 0.75*t - 0.75*b
    q_init[...,2] = 0.5*l - 0.5*r
    q_init[...,3] = 0.5*q_init[...,0]*(np.square(q_init[...,1]) + np.square(q_init[...,2])) + 1.0 / (gamma - 1.0)
    return q_init

@dace.program
def boundary_condition(q: dace.float64[M+4, N+4, 4]):
    """
    Periodic boundary condition
    q: double[M+4, N+4, 4], state matrix with halo of width 2
    """
    # upper boundary
    q[:, M+4-1, :] = q[:, 3, :]
    q[:, M+4-2, :] = q[:, 2, :]
    # lower boundary
    q[:, 0, :] = q[:, M+4-4, :]
    q[:, 1, :] = q[:, M+4-3, :]
    # left boundary
    q[0, :, :] = q[M+4-4, :, :]
    q[1, :, :] = q[M+4-3, :, :]
    # right boundary
    q[M+4-1, :, :] = q[3, :, :]
    q[M+4-2, :, :] = q[2, :, :]

fvm = fvm_2ndorder
@dace.program(auto_optimize=True)
def integrate(q_out: dace.float64[M, N, 4], q_init: dace.float64[M, N, 4], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], Te: dace.float64):
    """
    Solve 2D Euler equation
    q_init: double[M, N, 4], initial condition
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    q0 = np.zeros((M+4, N+4, 4))
    q1 = np.zeros_like(q0)
    q0[2:M+2, 2:N+2, :] = q_init[...]
    boundary_condition(q0)
    T: dace.float64 = 0.0
    it = 0

    while T < Te:
        dt_ = 5e-4*(128/max(M,N))
        dt = min(dt_, Te-T)
        fvm(q0, q1, vert, cell_area, dt)
        q0[...] = q1[...]
        it += 1
        T += dt
        if it % 100 == 0:
            # print("T =", T, "it =", it, "dt =", dt)
            tmp = q0[2:M+2, 2:N+2, 3]
            sum_ = np.sum(tmp * cell_area[:,:])
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)
    q_out[...] = q0[2:M+2, 2:N+2, :]

@dace.program(auto_optimize=True)
def integrate_slow(q_out: dace.float64[nT ,M, N, 4], q_init: dace.float64[M, N, 4], vert: dace.float64[M+1, N+1, 2], cell_area: dace.float64[M, N], Te: dace.float64, save_ts: dace.float64[nT]):
    """
    Solve 2D Euler equation
    q_init: double[M, N, 4], initial condition
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    save_ts: double[:], array of timesteps that need to be saved
    """
    q0 = np.zeros((M+4, N+4, 4))
    q1 = np.zeros_like(q0)
    q0[2:M+2, 2:N+2, :] = q_init[...]
    boundary_condition(q0)
    T: dace.float64 = 0.0
    it = 0
    save_it = 0

    while save_it < nT:
        if save_ts[save_it] - T < 1e-10:
            q_out[save_it, ...] = q0[2:M+2, 2:N+2, :]
            save_it += 1
            if save_it == nT:
                break
        dt_ = 5e-4*(128/max(M,N))
        dt = min(dt_, save_ts[save_it] - T)
        fvm(q0, q1, vert, cell_area, dt)
        q0[...] = q1[...]
        it += 1
        T += dt
        if it % 100 == 0:
            # print("T =", T, "it =", it, "dt =", dt)
            tmp = q0[2:M+2, 2:N+2, 3]
            sum_ = np.sum(tmp * cell_area[:,:, np.newaxis])
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

if __name__=="__main__":
    from argparse import ArgumentParser
    import MeshGrid

    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=100, type=int)
    parser.add_argument("--resolution_y", default=100, type=int)
    parser.add_argument("--period", default=0.3, type=float)
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
    q_init = initial_condition(x_range, y_range, M, N)
    q_out = np.zeros_like(q_init)
    vert, cell_area = MeshGrid.regular_grid(x_range, y_range, M, N)
    print(f"quad mesh grid {M}-{N} successfully generate!")

    path = f"{os.path.dirname(os.path.realpath(__file__))}/../data/dace/{Te}"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.save_xdmf:
        save_ts = np.arange(0, args.period+0.01, args.period/10.)
        q_out = np.zeros((len(save_ts), M, N, 4))
        if args.use_cache:
            sdfg, _ = integrate_slow.load_sdfg("_dacegraphs/program.sdfg")
            integrate_slow_compiled = load_precompiled_sdfg(sdfg.build_folder)
        else:
            integrate_slow_compiled = integrate_slow.compile(simplify=True, save=True)
        print("Start computing------------------------")
        start_time = time.time()
        integrate_slow_compiled(q_out=q_out, q_init=q_init, vert=vert, cell_area=cell_area, Te=Te, save_ts=save_ts,  M=M, N=N, nT=len(save_ts), print=print)
        end_time = time.time()
        print(f"End computing-------------------------\nComputing time:",end_time - start_time,"s")
        MeshGrid.save_quad_grid_xdmf(f"{path}/quad-euler-{M}-{N}.xdmf", zip(save_ts, q_out), vert)

    else:
        if args.use_cache:
            sdfg, _ = integrate.load_sdfg("_dacegraphs/program.sdfg")
            integrate_compiled = load_precompiled_sdfg(sdfg.build_folder)
        else :
            integrate_compiled = integrate.compile(simplify=True, save=True)

        print("Start computing------------------------")
        start_time = time.time()
        integrate_compiled(q_out=q_out, q_init=q_init, vert=vert, cell_area=cell_area, Te=Te, M=M, N=N, print=print)
        end_time = time.time()
        print(f"End computing-------------------------\nComputing time:",end_time - start_time,"s")

        if args.save_vtk:
            MeshGrid.save_quad_grid_vtk(f"{path}/quad-euler-{M}-{N}", q_out, vert)

        if args.plot:
            plt.figure(figsize=(10,8))
            plt.imshow(q_out[...,0], extent=[*x_range, *y_range], cmap='viridis', origin="lower")
            plt.colorbar()
            plt.savefig(f"{path}/quad-euler-{M}-{N}.png")
