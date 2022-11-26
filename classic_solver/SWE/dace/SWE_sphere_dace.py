# Numerical solver for shallow water equation on logically sphere mesh under Dace framework
import numpy as np
import dace
import time
import os
from dace.sdfg.utils import load_precompiled_sdfg
import matplotlib.pyplot as plt
# dace.config.Config.append('compiler', 'use_cache', value=True)

M = dace.symbol('M')
N = dace.symbol('N')
nT = dace.symbol("nT")

g = 11489.57219

@dace.program
def f(q:dace.float64[3]) -> dace.float64[3]:
    """
    Flux function for 2D shallow water equation on the direction of x-axis
    """
    out = np.zeros((3,))
    h = q[0]
    uh = q[1]
    vh = q[2]
    u = uh/h
    v = vh/h
    out[0] = uh
    out[1] = u*uh + 0.5*g*h*h
    out[2] = u*vh
    return out

@dace.program
def fn(q:dace.float64[4], n:dace.float64[3]) -> dace.float64[4]:
    out = np.zeros((4,))
    h = q[0]
    uh = q[1]
    vh = q[2]
    wh = q[3]
    u = uh/h
    v = vh/h
    w = wh/h
    out[0] = uh*n[0] + vh*n[1] + wh*n[2]
    out[1] = (uh*u + 0.5*g*h*h)*n[0] + uh*v*n[1] + uh*w*n[2]
    out[2] = uh*v*n[0] + (vh*v + 0.5*g*h*h)*n[1] + vh*w*n[2]
    out[3] = uh*w*n[0] + vh*w*n[1] + (wh*w+0.5*g*h*h)*n[2]
    return out

@dace.program
def d_sphere_3d(point1_3d:dace.float64[3], point2_3d:dace.float64[3], r)->dace.float64:
    delta = point1_3d - point2_3d
    distance = np.sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]) / r
    delta_sigma = 2*np.arcsin(distance/2.0)
    return r * delta_sigma

@dace.program
def max_eigenvalue(q: dace.float64[3]) -> dace.float64:
    """
    Maximum eigenvalue
    q: double, state vector
    """
    h = q[0]
    vh = q[1]
    wh = q[2]
    v = vh / h
    w = wh / h
    res = np.sqrt(np.abs(g*h)) + np.abs(v)
    return res

@dace.program
def roe_average(ql:dace.float64[3], qr:dace.float64[3]):
    out = np.zeros((3,))
    hl = ql[0]
    vhl = ql[1]
    whl = ql[2]

    hr = qr[0]
    vhr = qr[1]
    whr = qr[2]

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
def cross(a:dace.float64[3], b:dace.float64[3]) -> dace.float64[3]:
    out = np.zeros((3,))
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]
    return out

@dace.program
def rotate_3d(point1_3d, point2_3d) -> dace.float64[3]:
    mid_point = (point1_3d + point2_3d) / 2.0
    mid_point = mid_point / np.sqrt(mid_point[0]*mid_point[0] + mid_point[1]*mid_point[1] + mid_point[2]*mid_point[2])
    edge_vector = point2_3d - point1_3d
    edge_vector = edge_vector / np.sqrt(edge_vector[0]*edge_vector[0] + edge_vector[1]*edge_vector[1] + edge_vector[2]*edge_vector[2])
    tmp = cross(mid_point, edge_vector)
    out = tmp / np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2])
    return out

@dace.program
def velocity_tan(q:dace.float64[4], vector_nor:dace.float64[3], vector_tan:dace.float64[3])->dace.float64[3]:
    """
    Transfer the velocity from xyz to latlng coordinate system
    """
    q_tan = np.zeros((3,))
    q_tan[0] = q[0]
    q_tan[1] = np.dot(q[1:], vector_nor)
    q_tan[2] = np.dot(q[1:], vector_tan)
    return q_tan

@dace.program
def velocity_to_xyz(q:dace.float64[3], vector_nor:dace.float64[3], vector_tan:dace.float64[3])->dace.float64[4]:
    """
    Transfer the velocity from latlng to xyz coordinate system
    """
    q_xyz = np.zeros((4,))
    q_xyz[0] = q[0]
    v = q[1] * vector_nor + q[2] * vector_tan
    q_xyz[1] = v[0]
    q_xyz[2] = v[1]
    q_xyz[3] = v[2]
    return q_xyz

@dace.program
def momentum_on_tan(momentum:dace.float64[4], cell_center_3d:dace.float64[3])->dace.float64[4]:

    tmp = momentum[1]*cell_center_3d[0] + momentum[2]*cell_center_3d[1] + momentum[3]*cell_center_3d[2]

    momentum[1] = momentum[1] - tmp*cell_center_3d[0]
    momentum[2] = momentum[2] - tmp*cell_center_3d[1]
    momentum[3] = momentum[3] - tmp*cell_center_3d[2]

@dace.program
def flux_rusanov(ql: dace.float64[4], qr: dace.float64[4], point1_3d: dace.float64[3], point2_3d: dace.float64[3], r: dace.float64)->dace.float64[4]:
    """
    Rusanov flux scheme
    out: double[4], accumulate result with the out array
    ql: double[4], left state
    qr: double[4], right state
    """
    #compute the length of the edge
    edge_length = d_sphere_3d(point1_3d, point2_3d, r)
    #compute tan and normal vector
    edge_vector = point2_3d - point1_3d
    tan_vector = edge_vector / np.sqrt(edge_vector[0]*edge_vector[0] + edge_vector[1]*edge_vector[1] + edge_vector[2]*edge_vector[2])
    nor_vector = rotate_3d(point1_3d, point2_3d)
    #compute the velocity in ql and qr along latitude and longtitude
    ql_ = velocity_tan(ql, nor_vector, tan_vector)
    qr_ = velocity_tan(qr, nor_vector, tan_vector)
    #compute the numerical flux and rotate it back to xyz coordinate system
    diffu = max(max_eigenvalue(ql_), max_eigenvalue(qr_)) * (qr_ - ql_)
    fql = f(ql_)
    fqr = f(qr_)
    tmp = 0.5 * edge_length * ((fql+fqr) - diffu)
    out = velocity_to_xyz(tmp, nor_vector, tan_vector)
    return out

@dace.program
def flux_roe(ql: dace.float64[4], qr: dace.float64[4], point1_3d: dace.float64[3], point2_3d: dace.float64[3], r: dace.float64)->dace.float64[4]:
    #compute the length of the edge
    edge_length = d_sphere_3d(point1_3d, point2_3d, r)
    #compute tan and normal vector
    edge_vector = point2_3d - point1_3d
    tan_vector = edge_vector / np.sqrt(edge_vector[0]*edge_vector[0] + edge_vector[1]*edge_vector[1] + edge_vector[2]*edge_vector[2])
    nor_vector = rotate_3d(point1_3d, point2_3d)
    #compute the velocity in ql and qr along latitude and longtitude
    ql_ = velocity_tan(ql, nor_vector, tan_vector)
    qr_ = velocity_tan(qr, nor_vector, tan_vector)
    
    roe_ave = roe_average(ql_, qr_)
    hm = roe_ave[0]
    vm = roe_ave[1]
    wm = roe_ave[2]
    cm = np.sqrt(g*hm)

    eigval_abs = np.zeros((3,))
    eigval_abs[0] = abs(vm - cm)
    eigval_abs[1] = abs(vm)
    eigval_abs[2] = abs(vm + cm)

    du = qr_ - ql_

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
    fql = f(ql_)
    fqr = f(qr_)
    tmp = 0.5 * edge_length * ((fql + fqr) - diffu)
    out = velocity_to_xyz(tmp, nor_vector, tan_vector)
    return out

@dace.program
def flux_hll(ql: dace.float64[4], qr: dace.float64[4], point1_3d: dace.float64[3], point2_3d: dace.float64[3], r: dace.float64)->dace.float64[4]:
    # HLL's flux scheme
    #compute the length of the edge
    edge_length = d_sphere_3d(point1_3d, point2_3d, r)
    #compute tan and normal vector
    edge_vector = point2_3d - point1_3d
    tan_vector = edge_vector / np.sqrt(edge_vector[0]*edge_vector[0] + edge_vector[1]*edge_vector[1] + edge_vector[2]*edge_vector[2])
    nor_vector = rotate_3d(point1_3d, point2_3d)
    #compute the velocity in ql and qr along latitude and longtitude
    ql_ = velocity_tan(ql, nor_vector, tan_vector)
    qr_ = velocity_tan(qr, nor_vector, tan_vector)

    fql = f(ql_)
    fqr = f(qr_)

    eigvalsl = np.zeros((3,))
    eigvalsl[0] = ql_[1]/ql_[0] - np.sqrt(np.abs(g*ql_[0]))
    eigvalsl[1] = ql_[1]/ql_[0]
    eigvalsl[2] = ql_[1]/ql_[0] + np.sqrt(np.abs(g*ql_[0]))

    eigvalsr = np.zeros((3,))
    eigvalsr[0] = qr_[1]/qr_[0] - np.sqrt(np.abs(g*qr_[0]))
    eigvalsr[1] = qr_[1]/qr_[0]
    eigvalsr[2] = qr_[1]/qr_[0] + np.sqrt(np.abs(g*qr_[0]))

    sl = min(eigvalsl[0], eigvalsl[1], eigvalsl[2], eigvalsr[0], eigvalsr[1], eigvalsr[2])
    sr = max(eigvalsl[0], eigvalsl[1], eigvalsl[2], eigvalsr[0], eigvalsr[1], eigvalsr[2])

    if sl >= 0.0:
        tmp = fql * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    elif sr <= 0.0:
        tmp = fqr * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    else:
        tmp = (sl*sr*(qr_ - ql_) + fql*sr - fqr*sl)/(sr-sl) * edge_length 
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out

@dace.program
def flux_hlle(ql: dace.float64[4], qr: dace.float64[4], point1_3d: dace.float64[3], point2_3d: dace.float64[3], r: dace.float64)->dace.float64[4]:
    # HLLE flux scheme
    # https://www.sciencedirect.com/science/article/pii/S0898122199002965
    # A two-dimensional HLLE riemann solver and associated godunov-type difference scheme for gas dynamics

    #compute the length of the edge
    edge_length = d_sphere_3d(point1_3d, point2_3d, r)
    #compute tan and normal vector
    edge_vector = point2_3d - point1_3d
    tan_vector = edge_vector / np.sqrt(edge_vector[0]*edge_vector[0] + edge_vector[1]*edge_vector[1] + edge_vector[2]*edge_vector[2])
    nor_vector = rotate_3d(point1_3d, point2_3d)
    #compute the velocity in ql and qr along latitude and longtitude
    ql_ = velocity_tan(ql, nor_vector, tan_vector)
    qr_ = velocity_tan(qr, nor_vector, tan_vector)
    
    roe_ave = roe_average(ql_, qr_)
    hm = roe_ave[0]
    vm = roe_ave[1]
    wm = roe_ave[2]
    cm = np.sqrt(g*hm)

    fql = f(ql_)
    fqr = f(qr_)

    vl = ql_[1] / ql_[0]
    vr = qr_[1] / qr_[0]
    cl = np.sqrt(np.abs(g*ql_[0]))
    cr = np.sqrt(np.abs(g*qr_[0]))
    sl = min(vm - cm, vl - cl)
    sr = max(vm + cm, vr + cr)
    
    if sl >= 0.0:
        tmp = fql * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    elif sr <= 0.0:
        tmp = fqr * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    else:
        tmp = (sl*sr*(qr_ - ql_) + fql*sr - fqr*sl)/(sr-sl) * edge_length 
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out

@dace.program
def flux_hllc(ql: dace.float64[4], qr: dace.float64[4], point1_3d: dace.float64[3], point2_3d: dace.float64[3], r: dace.float64)->dace.float64[4]:
    # HLLC flux scheme
    # https://link.springer.com/content/pdf/10.1007%2F978-3-662-03490-3_10.pdf
    # p9 p301

    # HLLE speeds
    #compute the length of the edge
    edge_length = d_sphere_3d(point1_3d, point2_3d, r)
    #compute tan and normal vector
    edge_vector = point2_3d - point1_3d
    tan_vector = edge_vector / np.sqrt(edge_vector[0]*edge_vector[0] + edge_vector[1]*edge_vector[1] + edge_vector[2]*edge_vector[2])
    nor_vector = rotate_3d(point1_3d, point2_3d)
    #compute the velocity in ql and qr along latitude and longtitude
    ql_ = velocity_tan(ql, nor_vector, tan_vector)
    qr_ = velocity_tan(qr, nor_vector, tan_vector)

    fql = f(ql_)
    fqr = f(qr_)

    hl = ql_[0]
    hr = qr_[0]
    
    vl = ql_[1] / hl
    vr = qr_[1] / hr
    
    wl = ql_[2] / hl
    wr = qr_[2] / hr
    
    cl = np.sqrt(np.abs(g*hl))
    cr = np.sqrt(np.abs(g*hr))
    
    vm = 0.5 * (vl+vr) + cl - cr
    #hm = ((0.5*(np.sqrt(g*hl)+np.sqrt(g*hr))+0.25*(vl-vr))**2)/g
    cm = 0.25 * (vl-vr) + 0.5 * (cl+cr)
    if hl<=0:
        sl = vr - 2*np.sqrt(np.abs(g*hr))
    else:
        sl = min(vm - cm, vl - cl)
        
    if hr<=0:
        sr = vl + 2.0*np.sqrt(np.abs(g*hl))
    else:
        sr = max(vm + cm, vr + cr)
    
    # middle state estimate
    hlm = (sl - vl)/(sl - vm) * hl
    hrm = (sr - vr)/(sr - vm) * hr
    if sl >= 0.0:
        tmp = fql * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    elif sr <= 0.0:
        tmp = fqr * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    elif vm >= 0.0:
        u_cons = np.zeros((3,))
        u_cons[0] = hlm
        u_cons[1] = hlm * vm
        u_cons[2] = hlm * wl
        tmp = (fql + sl*(u_cons - ql_)) * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out
    else: # vm <= 0
        u_cons = np.zeros((3,))
        u_cons[0] = hrm
        u_cons[1] = hrm*vm
        u_cons[2] = hrm*wr
        tmp = (fqr + sr*(u_cons - qr_)) * edge_length
        out = velocity_to_xyz(tmp, nor_vector, tan_vector)
        return out

@dace.program
def coriolis(q:dace.float64[4], cell_center_3d:dace.float64[3], dt:dace.float64)->dace.float64[4]:
    # The source term models the Coriolis force using a 4-stage RK method
    df = 12.600576
    fq = np.zeros((4,))
    RK = np.zeros((4,3))
    #compute the cell center's sphere tangential vector along latitude and longtitude
    cell_center_3d = cell_center_3d / np.sqrt(cell_center_3d[0]**2 + cell_center_3d[1]**2 + cell_center_3d[2]**2)
    erx = cell_center_3d[0]
    ery = cell_center_3d[1]
    erz = cell_center_3d[2]
    # calculate Coriolis term
    fcor = df * erz

    #stage 1
    hu_1 = q[1]
    hv_1 = q[2]
    hw_1 = q[3]
                   
    RK[0,0] = fcor*dt*(erz*hv_1-ery*hw_1)
    RK[0,1] = fcor*dt*(erx*hw_1-erz*hu_1)
    RK[0,2] = fcor*dt*(ery*hu_1-erx*hv_1)
    #stage 2
    hu_2 = q[1] + 0.5*RK[0,0]
    hv_2 = q[2] + 0.5*RK[0,1]
    hw_2 = q[3] + 0.5*RK[0,2]

    RK[1,0] = fcor*dt*(erz*hv_2-ery*hw_2)
    RK[1,1] = fcor*dt*(erx*hw_2-erz*hu_2)
    RK[1,2] = fcor*dt*(ery*hu_2-erx*hv_2)
    #stage 3
    hu_3 = q[1] + 0.5*RK[1,0]
    hv_3 = q[2] + 0.5*RK[1,1]
    hw_3 = q[3] + 0.5*RK[1,2]

    RK[2,0] = fcor*dt*(erz*hv_3-ery*hw_3)
    RK[2,1] = fcor*dt*(erx*hw_3-erz*hu_3)
    RK[2,2] = fcor*dt*(ery*hu_3-erx*hv_3)
    #stage 4
    hu_4 = q[1] + 0.5*RK[2,0]
    hv_4 = q[2] + 0.5*RK[2,1]
    hw_4 = q[3] + 0.5*RK[2,2]

    RK[3,0] = fcor*dt*(erz*hv_4-ery*hw_4)
    RK[3,1] = fcor*dt*(erx*hw_4-erz*hu_4)
    RK[3,2] = fcor*dt*(ery*hu_4-erx*hv_4)

    for i in range(1,4):
        fq[i] = (RK[0,i-1] + 2.0*RK[1,i-1] + 2.0*RK[2,i-1] + RK[3,i-1]) / 6.0
    fq[0] = 0.0 
    return fq

@dace.program
def coriolis_euler(q:dace.float64[4], cell_center_3d:dace.float64[3], dt:dace.float64)->dace.float64[4]:
    # The source term models the Coriolis force using a Euler method
    df = 12.600576
    fq = np.zeros((4,))
    #compute the cell center's sphere tangential vector along latitude and longtitude
    cell_center_3d = cell_center_3d / np.sqrt(cell_center_3d[0]**2 + cell_center_3d[1]**2 + cell_center_3d[2]**2)
    erx = cell_center_3d[0]
    ery = cell_center_3d[1]
    erz = cell_center_3d[2]
    # calculate Coriolis term
    fcor = df * erz
    hu = q[1]
    hv = q[2]
    hw = q[3]

    fq[0] = 0.0 
    fq[1] = fcor*dt*(erz*hv-ery*hw)
    fq[2] = fcor*dt*(erx*hw-erz*hu)
    fq[3] = fcor*dt*(ery*hu-erx*hv)
    return fq

@dace.program
def correction(q:dace.float64[4], vertbl_3d:dace.float64[3], vertul_3d:dace.float64[3], vertur_3d:dace.float64[3], vertbr_3d:dace.float64[3],r:dace.float64)->dace.float64[4]:
    n_l = rotate_3d(vertbl_3d, vertul_3d)
    n_u = rotate_3d(vertul_3d, vertur_3d)
    n_r = rotate_3d(vertur_3d, vertbr_3d)
    n_b = rotate_3d(vertbr_3d, vertbl_3d)

    h_l = d_sphere_3d(vertbl_3d, vertul_3d, r)
    h_u = d_sphere_3d(vertul_3d, vertur_3d, r)
    h_r = d_sphere_3d(vertur_3d, vertbr_3d, r)
    h_b = d_sphere_3d(vertbr_3d, vertbl_3d, r)
    
    cor = fn(q, -h_l*n_l - h_r*n_r - h_b*n_b - h_u*n_u)

    out = np.zeros((4,))

    out[0] = 0.0
    out[1] = cor[1]
    out[2] = cor[2]
    out[3] = cor[3]
    return out

flux = flux_roe

@dace.program
def fvm_1st_order(q0:dace.float64[M+4, N+4, 4], q1:dace.float64[M+4, N+4, 4], vert_3d:dace.float64[M+1, N+1, 3], cell_center_3d:dace.float64[M, N, 3], cell_area:dace.float64[M, N], dt:dace.float64, r:dace.float64):
    """
    q0: double[M+2,N+2,4], state matrix of current timestep
    q1: double[M+2,N+2,4], state matrix of next timestep
    vert_3d: double[M+1,N+1,3], matrix of vertices
    vert_sphere: double[M+1,N+1,2], matrix of vertices in latlng system
    cell_area: double[M, N], matrix of cell area 
    """
    for i in range(2, M+2):
        for j in range(2, N+2):
            vertbl_3d = vert_3d[i-2, j-2, :]
            vertul_3d = vert_3d[i-2, j-1, :]
            vertur_3d = vert_3d[i-1, j-1, :]
            vertbr_3d = vert_3d[i-1, j-2, :]

            qc = q0[i, j, :]
            ql = q0[i-1, j, :]
            qu = q0[i, j+1, :]
            qr = q0[i+1, j, :]
            qb = q0[i, j-1, :]
            
            cell_center = cell_center_3d[i-2, j-2]

            area = cell_area[i-2, j-2]
            #compute flux term
            flux_sum = flux(qc, ql, vertbl_3d, vertul_3d, r) +\
                       flux(qc, qu, vertul_3d, vertur_3d, r) +\
                       flux(qc, qr, vertur_3d, vertbr_3d, r) +\
                       flux(qc, qb, vertbr_3d, vertbl_3d, r)
            #compute source and correction term
            fq = coriolis(qc, cell_center, dt)
            cor = correction(qc, vertbl_3d, vertul_3d, vertur_3d, vertbr_3d, r)

            # project momentum components of q onto tangent plane:
            momentum_on_tan(flux_sum, cell_center)
            momentum_on_tan(fq, cell_center)
            momentum_on_tan(cor, cell_center)
            q1[i, j, :] = qc - dt/area * (flux_sum + cor) + fq
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
def fvm_2ndorder_space(q0:dace.float64[M+4, N+4, 4], q1:dace.float64[M+4, N+4, 4], vert_3d:dace.float64[M+1, N+1, 3], cell_center_3d:dace.float64[M, N, 3], cell_area:dace.float64[M, N], dt:dace.float64, r:dace.float64):
    # 2nd order fvm scheme using limiters for conservative variables in space
    #compute x-direction slope
    fdx = q0[2:, 1:-1 ,:] - q0[1:-1, 1:-1, :]
    bdx = q0[1:-1, 1:-1, :] - q0[:-2, 1:-1, :]
    cdx = 0.5 * (q0[2:, 1:-1 ,:] - q0[:-2, 1:-1, :])
    slope_x = limiter(fdx, cdx, bdx)
    #compute y-direction slope
    fdy = q0[1:-1, 2: ,:] - q0[1:-1, 1:-1, :]
    bdy = q0[1:-1, 1:-1, :] - q0[1:-1, :-2, :]
    cdy = 0.5 * (q0[1:-1, 2: ,:] - q0[1:-1, :-2, :])
    slope_y = limiter(fdy, cdy, bdy)
    # #compute cell distance
    q0l = np.zeros_like(q0)
    q0r = np.zeros_like(q0)
    q0u = np.zeros_like(q0)
    q0b = np.zeros_like(q0)

    q0l[1:-1, 1:-1, :] = q0[1:-1, 1:-1, :] - 0.5*slope_x
    q0r[1:-1, 1:-1, :] = q0[1:-1, 1:-1, :] + 0.5*slope_x
    q0u[1:-1, 1:-1, :] = q0[1:-1, 1:-1, :] + 0.5*slope_y
    q0b[1:-1, 1:-1, :] = q0[1:-1, 1:-1, :] - 0.5*slope_y

    for i in range(2, M+2):
        for j in range(2, N+2):
            vertbl_3d = vert_3d[i-2, j-2, :]
            vertul_3d = vert_3d[i-2, j-1, :]
            vertur_3d = vert_3d[i-1, j-1, :]
            vertbr_3d = vert_3d[i-1, j-2, :]

            qc = q0[i, j, :]

            qlp = q0r[i-1, j, :]
            qlm = q0l[i, j, :]

            qrp = q0l[i+1, j, :]
            qrm = q0r[i, j, :]

            qbp = q0u[i, j-1, :]
            qbm = q0b[i, j, :]
            
            qup = q0b[i, j+1, :]
            qum = q0u[i, j, :]

            cell_center = cell_center_3d[i-2, j-2]

            area = cell_area[i-2, j-2]

            #compute flux term
            flux_sum = flux(qlm, qlp, vertbl_3d, vertul_3d, r) +\
                       flux(qum, qup, vertul_3d, vertur_3d, r) +\
                       flux(qrm, qrp, vertur_3d, vertbr_3d, r) +\
                       flux(qbm, qbp, vertbr_3d, vertbl_3d, r)
            #compute source and correction term
            fq = coriolis(qc, cell_center, dt)
            cor = correction(qc, vertbl_3d, vertul_3d, vertur_3d, vertbr_3d, r)
            # project momentum components of q onto tangent plane:
            momentum_on_tan(flux_sum, cell_center)
            momentum_on_tan(fq, cell_center)
            momentum_on_tan(cor, cell_center)
            q1[i, j, :] = qc - dt/area * (flux_sum + cor) + fq
    boundary_condition(q1)

@dace.program
def fvm_2nd_order(q0:dace.float64[M+4, N+4, 4], q1:dace.float64[M+4, N+4, 4], vert_3d:dace.float64[M+1, N+1, 3], cell_center_3d:dace.float64[M, N, 3], cell_area:dace.float64[M, N], dt:dace.float64, r:dace.float64):
    """
    q0: double[M+2,N+2,3], state matrix of current timestep
    q1: double[M+2,N+2,3], state matrix of next timestep
    vert: double[M+1,N+1,2], matrix of vertices
    cell_area: double[M, N], matrix of cell area 
    """
    qs = np.zeros_like(q0)
    qss = np.zeros_like(q0)

    fvm_2ndorder_space(q0, qs, vert_3d, cell_center_3d, cell_area, dt, r)
    fvm_2ndorder_space(qs, qss, vert_3d, cell_center_3d, cell_area, dt, r)
    q1[...] = 0.5 * (q0 + qss)

def initial_condition(x_range, y_range, M, N, cell_center_3d):
    a = 6.37122e6
    K = 7.848e-6
    G = 9.8
    t0 = 86400.
    h0 = 8.e3
    r = 4.0
    omega = 7.292e-5
    q_init = np.zeros((M, N, 4))
    vert_3d = np.zeros((M, N, 3))
    vert_sphere = np.zeros((M, N, 2))

    xmin, xmax = x_range
    ymin, ymax = y_range
    dx = (xmax - xmin)/M
    dy = (ymax - ymin)/N
    x = np.linspace(xmin+dx/2, xmax-dx/2, M)
    y = np.linspace(ymin+dy/2, ymax-dy/2, N)
    vert = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    #initial vert_sphere, vert_3d
    MeshGrid.transfer_sphere_surface(vert_3d, vert, r)
    MeshGrid.xyz_to_latlng(vert_sphere, vert_3d, r)

    xp = vert_sphere[:,:,1] * np.pi/ 180
    yp = vert_sphere[:,:,0] * np.pi/ 180

    A = 0.5 * K * (2.0*omega + K) * np.cos(yp)**2.0 + \
                0.25 * K * K * np.cos(yp)**(2.0*r)*( \
                (1.0*r+1.0) * np.cos(yp)**2.0 \
                +(2.0*r*r - 1.0*r - 2.0) \
                - 2.0*r*r*(np.cos(yp))**(-2.0))

    B = (2.0*(omega+K) * K) / ((1.0*r+1.0)*(1.0*r+2.0)) \
                *np.cos(yp)**r*( (1.0*r*r + 2.0*r + 2.0) \
                - (1.0*r+1.0)**(2)*np.cos(yp)**2)

    C = 0.25*K*K*np.cos(yp)**(2*r)*( (1.0*r + 1.0)* \
                np.cos(yp)**2 - (1.0*r + 2.0))

    x_angular = (K*np.cos(yp)+K*np.cos(yp)**(r-1.)*(r*np.sin(yp)**2.0 \
                - np.cos(yp)**2.0 )*np.cos(r*xp))*t0

    y_angular = (-K*r*np.cos(yp)**(r-1.0)*np.sin(yp)*np.sin(r*xp))*t0

    z_angular = 0.0

    q = (-np.sin(xp)*x_angular-np.sin(yp)*np.cos(xp)*y_angular)
    v = (np.cos(xp)*x_angular-np.sin(yp)*np.sin(xp)*y_angular)
    w = np.cos(yp)*y_angular

    # q = -np.sin(xp)*np.cos(yp)*x_angular - np.cos(xp)*np.sin(yp)*y_angular
    # v = np.cos(xp)*np.cos(yp)*x_angular - np.sin(xp)*np.sin(yp)*y_angular
    # w = np.cos(yp)*y_angular

    q_init[:,:,0] =  h0/a + (a/G)*( A + B*np.cos(r*xp) \
                + C*np.cos(2.0*r*xp))

    q_init[:,:,1] = q_init[:,:,0] * q
    q_init[:,:,2] = q_init[:,:,0] * v
    q_init[:,:,3] = q_init[:,:,0] * w

    # project momentum components of q onto tangent plane:

    tmp = q_init[:,:,1]*cell_center_3d[:,:,0] + q_init[:,:,2]*cell_center_3d[:,:,1] + q_init[:,:,3]*cell_center_3d[:,:,2]
    q_init[:,:,1] -= tmp*cell_center_3d[:,:,0]
    q_init[:,:,2] -= tmp*cell_center_3d[:,:,1]
    q_init[:,:,3] -= tmp*cell_center_3d[:,:,2]

    # q_init[:,:,1] = 0.0
    # q_init[:,:,2] = 0.0
    # q_init[:,:,3] = 0.0
    return q_init

@dace.program
def boundary_condition(q:dace.float64[M+4, N+4, 4]):
    """
    Periodic boundary condition
    q: double[M+2, N+2, 4], state matrix with halo of width 1
    """
    # left boundary
    q[0, :, :] = q[-4, :, :]
    q[1, :, :] = q[-3, :, :]
    # right boundary
    q[-1, :, :] = q[3, :, :]
    q[-2, :, :] = q[2, :, :]
    for i in range(0,M+4):
        # lower boundary
        q[i, 0, :] = q[M+1-i, 3, :]
        q[i, 1, :] = q[M+1-i, 2, :]
        # upper boundary
        q[i, -1, :] = q[M+1-i, -4, :]
        q[i, -2, :] = q[M+1-i, -3, :]

fvm = fvm_2nd_order
@dace.program(auto_optimize=True)
def integrate(q_out: dace.float64[M, N, 4], q_init: dace.float64[M, N, 4], vert_3d: dace.float64[M+1, N+1, 3], cell_center_3d:dace.float64[M, N, 3], cell_area: dace.float64[M, N], r:dace.float64, Te: dace.float64):
    """
    Solve shallow water equation
    q_init: double[M, N, 3], initial condition
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
        dt_ = 0.001*(128/max(N, M/2))
        dt = min(dt_, Te-T)
        fvm(q0, q1, vert_3d, cell_center_3d, cell_area, dt, r)
        q0[...] = q1[...]
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = np.zeros((M,N,2))
            tmp[..., 0] = 0.5 * np.square(q0[2:M+2, 2:N+2, 0]) * g
            tmp[..., 1] = 0.5 * (np.square(q0[2:M+2, 2:N+2, 1]) + np.square(q0[2:M+2, 2:N+2, 2]) + np.square(q0[2:M+2, 2:N+2, 3])) / q0[2:M+2, 2:N+2, 0]
            sum_ = np.sum(tmp * cell_area[:,:, np.newaxis])
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)
    q_out[...] = q0[2:M+2, 2:N+2, :]

@dace.program(auto_optimize=True)
def integrate_slow(q_out: dace.float64[nT, M, N, 4], q_init: dace.float64[M, N, 4], vert_3d: dace.float64[M+1, N+1, 3], cell_center_3d:dace.float64[M, N, 3], cell_area: dace.float64[M, N], r:dace.float64, Te: dace.float64, save_ts: dace.float64[nT]):
    """
    Solve shallow water equation
    q_init: double[M, N, 3], initial condition
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
        dt_ = 0.001*(128/max(N, M/2))
        dt = min(dt_, save_ts[save_it] - T)
        fvm(q0, q1, vert_3d, cell_center_3d, cell_area, dt, r)
        q0[...] = q1[...]
        it += 1
        T += dt
        if it % 100 == 0:
            tmp = np.zeros((M,N,2))
            tmp[..., 0] = 0.5 * np.square(q0[2:M+2, 2:N+2, 0]) * g
            tmp[..., 1] = 0.5 * (np.square(q0[2:M+2, 2:N+2, 1]) + np.square(q0[2:M+2, 2:N+2, 2]) + np.square(q0[2:M+2, 2:N+2, 3])) / q0[2:M+2, 2:N+2, 0]
            sum_ = np.sum(tmp * cell_area[:,:, np.newaxis])
            print("T =", T, "it =", it, "dt =", dt, "Total energy:", sum_)

if __name__=="__main__":
    from argparse import ArgumentParser
    import MeshGrid
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=128, type=int)
    parser.add_argument("--resolution_y", default=64, type=int)
    parser.add_argument("--period", default=1.0, type=float)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--save_vtk", action="store_true")
    parser.add_argument("--save_xdmf", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    
    M = args.resolution_x
    N = args.resolution_y
    Te = args.period
    x_range = (-3., 1.)
    y_range = (-1., 1.)
    r = 1.0
    vert_3d, vert_sphere, cell_area = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
    cell_center_3d = MeshGrid.sphere_cell_center(x_range, y_range, M, N, 1.0)
    q_init = initial_condition(x_range, y_range, M, N, cell_center_3d)

    print(f"Sphere mesh grid {M}-{N} successfully generate!")

    path = f"{os.path.dirname(os.path.realpath(__file__))}/../data/dace/{fvm.name[4:]}/{Te}/"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.save_xdmf:
        save_ts = np.arange(0, args.period+0.01, args.period/10.)
        q_out = np.zeros((len(save_ts), M, N, 4))
        if args.use_cache:
            sdfg, _ = integrate_slow.load_sdfg("_dacegraphs/program.sdfg")
            integrate_slow_compiled = load_precompiled_sdfg(sdfg.build_folder)
        else :
            integrate_slow_compiled = integrate_slow.compile(simplify=True, save=True)
        print("Start computing------------------------")
        start_time = time.time()
        integrate_slow_compiled(q_out=q_out, q_init=q_init, vert_3d=vert_3d, cell_center_3d=cell_center_3d, cell_area=cell_area, r=r, Te=Te, save_ts=save_ts, M=M, N=N, nT=len(save_ts), print=print)
        end_time = time.time()
        print(f"End computing------------------------\nComputing time:",end_time - start_time,"s")
        MeshGrid.save_sphere_grid_xdmf(f"{path}/sphere-SWE-{M}-{N}-{flux.name[5:]}.xdmf", zip(save_ts, q_out), vert_3d)

    else:
        q_out= np.zeros_like(q_init)
        if args.use_cache:
            sdfg, _ = integrate_slow.load_sdfg("_dacegraphs/program.sdfg")
            integrate_compiled = load_precompiled_sdfg(sdfg.build_folder)
        else :
            integrate_compiled = integrate.compile(simplify=True, save=True)
        print("Start computing------------------------")
        start_time = time.time()
        integrate_compiled(q_out=q_out, q_init=q_init, vert_3d=vert_3d, cell_center_3d=cell_center_3d, cell_area=cell_area, r=r, Te=Te, M=M, N=N, print=print)    
        end_time = time.time()
        print(f"End computing------------------------\nComputing time:", end_time - start_time,"s")
        if args.save_vtk:
            MeshGrid.save_sphere_grid_vtk(f"{path}/sphere-SWE-{M}-{N}-{flux.name[5:]}", q_out, vert_3d)

        if args.plot:
            plt.figure(figsize=(10,8))
            plt.imshow(q_out[...,0], extent=[*x_range, *y_range], cmap='viridis', origin="lower")
            plt.colorbar()
            plt.savefig(f"{path}/sphere-SWE-{M}-{N}-{flux.name[5:]}.png")