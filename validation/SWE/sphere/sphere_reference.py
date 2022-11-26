#!/usr/bin/env python
# encoding: utf-8
"""
Shallow water flow on the sphere
================================
2D shallow water equations on a spherical surface. The approximation of the 
three-dimensional equations is restricted to the surface of the sphere. 
Therefore only the solution on the surface is updated. 
Reference: Logically Rectangular Grids and Finite Volume Methods for PDEs in 
           Circular and Spherical Domains. 
           By Donna A. Calhoun, Christiane Helzel, and Randall J. LeVeque
           SIAM Review 50 (2008), 723-752. 
"""

from __future__ import absolute_import
from __future__ import print_function
import math
import os
import sys

import numpy as np
import meshio
from clawpack import pyclaw
from clawpack import riemann
from clawpack.pyclaw.util import inplace_build
import MeshGrid

try:
    from clawpack.pyclaw.examples.shallow_sphere import problem
    from clawpack.pyclaw.examples.shallow_sphere import classic2

except ImportError:
    this_dir = os.path.dirname(__file__)
    if this_dir == '':
        this_dir = os.path.abspath('.')
    inplace_build(this_dir)

    try:
        # Now try to import again
        from clawpack.pyclaw.examples.shallow_sphere import problem
        from clawpack.pyclaw.examples.shallow_sphere import classic2
    except ImportError:
        print("***\nUnable to import problem module or automatically build, try running (in the directory of this file):\n python setup.py build_ext -i\n***", file=sys.stderr)
        raise


# Nondimensionalized radius of the earth
Rsphere = 1.0

def save_structured_grid(filename, controller, vert, save_time_series=False, rank=0):
    #vert_x, vert_y = controller.grid.p_nodes
    q = controller.solution.state.get_q_global()
    if rank == 0:
        _, nx, ny = q.shape
        # xs = np.linspace(0., 1., nx+1)
        # ys = np.linspace(0., 1., ny+1)
        # vert_x, vert_y = np.meshgrid(xs, ys, indexing="ij")
        # indi, indj = np.meshgrid(np.arange(vert_x.shape[0]), np.arange(vert_x.shape[1]), indexing="ij")
        # cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
        # cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
        # cells = [("quad", (cell_i*(vert_x.shape[1])+cell_j).reshape((-1, 4)))]
        # verts = np.stack((vert_x, vert_y), axis=-1).reshape((-1, 3))
        indi, indj = np.meshgrid(np.arange(vert.shape[0]), np.arange(vert.shape[1]), indexing="ij")
        cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
        cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
        cells = [("quad", (cell_i*(vert.shape[1])+cell_j).reshape((-1, 4)))]
        verts = vert.reshape((-1,3))
    if save_time_series:
        if rank == 0:
            writer = meshio.xdmf.TimeSeriesWriter(f"{filename}.xdmf")
            writer.__enter__()
            writer.write_points_cells(verts, cells)
        for sol in controller.frames:
            q = sol.state.get_q_global()
            if rank == 0:
                q = q.reshape((4, -1))
                writer.write_data(sol.t, cell_data={"h": [q[0, ...]], "hu": [q[1, ...]], "hv": [q[2, ...]], "hw": [q[3, ...]]})
        if rank == 0:
            writer.__exit__()
        #for sol in controller.frames:
        #    q = sol.state.get_q_global()
        #    if rank == 0:
        #        q = q.reshape((3, -1))
        #        meshio.write_points_cells(f"{filename}-{sol.t:0.3f}.vtk", verts, cells, cell_data={"h": [q[0, ...]], "hu": [q[1, ...]], "hv": [q[2, ...]]})
    else:
        if rank == 0:
            q = q.reshape((3, -1))
            meshio.write_points_cells(f"{filename}.vtk", verts, cells, cell_data={"h": [q[0, ...]], "hu": [q[1, ...]], "hv": [q[2, ...]], "hw": [q[3, ...]]})

def fortran_src_wrapper(solver,state,dt):
    """
    Wraps Fortran src2.f routine. 
    src2.f contains the discretization of the source term.
    """
    # Some simplifications
    grid = state.grid

    # Get parameters and variables that have to be passed to the fortran src2
    # routine.
    mx, my = grid.num_cells[0], grid.num_cells[1]
    num_ghost = solver.num_ghost
    xlower, ylower = grid.lower[0], grid.lower[1]
    dx, dy = grid.delta[0], grid.delta[1]
    q = state.q
    aux = state.aux
    t = state.t

    # Call src2 function
    state.q = problem.src2(mx,my,num_ghost,xlower,ylower,dx,dy,q,aux,t,dt,Rsphere)

def mapc2p_sphere_nonvectorized(X,Y):
    """
    Maps to points on a sphere of radius Rsphere. Nonvectorized version (slow).
    
    Inputs: x-coordinates, y-coordinates in the computational space.
    Output: list of x-, y- and z-coordinates in the physical space.
    NOTE: this function is not used in the standard script.
    """

    # Get number of cells in both directions
    mx, my = X.shape

    # Define new list of numpy array, pC = physical coordinates
    pC = []

    for i in range(mx):
        for j in range(my):
            xc = X[i][j]
            yc = Y[i][j]

            # Ghost cell values outside of [-3,1]x[-1,1] get mapped to other
            # hemisphere:
            if (xc >= 1.0):
                xc = xc - 4.0
            if (xc <= -3.0):
                xc = xc + 4.0

            if (yc >= 1.0):
                yc = 2.0 - yc
                xc = -2.0 - xc

            if (yc <= -1.0):
                yc = -2.0 - yc
                xc = -2.0 - xc

            if (xc <= -1.0):
                # Points in [-3,-1] map to lower hemisphere - reflect about x=-1
                # to compute x,y mapping and set sgnz appropriately:
                xc = -2.0 - xc
                sgnz = -1.0
            else:
                sgnz = 1.0

            sgnxc = math.copysign(1.0,xc)
            sgnyc = math.copysign(1.0,yc)

            xc1 = np.abs(xc)
            yc1 = np.abs(yc)
            d = np.maximum(np.maximum(xc1,yc1), 1.0e-10)     

            DD = Rsphere*d*(2.0 - d) / np.sqrt(2.0)
            R = Rsphere
            centers = DD - np.sqrt(np.maximum(R**2 - DD**2, 0.0))
            
            xp = DD/d * xc1
            yp = DD/d * yc1

            if (yc1 >= xc1):
                yp = centers + np.sqrt(np.maximum(R**2 - xp**2, 0.0))
            else:
                xp = centers + np.sqrt(np.maximum(R**2 - yp**2, 0.0))

            # Compute physical coordinates
            zp = np.sqrt(np.maximum(Rsphere**2 - (xp**2 + yp**2), 0.0))
            pC.append(xp*sgnxc)
            pC.append(yp*sgnyc)
            pC.append(zp*sgnz)

    return pC

def mapc2p_sphere_vectorized(X,Y):
    """
    Maps to points on a sphere of radius Rsphere. Vectorized version (fast).  
    Inputs: x-coordinates, y-coordinates in the computational space.
    Output: list of x-, y- and z-coordinates in the physical space.
    NOTE: this function is used in the standard script.
    """

    # Get number of cells in both directions
    mx, my = X.shape
    
    # 2D array useful for the vectorization of the function
    sgnz = np.ones((mx,my))

    # 2D coordinates in the computational domain
    xc = X[:][:]
    yc = Y[:][:]

    # Compute 3D coordinates in the physical domain
    # =============================================

    # Note: yc < -1 => second copy of sphere:
    ij2 = np.where(yc < -1.0)
    xc[ij2] = -xc[ij2] - 2.0;
    yc[ij2] = -yc[ij2] - 2.0;

    ij = np.where(xc < -1.0)
    xc[ij] = -2.0 - xc[ij]
    sgnz[ij] = -1.0;
    xc1 = np.abs(xc)
    yc1 = np.abs(yc)
    d = np.maximum(xc1,yc1)
    d = np.maximum(d, 1e-10)
    D = Rsphere*d*(2-d) / np.sqrt(2)
    R = Rsphere*np.ones((np.shape(d)))

    centers = D - np.sqrt(R**2 - D**2)
    xp = D/d * xc1
    yp = D/d * yc1

    ij = np.where(yc1==d)
    yp[ij] = centers[ij] + np.sqrt(R[ij]**2 - xp[ij]**2)
    ij = np.where(xc1==d)
    xp[ij] = centers[ij] + np.sqrt(R[ij]**2 - yp[ij]**2)
    
    # Define new list of numpy array, pC = physical coordinates
    pC = []

    xp = np.sign(xc) * xp
    yp = np.sign(yc) * yp
    zp = sgnz * np.sqrt(Rsphere**2 - (xp**2 + yp**2))
    
    pC.append(xp)
    pC.append(yp)
    pC.append(zp)

    return pC

def qinit(state,mx,my):
    r"""
    Initialize solution with 4-Rossby-Haurwitz wave.
    NOTE: this function is not used in the standard script.
    """
    # Parameters
    a = 6.37122e6     # Radius of the earth
    Omega = 7.292e-5  # Rotation rate
    G = 9.80616       # Gravitational acceleration

    K = 7.848e-6   
    t0 = 86400.0     
    h0 = 8.e3         # Minimum fluid height at the poles        
    R = 4.0

    # Compute the the physical coordinates of the cells' centerss
    # state.grid.compute_p_centers(recompute=True)
 
    for i in range(mx):
        for j in range(my):
            xp = state.grid.p_centers[0][i][j]
            yp = state.grid.p_centers[1][i][j]
            zp = state.grid.p_centers[2][i][j]

            rad = np.maximum(np.sqrt(xp**2 + yp**2),1.e-6)

            if (xp >= 0.0 and yp >= 0.0):
                theta = np.arcsin(yp/rad) 
            elif (xp <= 0.0 and yp >= 0.0):
                theta = np.pi - np.arcsin(yp/rad)
            elif (xp <= 0.0 and yp <= 0.0):
                 theta = -np.pi + np.arcsin(-yp/rad)
            elif (xp >= 0.0 and yp <= 0.0):
                theta = -np.arcsin(-yp/rad)

            # Compute phi, at north pole: pi/2 at south pool: -pi/2
            if (zp >= 0.0): 
                phi =  np.arcsin(zp/Rsphere) 
            else:
                phi = -np.arcsin(-zp/Rsphere)  
        
            xp = theta 
            yp = phi 


            bigA = 0.5*K*(2.0*Omega + K)*np.cos(yp)**2.0 + \
                   0.25*K*K*np.cos(yp)**(2.0*R)*((1.0*R+1.0)*np.cos(yp)**2.0 + \
                   (2.0*R*R - 1.0*R - 2.0) - 2.0*R*R*(np.cos(yp))**(-2.0))
            bigB = (2.0*(Omega + K)*K)/((1.0*R + 1.0)*(1.0*R + 2.0)) * \
                   np.cos(yp)**R*( (1.0*R*R + 2.0*R + 2.0) - \
                   (1.0*R + 1.0)**(2)*np.cos(yp)**2 )
            bigC = 0.25*K*K*np.cos(yp)**(2*R)*( (1.0*R + 1.0)* \
                   np.cos(yp)**2 - (1.0*R + 2.0))


            # Calculate local longitude-latitude velocity vector
            # ==================================================
            Uin = np.zeros(3)

            # Longitude (angular) velocity component
            Uin[0] = (K*np.cos(yp)+K*np.cos(yp)**(R-1.)*( R*np.sin(yp)**2.0 - \
                     np.cos(yp)**2.0)*np.cos(R*xp))*t0

            # Latitude (angular) velocity component
            Uin[1] = (-K*R*np.cos(yp)**(R-1.0)*np.sin(yp)*np.sin(R*xp))*t0

            # Radial velocity component
            Uin[2] = 0.0 # The fluid does not enter in the sphere
            

            # Calculate velocity vetor in cartesian coordinates
            # =================================================
            Uout = np.zeros(3)

            Uout[0] = (-np.sin(xp)*Uin[0]-np.sin(yp)*np.cos(xp)*Uin[1])
            Uout[1] = (np.cos(xp)*Uin[0]-np.sin(yp)*np.sin(xp)*Uin[1])
            Uout[2] = np.cos(yp)*Uin[1]

            # Set the initial condition
            # =========================
            state.q[0,i,j] =  h0/a + (a/G)*( bigA + bigB*np.cos(R*xp) + \
                              bigC*np.cos(2.0*R*xp))

            state.q[1,i,j] = state.q[0,i,j]*Uout[0] 
            state.q[2,i,j] = state.q[0,i,j]*Uout[1] 
            state.q[3,i,j] = state.q[0,i,j]*Uout[2] 

            # state.q[1,i,j] = 0.0
            # state.q[2,i,j] = 0.0 
            # state.q[3,i,j] = 0.0

def qbc_lower_y(state,dim,t,qbc,auxbc,num_ghost):
    """
    Impose periodic boundary condition to q at the bottom boundary for the 
    sphere. This function does not work in parallel.
    """
    for j in range(num_ghost):
        qbc1D = np.copy(qbc[:,:,2*num_ghost-1-j])
        qbc[:,:,j] = qbc1D[:,::-1]

def qbc_upper_y(state,dim,t,qbc,auxbc,num_ghost):
    """
    Impose periodic boundary condition to q at the top boundary for the sphere.
    This function does not work in parallel.
    """
    my = state.grid.num_cells[1]
    for j in range(num_ghost):
        qbc1D = np.copy(qbc[:,:,my+num_ghost-1-j])
        qbc[:,:,my+num_ghost+j] = qbc1D[:,::-1]

def auxbc_lower_y(state,dim,t,qbc,auxbc,num_ghost):
    """
    Impose periodic boundary condition to aux at the bottom boundary for the 
    sphere.
    """
    grid=state.grid

    # Get parameters and variables that have to be passed to the fortran src2
    # routine.
    mx, my = grid.num_cells[0], grid.num_cells[1]
    xlower, ylower = grid.lower[0], grid.lower[1]
    dx, dy = grid.delta[0],grid.delta[1]

    # Impose BC
    auxtemp = auxbc.copy()
    auxtemp = problem.setaux(mx,my,num_ghost,mx,my,xlower,ylower,dx,dy,auxtemp,Rsphere)
    auxbc[:,:,:num_ghost] = auxtemp[:,:,:num_ghost]

def auxbc_upper_y(state,dim,t,qbc,auxbc,num_ghost):
    """
    Impose periodic boundary condition to aux at the top boundary for the 
    sphere. 
    """
    grid=state.grid

    # Get parameters and variables that have to be passed to the fortran src2
    # routine.
    mx, my = grid.num_cells[0], grid.num_cells[1]
    xlower, ylower = grid.lower[0], grid.lower[1]
    dx, dy = grid.delta[0],grid.delta[1]
    
    # Impose BC
    auxtemp = auxbc.copy()
    auxtemp = problem.setaux(mx,my,num_ghost,mx,my,xlower,ylower,dx,dy,auxtemp,Rsphere)
    auxbc[:,:,-num_ghost:] = auxtemp[:,:,-num_ghost:]

def setup(use_petsc=False,solver_type='classic', resolution_x=150, resolution_y=150, period=0.15):
    if use_petsc:
        raise Exception("petclaw does not currently support mapped grids (go bug Lisandro who promised to implement them)")

    if solver_type != 'classic':
        raise Exception("Only Classic-style solvers (solver_type='classic') are supported on mapped grids")

    solver = pyclaw.ClawSolver2D(riemann.shallow_sphere_2D)
    solver.fmod = classic2

    # Set boundary conditions
    # =======================
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.custom  # Custom BC for sphere
    solver.bc_upper[1] = pyclaw.BC.custom  # Custom BC for sphere

    solver.user_bc_lower = qbc_lower_y
    solver.user_bc_upper = qbc_upper_y

    # Auxiliary array
    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[1] = pyclaw.BC.custom  # Custom BC for sphere
    solver.aux_bc_upper[1] = pyclaw.BC.custom  # Custom BC for sphere

    solver.user_aux_bc_lower = auxbc_lower_y
    solver.user_aux_bc_upper = auxbc_upper_y


    # Dimensional splitting ?
    # =======================
    solver.dimensional_split = 0
 
    # Transverse increment waves and transverse correction waves are computed 
    # and propagated.
    # =======================================================================
    solver.transverse_waves = 2

    
    # Use source splitting method
    # ===========================
    solver.source_split = 2

    # Set source function
    # ===================
    solver.step_source = fortran_src_wrapper

    #self_set_parameters
    # solver.order = 2
    # solver.cfl_desired = 0.15
    # solver.cfl_max = 0.2

    # Set the limiter for the waves
    # =============================
    solver.limiters = pyclaw.limiters.tvd.MC


    #===========================================================================
    # Initialize domain and state, then initialize the solution associated to the 
    # state and finally initialize aux array
    #===========================================================================
    # Domain:
    xlower = -3.0
    xupper = 1.0
    mx = resolution_x

    ylower = -1.0
    yupper = 1.0
    my = resolution_y

    # Check whether or not the even number of cells are used in in both 
    # directions. If odd numbers are used a message is print at screen and the 
    # simulation is interrputed.
    if(mx % 2 != 0 or my % 2 != 0):
        message = 'Please, use even numbers of cells in both direction. ' \
                  'Only even numbers allow to impose correctly the boundary ' \
                  'conditions!'
        raise ValueError(message)


    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    y = pyclaw.Dimension(ylower,yupper,my,name='y')
    domain = pyclaw.Domain([x,y])
    dx = domain.grid.delta[0]
    dy = domain.grid.delta[1]

    # Define some parameters used in Fortran common blocks 
    solver.fmod.comxyt.dxcom = dx
    solver.fmod.comxyt.dycom = dy
    solver.fmod.sw.g = 11489.57219  
    solver.rp.comxyt.dxcom = dx
    solver.rp.comxyt.dycom = dy
    solver.rp.sw.g = 11489.57219  

    # Define state object
    # ===================
    num_aux = 16 # Number of auxiliary variables
    state = pyclaw.State(domain,solver.num_eqn,num_aux)

    # Override default mapc2p function
    # ================================
    state.grid.mapc2p = mapc2p_sphere_vectorized
        

    # Set auxiliary variables
    # =======================
    
    # Get lower left corner coordinates 
    xlower,ylower = state.grid.lower[0],state.grid.lower[1]

    num_ghost = 2
    auxtmp = np.ndarray(shape=(num_aux,mx+2*num_ghost,my+2*num_ghost), dtype=float, order='F')
    auxtmp = problem.setaux(mx,my,num_ghost,mx,my,xlower,ylower,dx,dy,auxtmp,Rsphere)
    state.aux[:,:,:] = auxtmp[:,num_ghost:-num_ghost,num_ghost:-num_ghost]

    # Set index for capa
    state.index_capa = 0

    # Set initial conditions
    # ====================== 
    # 1) Call fortran function
    # qtmp = np.ndarray(shape=(solver.num_eqn,mx+2*num_ghost,my+2*num_ghost), dtype=float, order='F')
    # qtmp = problem.qinit(mx,my,num_ghost,mx,my,xlower,ylower,dx,dy,qtmp,auxtmp,Rsphere)
    # state.q[:,:,:] = qtmp[:,num_ghost:-num_ghost,num_ghost:-num_ghost]

    # 2) call python function define above
    qinit(state,mx,my)


    #===========================================================================
    # Set up controller and controller parameters
    #===========================================================================
    claw = pyclaw.Controller()
    # if disable_output:
    #     claw.output_format = None
    claw.output_style = 1
    claw.num_output_times = 10
    claw.tfinal = period
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.keep_copy = True

    # claw.outdir = outdir

    return claw

        
if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=400, type=int)
    parser.add_argument("--resolution_y", default=200, type=int)
    parser.add_argument("--period", default=1.5, type=float)
    parser.add_argument("--solver", default="classic")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--savets", action="store_true")
    args = parser.parse_args()
    x_range = (-3., 1.)
    y_range = (-1., 1.)
    M = resolution_x=args.resolution_x
    N = resolution_y=args.resolution_y
    vert_3d, vert_sphere, cell_area = MeshGrid.sphere_grid(x_range, y_range, M, N, Rsphere)

    controller = setup(solver_type=args.solver, resolution_x=args.resolution_x, resolution_y=args.resolution_y, use_petsc=args.parallel, period=args.period)
    output = controller.run()
    if args.parallel:
        from petsc4py import PETSc
        comm = PETSc.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0
    path = os.path.dirname(os.path.realpath(__file__))
    fname = f"{path}/{args.solver}-{args.resolution_x}-{args.resolution_y}"
    save_structured_grid(fname, controller, vert_3d, save_time_series=args.savets, rank=rank)
    if rank == 0:
        print(output)
