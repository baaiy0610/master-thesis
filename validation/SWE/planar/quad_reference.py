#!/usr/bin/env python
# encoding: utf-8
# http://www.clawpack.org/gallery/pyclaw/gallery/radial_dam_break.html
r"""
2D shallow water: radial dam break
==================================

Solve the 2D shallow water equations:

.. math::
    h_t + (hu)_x + (hv)_y = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y = 0 \\
    (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y = 0.

The initial condition is a circular area with high depth surrounded by lower-depth water.
The top and right boundary conditions reflect, while the bottom and left boundaries
are outflow.
"""

from __future__ import absolute_import
import numpy as np
import meshio
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import depth, x_momentum, y_momentum, num_eqn
import os

def save_structured_grid(filename, controller, save_time_series=False, rank=0):
    #vert_x, vert_y = controller.grid.p_nodes
    q = controller.solution.state.get_q_global()
    if rank == 0:
        _, nx, ny = q.shape
        xs = np.linspace(0., 1., nx+1)
        ys = np.linspace(0., 1., ny+1)
        vert_x, vert_y = np.meshgrid(xs, ys, indexing="ij")
        indi, indj = np.meshgrid(np.arange(vert_x.shape[0]), np.arange(vert_x.shape[1]), indexing="ij")
        cell_i = np.stack((indi[:-1, :-1], indi[1:, :-1], indi[1:, 1:], indi[:-1, 1:]), axis=-1)
        cell_j = np.stack((indj[:-1, :-1], indj[1:, :-1], indj[1:, 1:], indj[:-1, 1:]), axis=-1)
        cells = [("quad", (cell_i*(vert_x.shape[1])+cell_j).reshape((-1, 4)))]
        verts = np.stack((vert_x, vert_y), axis=-1).reshape((-1, 2))
    if save_time_series:
        if rank == 0:
            writer = meshio.xdmf.TimeSeriesWriter(f"{filename}.xdmf")
            writer.__enter__()
            writer.write_points_cells(verts, cells)
        for sol in controller.frames:
            q = sol.state.get_q_global()
            if rank == 0:
                q = q.reshape((3, -1))
                writer.write_data(sol.t, cell_data={"h": [q[0, ...]], "hu": [q[1, ...]], "hv": [q[2, ...]]})
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
            meshio.write_points_cells(f"{filename}.vtk", verts, cells, cell_data={"h": [q[0, ...]], "hu": [q[1, ...]], "hv": [q[2, ...]]})

def qinit(state,h_in=1.0,h_out=0.5):
    x0=0.5
    y0=0.5
    sigma=0.5
    X, Y = state.p_centers
    r2 = (X-x0)**2 + (Y-y0)**2

    state.q[depth     ,:,:] = h_in*np.exp(-r2/sigma**2) + h_out
    state.q[x_momentum,:,:] = 0.
    state.q[y_momentum,:,:] = 0.

    
def setup(kernel_language='Fortran', use_petsc=False,
          solver_type='classic', riemann_solver='roe', resolution_x=150, resolution_y=150, extent=[0.,1.,0.,1.], period=0.15):
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if riemann_solver.lower() == 'roe':
        rs = riemann.shallow_roe_with_efix_2D
    elif riemann_solver.lower() == 'hlle':
        rs = riemann.shallow_hlle_2D

    if solver_type == 'classic':
        solver = pyclaw.ClawSolver2D(rs)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.dimensional_split=1
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D(rs)

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    # Domain:
    xlower = extent[0]
    xupper = extent[1]
    mx = resolution_x
    ylower = extent[0]
    yupper = extent[1]
    my = resolution_y
    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    y = pyclaw.Dimension(ylower,yupper,my,name='y')
    domain = pyclaw.Domain([x,y])

    state = pyclaw.State(domain,num_eqn)

    # Gravitational constant
    state.problem_data['grav'] = 9.8

    qinit(state)

    claw = pyclaw.Controller()
    claw.tfinal = period
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.keep_copy = True
    claw.output_style = 1
    #if disable_output:
    claw.output_format = None
    #claw.outdir = outdir
    claw.num_output_times = 10
    #claw.setplot = setplot
    claw.keep_copy = True

    return claw

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=1000, type=int)
    parser.add_argument("--resolution_y", default=1000, type=int)
    parser.add_argument("--period", default=1.5, type=float)
    parser.add_argument("--solver", default="sharpclaw")
    parser.add_argument("--riemann_solver", default="hlle")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--savets", action="store_true")
    args = parser.parse_args()
    xmin, xmax, ymin, ymax = 0., 1., 0., 1.
    extent = [xmin, xmax, ymin, ymax]
    controller = setup(riemann_solver=args.riemann_solver, solver_type=args.solver, resolution_x=args.resolution_x, resolution_y=args.resolution_y, use_petsc=args.parallel, extent=extent, period=args.period)
    output = controller.run()
    
    if args.parallel:
        from petsc4py import PETSc
        comm = PETSc.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0
    path = os.path.dirname(os.path.realpath(__file__))
    # fname = f"{path}/../data/{args.solver}-{args.riemann_solver}-{args.resolution_x}-{args.resolution_y}"
    fname = f"{path}/{args.solver}-{args.resolution_x}-{args.resolution_y}"
    save_structured_grid(fname, controller, save_time_series=args.savets, rank=rank)
    
    if rank == 0:
        print(output)
        
    if args.plot:
        q = controller.solution.state.get_q_global()
        if rank == 0:
            import matplotlib.pyplot as plt
            h = q[0, ...]
            plt.figure(figsize=(10,8))
            plt.imshow(h.transpose(), extent=extent)
            plt.colorbar()
            plt.savefig(f"{fname}.png")
