#!/usr/bin/env python
# encoding: utf-8
r"""
2D shallow water: flow over a sill
==================================
Solve the 2D shallow water equations with
variable bathymetry:
.. :math:
    h_t + (hu)_x + (hv)_y & = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y & = -g h b_x \\
    (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y & = -g h b_y.
The bathymetry contains a gaussian hump.
are outflow.
"""

from __future__ import absolute_import
from clawpack import riemann
from clawpack import pyclaw
from clawpack.riemann.shallow_roe_with_efix_2D_constants import depth, x_momentum, y_momentum, num_eqn
import numpy as np
import os
import meshio

def save_structured_grid(filename, controller, Z, save_time_series=False, rank=0):
    Z = Z.flatten()
    print(Z.shape)
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
                writer.write_data(sol.t, cell_data={"h": [q[0, ...]] + Z, "hu": [q[1, ...]], "hv": [q[2, ...]], "bed": [Z]})
        if rank == 0:
            writer.__exit__()
    else:
        if rank == 0:
            q = q.reshape((3, -1))
            meshio.write_points_cells(f"{filename}.vtk", verts, cells, cell_data={"h": [q[0, ...]], "hu": [q[1, ...]], "hv": [q[2, ...]]})


def bathymetry(x,y):
    r2 = (x-1.)**2 + (y-0.5)**2
    return 0.8*np.exp(-10*r2)

def qinit(state,h_in=1.0,h_out=0.5):
    x0=0.5
    y0=0.5
    sigma=0.5
    X, Y = state.p_centers
    r2 = (X-x0)**2 + (Y-y0)**2
    state.q[depth     ,:,:] = h_in*np.exp(-r2/sigma**2) + h_out
    state.q[x_momentum,:,:] = 0.
    state.q[y_momentum,:,:] = 0.

def setup(kernel_language='Fortran', solver_type='classic', riemann_solver='roe', use_petsc=False,
          outdir='./_output', resolution_x=150, resolution_y=150, extent=[0.,1.,0.,1.], period=0.15):

    solver = pyclaw.ClawSolver2D(riemann.shallow_bathymetry_fwave_2D)
    solver.dimensional_split = 1  # No transverse solver available

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.wall
    solver.bc_upper[1] = pyclaw.BC.wall

    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.extrap
    solver.aux_bc_upper[1] = pyclaw.BC.extrap

    xlower = extent[0]
    xupper = extent[1]
    mx = resolution_x
    ylower = extent[2]
    yupper = extent[3]
    my = resolution_y

    x = pyclaw.Dimension(xlower, xupper, mx, name='x')
    y = pyclaw.Dimension(ylower, yupper, my, name='y')
    domain = pyclaw.Domain([x,y])
    state = pyclaw.State(domain, num_eqn, num_aux=1)

    X, Y = state.p_centers
    Z = bathymetry(X,Y)
    Z = np.zeros_like(Z)
    state.aux[0,:,:] = Z

    qinit(state)

    state.problem_data['grav'] = 9.8
    state.problem_data['dry_tolerance'] = 1.e-3
    state.problem_data['sea_level'] = 0.

    claw = pyclaw.Controller()
    claw.tfinal = period
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = 10
    claw.keep_copy = True

    return claw, Z

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--resolution_x", default=100, type=int)
    parser.add_argument("--resolution_y", default=100, type=int)
    parser.add_argument("--period", default=1.5, type=float)
    parser.add_argument("--solver", default="sharpclaw")
    parser.add_argument("--riemann_solver", default="hlle")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--savets", action="store_true")
    args = parser.parse_args()

    xmin, xmax, ymin, ymax = 0., 1., 0., 1.
    extent = [xmin, xmax, ymin, ymax]
    controller, Z = setup(riemann_solver=args.riemann_solver, solver_type=args.solver, resolution_x=args.resolution_x, resolution_y=args.resolution_y, use_petsc=args.parallel, extent=extent, period=args.period)
    output = controller.run()

    path = os.path.dirname(os.path.realpath(__file__))
    f_name = f"{args.solver}-{args.riemann_solver}-{args.resolution_x}-{args.resolution_y}"
    fname = f"{path}/{f_name}"

    if args.parallel:
        from petsc4py import PETSc
        comm = PETSc.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0

    if args.savets:
        save_structured_grid(fname, controller, Z, save_time_series=args.savets, rank=rank)

    if rank == 0:
        print(output)
