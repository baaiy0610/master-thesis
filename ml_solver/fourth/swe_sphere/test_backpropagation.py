import torch
import torch.nn as nn
import torch.nn.functional as F
from SWE_sphere import boundary_condition, g, limiter, flux_roe, fvm_2ndorder_space_classic, initial_condition
import SWE_sphere, MeshGrid
SWE_sphere.flux = flux_roe


M = 400
N = 200
x_range = (-3., 1.)
y_range = (-1., 1.)
r = 1.0
nhalo = 3
dt = 0.001*(128/max(N, M/2))
vert_3d, vert_lonlat, cell_area, cell_center_3d = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
q_init = initial_condition(x_range, y_range, M, N, cell_center_3d)
q_init = nn.Parameter(q_init, requires_grad=True)
q0 = boundary_condition(q_init, nhalo)
q1 = fvm_2ndorder_space_classic(q0, vert_3d, cell_center_3d, cell_area, dt, r, nhalo)
loss = (q1[:, nhalo:-nhalo, nhalo:-nhalo] - q0[:, nhalo:-nhalo, nhalo:-nhalo]).square().mean()
loss.backward()
print(loss)
print(q_init.grad)