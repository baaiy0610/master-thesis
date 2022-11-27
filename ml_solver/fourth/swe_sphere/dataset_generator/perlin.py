#https://github.com/caseman/noise/blob/bb32991ab97e90882d0e46e578060717c5b90dc5/_simplex.c


from math import floor, fmod, sqrt
from random import randint
import numba
from numba import njit
import numpy as np

# 3D Gradient vectors
_GRAD3 = ((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0), 
	(1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1), 
	(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
	(1,1,0),(0,-1,1),(-1,1,0),(0,-1,-1),
) 

# 4D Gradient vectors
_GRAD4 = ((0,1,1,1), (0,1,1,-1), (0,1,-1,1), (0,1,-1,-1), 
	(0,-1,1,1), (0,-1,1,-1), (0,-1,-1,1), (0,-1,-1,-1), 
	(1,0,1,1), (1,0,1,-1), (1,0,-1,1), (1,0,-1,-1), 
	(-1,0,1,1), (-1,0,1,-1), (-1,0,-1,1), (-1,0,-1,-1), 
	(1,1,0,1), (1,1,0,-1), (1,-1,0,1), (1,-1,0,-1), 
	(-1,1,0,1), (-1,1,0,-1), (-1,-1,0,1), (-1,-1,0,-1), 
	(1,1,1,0), (1,1,-1,0), (1,-1,1,0), (1,-1,-1,0), 
	(-1,1,1,0), (-1,1,-1,0), (-1,-1,1,0), (-1,-1,-1,0))

# A lookup table to traverse the simplex around a given point in 4D. 
# Details can be found where this table is used, in the 4D noise method. 
_SIMPLEX = (
	(0,1,2,3),(0,1,3,2),(0,0,0,0),(0,2,3,1),(0,0,0,0),(0,0,0,0),(0,0,0,0),(1,2,3,0), 
	(0,2,1,3),(0,0,0,0),(0,3,1,2),(0,3,2,1),(0,0,0,0),(0,0,0,0),(0,0,0,0),(1,3,2,0), 
	(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0), 
	(1,2,0,3),(0,0,0,0),(1,3,0,2),(0,0,0,0),(0,0,0,0),(0,0,0,0),(2,3,0,1),(2,3,1,0), 
	(1,0,2,3),(1,0,3,2),(0,0,0,0),(0,0,0,0),(0,0,0,0),(2,0,3,1),(0,0,0,0),(2,1,3,0), 
	(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0), 
	(2,0,1,3),(0,0,0,0),(0,0,0,0),(0,0,0,0),(3,0,1,2),(3,0,2,1),(0,0,0,0),(3,1,2,0), 
	(2,1,0,3),(0,0,0,0),(0,0,0,0),(0,0,0,0),(3,1,0,2),(0,0,0,0),(3,2,0,1),(3,2,1,0))

# Simplex skew constants
_F2 = 0.5 * (sqrt(3.0) - 1.0)
_G2 = (3.0 - sqrt(3.0)) / 6.0
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0

@njit("double(double, double, double, int64[:])")
def _snoise3(x, y, z, permutation):
    """3D Perlin simplex noise. 
    
    Return a floating point value from -1 to 1 for the given x, y, z coordinate. 
    The same value is always returned for a given x, y, z pair unless the
    permutation table changes (see randomize above).
    """
    period = len(permutation) // 2
    # Skew the input space to determine which simplex cell we're in
    s = (x + y + z) * _F3
    i = floor(x + s)
    j = floor(y + s)
    k = floor(z + s)
    t = (i + j + k) * _G3
    x0 = x - (i - t) # "Unskewed" distances from cell origin
    y0 = y - (j - t)
    z0 = z - (k - t)

    # For the 3D case, the simplex shape is a slightly irregular tetrahedron. 
    # Determine which simplex we are in. 
    if x0 >= y0:
        if y0 >= z0:
            i1 = 1; j1 = 0; k1 = 0
            i2 = 1; j2 = 1; k2 = 0
        elif x0 >= z0:
            i1 = 1; j1 = 0; k1 = 0
            i2 = 1; j2 = 0; k2 = 1
        else:
            i1 = 0; j1 = 0; k1 = 1
            i2 = 1; j2 = 0; k2 = 1
    else: # x0 < y0
        if y0 < z0:
            i1 = 0; j1 = 0; k1 = 1
            i2 = 0; j2 = 1; k2 = 1
        elif x0 < z0:
            i1 = 0; j1 = 1; k1 = 0
            i2 = 0; j2 = 1; k2 = 1
        else:
            i1 = 0; j1 = 1; k1 = 0
            i2 = 1; j2 = 1; k2 = 0
    
    # Offsets for remaining corners
    x1 = x0 - i1 + _G3
    y1 = y0 - j1 + _G3
    z1 = z0 - k1 + _G3
    x2 = x0 - i2 + 2.0 * _G3
    y2 = y0 - j2 + 2.0 * _G3
    z2 = z0 - k2 + 2.0 * _G3
    x3 = x0 - 1.0 + 3.0 * _G3
    y3 = y0 - 1.0 + 3.0 * _G3
    z3 = z0 - 1.0 + 3.0 * _G3

    # Calculate the hashed gradient indices of the four simplex corners
    perm = permutation
    ii = int(i) % period
    jj = int(j) % period
    kk = int(k) % period
    gi0 = perm[ii + perm[jj + perm[kk]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

    # Calculate the contribution from the four corners
    noise = 0.0
    tt = 0.6 - x0**2 - y0**2 - z0**2
    if tt > 0:
        g = _GRAD3[gi0]
        noise = tt**4 * (g[0] * x0 + g[1] * y0 + g[2] * z0)
    else:
        noise = 0.0
    
    tt = 0.6 - x1**2 - y1**2 - z1**2
    if tt > 0:
        g = _GRAD3[gi1]
        noise += tt**4 * (g[0] * x1 + g[1] * y1 + g[2] * z1)
    
    tt = 0.6 - x2**2 - y2**2 - z2**2
    if tt > 0:
        g = _GRAD3[gi2]
        noise += tt**4 * (g[0] * x2 + g[1] * y2 + g[2] * z2)
    
    tt = 0.6 - x3**2 - y3**2 - z3**2
    if tt > 0:
        g = _GRAD3[gi3]
        noise += tt**4 * (g[0] * x3 + g[1] * y3 + g[2] * z3)
    
    return noise * 32.0

@njit("void(double[:,::1], double[:,:,::1])")
def noise3_vec(out, coords):
    _, M, N = coords.shape
    permutation = np.random.permutation(499).astype(np.int64)
    permutation = np.concatenate((permutation, permutation))
    for i in range(M):
        for j in range(N):
            coord = coords[:, i, j]
            out[i, j] = _snoise3(coord[0], coord[1], coord[2], permutation)

if __name__ == "__main__":
    import MeshGrid
    import matplotlib.pyplot as plt
    x_range = (-3., 1.)
    y_range = (-1., 1.)
    r = 1.0
    M = 2000
    N = 1000
    _, _, _, cell_center_3d = MeshGrid.sphere_grid(x_range, y_range, M, N, r)
    out = np.zeros((M, N))
    coords = cell_center_3d.numpy()
    noise3_vec(out, coords)
    plt.imshow(out.T)
    plt.colorbar()
    plt.savefig("perlin.png")