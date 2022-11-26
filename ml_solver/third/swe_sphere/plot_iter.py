import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import xarray as xr
from tqdm import tqdm
import numpy as np

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# create OpenCV video writer
video = cv2.VideoWriter('training_data_h.mp4', cv2.VideoWriter_fourcc('a','v','c','1'), 1, (1440,720))

ds = xr.open_zarr("/scratch/lhuang/SWE/training_data/sphere_torch_2000_8x_randvel_iter.zarr")

# loop over your images
for i in tqdm(range(2560)): #ds.q.shape[0]

    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    plt.imshow(ds.q[i, 0, :, :].T, cmap="coolwarm")

    # put pixel buffer in numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)

    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    # write frame to video
    video.write(mat)

    plt.close('all')

# close video writer
cv2.destroyAllWindows()
video.release()