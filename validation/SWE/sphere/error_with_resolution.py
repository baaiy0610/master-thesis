from cProfile import label
from curses import noecho
from email import header
from fileinput import filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pytools import P

pathdirs = ["rusanov", "roe", "hll", "hlle", "hllc"]
markers = [".", "s", "o","_", "+"]
colors = ["b","k","g","r","y"]

error_norms = ["L1(error)", "L2(error)", "Linf(error)"]



path = os.path.dirname(os.path.realpath(__file__))
resolution = 2.0 / np.array([64, 128, 256, 562 ,1024])

for error_norm in error_norms:
    plt.figure(figsize=(8, 6))

    for i in range(len(pathdirs)):
        error = []
        filenames = ["dace-128-64-claw-1000-500.xlsx",  "dace-256-128-claw-1000-500.xlsx", "dace-512-256-claw-1000-500.xlsx", "dace-1024-512-claw-1000-500.xlsx", "dace-2048-1024-claw-1000-500.xlsx"]
        for filename in filenames:
            df = pd.read_excel(f"{path}/{pathdirs[i]}/{filename}")
            error.append(df[error_norm][0])
        error = np.array(error)
        plt.loglog(resolution, error, label=pathdirs[i], marker=markers[i], color=colors[i]) 
        slope, intercept = np.polyfit(np.log(resolution), np.log(error), 1)
        # slope = (np.log(error[-1])-np.log(error[0]))/(np.log(resolution[-1])-np.log(resolution[0]))
        print(slope, pathdirs[i])

    ax=plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # if error_norm == "L2(error)":
    #     ax.set_xlim(10e-2, 1e-3)
    #     ax.set_ylim(10e-7, 1e-9)
    # else:
    ax.set_xlim(10e-2, 1e-3)
    ax.set_ylim(10e-5, 1e-6)

    # ax.set_adjustable("datalim")
    ax.set_aspect("equal")

    plt.xlabel("resolution")
    plt.ylabel(f"h {error_norm}")
    plt.title(f"resolution vs h {error_norm}")
    plt.legend()
    plt.savefig(f'{error_norm}.png', dpi=100)
    # plt.show()
