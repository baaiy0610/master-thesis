import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


pathdirs = ["rusanov", "roe", "hll", "hlle", "hllc"]
# pathdirs = ["claw"]
markers = [".", "o", "*", "+", "x"]
colors = ["b", "g", "r", "y", "m"]

error_norms = ["L1(error)", "L2(error)", "Linf(error)"]
orders = ["1order", "2order"]


path = os.path.dirname(os.path.realpath(__file__))
resolution = 1 / np.array([100, 200, 400, 800])

for order in orders:
    for error_norm in error_norms:
        print(order, error_norm)
        plt.figure(figsize=(8, 6))

        for i in range(len(pathdirs)):
            error = []
            filenames = ["torch-100-100-claw-1000-1000.xlsx",  "torch-200-200-claw-1000-1000.xlsx", "torch-400-400-claw-1000-1000.xlsx", "torch-800-800-claw-1000-1000.xlsx"]
            for filename in filenames:
                df = pd.read_excel(f"{path}/{order}/{pathdirs[i]}/{filename}")
                error.append(df[error_norm][0])
            error = np.array(error)
            plt.loglog(resolution, error, label=pathdirs[i], marker=markers[i], color=colors[i]) 
            # slope, intercept = np.polyfit(np.log(resolution), np.log(error), 1)
            slope = (np.log(error[-1])-np.log(error[0]))/(np.log(resolution[-1])-np.log(resolution[0]))
            print(slope, pathdirs[i])

        ax=plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # if error_norm == "L2(error)":
        #     ax.set_xlim(10e-2, 1e-3)
        #     ax.set_ylim(10e-7, 1e-9)
        # else:
        # ax.set_xlim(10e-2, 1e-3)
        # ax.set_ylim(10e-5, 1e-6)

        # ax.set_adjustable("datalim")
        ax.set_aspect("equal")

        plt.xlabel("resolution")
        plt.ylabel(f"h {error_norm}")
        plt.title(f"resolution vs h {error_norm}")
        plt.legend()
        plt.savefig(f'{error_norm}-{order}.png', dpi=100)
        # plt.show()
