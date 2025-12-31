import matplotlib.pyplot as plt
import pandas as pd
import csv

import numpy as xp

def comparison(fluctuation_MD, fluctuation_CG, traj2, traj5, no_of_bonds, N, no_frames_CG, bo_md, bo_cg, path):

    df = pd.read_csv(path + '/POST_PROCESSING/K_values_final.csv', header=None)
    bond_list = xp.array(df)

    ########################### bond fluctuation plot ########################################
    plt.rcParams.update({'font.size': 60})  
    plt.figure(figsize=(60, 20))
    res = xp.arange(1, len(fluctuation_MD) + 1, 1)

    # plotting the points 
    plt.plot(res, fluctuation_MD, linewidth=7)
    plt.plot(res, fluctuation_CG, linewidth=7)

    # naming the x axis
    plt.xlabel('BOND IJ')
    # naming the y axis
    plt.ylabel('FLUCTUATION')
    # giving a title to my graph
    plt.title('BOND FLUCTUATIONS')
    plt.savefig(path + '/OUTPUT/BOND_FLUCTUTION.eps')
    # function to show the plot
    plt.show()

    return  # rest of the function is commented out and left untouched
