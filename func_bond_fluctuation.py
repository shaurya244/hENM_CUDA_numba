import gc 
import math                           
import numpy as xp

from scipy.stats import norm
import matplotlib.pyplot as plt
import func_fas_identical_bonds_for_any_quantity
import func_mspa_identical_bonds_for_any_quantity

def bond_fluctuations(traj, no_frames, no_of_bonds, flag_identical_bonds, path, system, bond_list):

    mean = xp.zeros((no_of_bonds))
    variance = xp.zeros((no_of_bonds))
    std_dev = xp.zeros((no_of_bonds))

    for ids, bond in enumerate(bond_list):
        j, m = int(bond[0]) - 1, int(bond[1]) - 1

        avg = math.sqrt(xp.sum((traj.xyz[0, j, :] - traj.xyz[0, m, :]) ** 2))
        ssq = xp.sum((traj.xyz[0, j, :] - traj.xyz[0, m, :]) ** 2)

        for k in range(2, no_frames + 1):
            avg = (avg * (k - 1) + math.sqrt(xp.sum((traj.xyz[k - 1, j, :] - traj.xyz[k - 1, m, :]) ** 2))) / k
            ssq = ssq + xp.sum((traj.xyz[k - 1, j, :] - traj.xyz[k - 1, m, :]) ** 2)

        mean[ids] = avg
        variance[ids] = ((ssq / (no_frames - 1)) - (avg ** 2) * ((no_frames) / (no_frames - 1)))
        std_dev[ids] = math.sqrt(variance[ids])

    if flag_identical_bonds == 'yes':
        if system == 'mspa':  
            variance = func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(variance, path) 
        if system == 'fas':
            variance = func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(variance, path)

    return variance, mean, std_dev
