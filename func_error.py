import numpy as xp

import math

def NMA_error(bond_list, traj4, mass_weights, fluctuation_MD, T, N, v, w):
    fluctuation_NMA = xp.zeros((len(bond_list)))  # NMA FLUCTUATION MATRIX
    kB = 8.314462618 * 0.001  # kJ/mol K

    for ids, bond in enumerate(bond_list):
        i, j = int(bond[0]) - 1, int(bond[1]) - 1  
        DELTA_i_j = 0

        for k in range(0, (3 * N) - 6):
            DELTA_i_j += (1 / math.sqrt(abs(w.real[k]))) * (
                (traj4.xyz[0, i, 0] - traj4.xyz[0, j, 0]) * ((v.real[3 * i][k] / math.sqrt(mass_weights[i])) - (v.real[3 * j][k] / math.sqrt(mass_weights[j]))) +
                (traj4.xyz[0, i, 1] - traj4.xyz[0, j, 1]) * ((v.real[3 * i + 1][k] / math.sqrt(mass_weights[i])) - (v.real[3 * j + 1][k] / math.sqrt(mass_weights[j]))) +
                (traj4.xyz[0, i, 2] - traj4.xyz[0, j, 2]) * ((v.real[3 * i + 2][k] / math.sqrt(mass_weights[i])) - (v.real[3 * j + 2][k] / math.sqrt(mass_weights[j])))
            ) ** 2

        fluctuation_NMA[ids] = kB * T * DELTA_i_j * ((1 / bond_list[ids][3]) ** 2)

    error = xp.zeros((len(bond_list)))
    for ids, bond in enumerate(bond_list):
        i, j = int(bond[0]) - 1, int(bond[1]) - 1
        error[ids] = fluctuation_NMA[ids] - fluctuation_MD[ids]

    return error
def NMA_fluctuations(bond_list, traj4, mass_weights, T, N, v, w):
    fluctuation_NMA = xp.zeros((len(bond_list)))  # NMA FLUCTUATION MATRIX
    kB = 8.314462618 * 0.001  # kJ/mol K

    for ids, bond in enumerate(bond_list):
        i, j = int(bond[0]) - 1, int(bond[1]) - 1  
        DELTA_i_j = 0

        for k in range(0, (3 * N) - 6):
            DELTA_i_j += (1 / math.sqrt(abs(w.real[k]))) * (
                (traj4.xyz[0, i, 0] - traj4.xyz[0, j, 0]) * ((v.real[3 * i][k] / math.sqrt(mass_weights[i])) - (v.real[3 * j][k] / math.sqrt(mass_weights[j]))) +
                (traj4.xyz[0, i, 1] - traj4.xyz[0, j, 1]) * ((v.real[3 * i + 1][k] / math.sqrt(mass_weights[i])) - (v.real[3 * j + 1][k] / math.sqrt(mass_weights[j]))) +
                (traj4.xyz[0, i, 2] - traj4.xyz[0, j, 2]) * ((v.real[3 * i + 2][k] / math.sqrt(mass_weights[i])) - (v.real[3 * j + 2][k] / math.sqrt(mass_weights[j])))
            ) 

        fluctuation_NMA[ids] = DELTA_i_j 


    return fluctuation_NMA
