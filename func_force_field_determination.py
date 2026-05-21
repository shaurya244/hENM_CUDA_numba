import gc
import math
import numpy as xp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import func_fas_identical_bonds_for_any_quantity
import func_lammps_data
import func_mspa_identical_bonds_for_any_quantity
from lammps import lammps
import mdtraj as md
import csv


def NMA(N, bond_list, fluctuation_MD, T, traj4, mass_weights):
    hessian = xp.zeros((3 * N, 3 * N))

    for ids, bond in enumerate(bond_list):
        i, j = int(bond[0]) - 1, int(bond[1]) - 1
        for k in range(3):
            for l in range(3):
                hij = (
                    (-bond_list[ids][2])
                    / (
                        (bond_list[ids][3] ** 2)
                        * math.sqrt(mass_weights[i])
                        * math.sqrt(mass_weights[j])
                    )
                    * (traj4.xyz[0, i, k] - traj4.xyz[0, j, k])
                    * (traj4.xyz[0, i, l] - traj4.xyz[0, j, l])
                )
                hessian[3 * i + k, 3 * j + l] = hij
                hessian[3 * j + l, 3 * i + k] = hij
    # print("off diagonal")

    for i in range(N):
        for k in range(3):
            for l in range(3):
                value = sum(
                    hessian[3 * i + k, 3 * j + l]
                    for j in range(N)
                    if i != j
                )
                hessian[3 * i + k, 3 * i + l] = -value
    # print("diagonal")

    # Use NumPy's eig (CuPy's linalg.eig has compatibility issues with all eigenvectors/eigenvalues on all platforms)
    w, v = LA.eig(hessian)
 

    # Filter out small eigenvalues (translations + rotations)
    v_edit = []
    w_edit = []
    for i in range(len(w)):
        if w[i] > 1e-5:
            v_edit.append(v[:, i])
            w_edit.append(w[i])
    w = xp.array(w_edit)
    v = xp.array(v_edit).T
    del w_edit
    del v_edit

    if len(w) != 3 * N - 6:
        print("⚠️ More than 6 eigenvalues were filtered")

    kB = 8.314462618 * 0.001  # kJ/mol·K
    fluctuation_NMA = xp.zeros(len(bond_list))

    for ids, bond in enumerate(bond_list):
        i, j = int(bond[0]) - 1, int(bond[1]) - 1
        DELTA_i_j = 0
        for k in range(3 * N - 6):
            delta = (
                (traj4.xyz[0, i, 0] - traj4.xyz[0, j, 0])
                * ((v[3 * i, k].real / math.sqrt(mass_weights[i])) - (v[3 * j, k].real / math.sqrt(mass_weights[j])))
                + (traj4.xyz[0, i, 1] - traj4.xyz[0, j, 1])
                * ((v[3 * i + 1, k].real / math.sqrt(mass_weights[i])) - (v[3 * j + 1, k].real / math.sqrt(mass_weights[j])))
                + (traj4.xyz[0, i, 2] - traj4.xyz[0, j, 2])
                * ((v[3 * i + 2, k].real / math.sqrt(mass_weights[i])) - (v[3 * j + 2, k].real / math.sqrt(mass_weights[j])))
            )
            DELTA_i_j += (1 / math.sqrt(w[k].real)) * delta**2

        fluctuation_NMA[ids] = kB * T * DELTA_i_j * (1 / bond_list[ids][3]) ** 2

    error = xp.zeros(len(bond_list))
    for ids, bond in enumerate(bond_list):
        error[ids] = fluctuation_NMA[ids] - fluctuation_MD[ids]

    gc.collect()
    del fluctuation_NMA

    return v, w, error, hessian


def force_constants(N, T, ALPHA, beta, max_itr, bond_list, fluctuation_MD, run, tolerance, path, flag_identical_bonds,
                    flag_automate_ALPHA_beta_values, iterative_method, system, traj4, box_size, mass_weights):

    SUM_OF_SQUARE_ERROR = []
    STEP = []
    itr = []
    iteration = 0
    start_SOE = 0

    sourceFile = open(path + '/POST_PROCESSING/array_kzeros%s.txt' % run, 'w')

    for iteration in range(max_itr):
        print("Iteration:", iteration)
        if flag_automate_ALPHA_beta_values == 'yes':
            ALPHA = 1 / (max_itr + 1)
            beta = 1 / (max_itr + 1)

        bond_list_incre = xp.zeros((len(bond_list), 4))
        for ids, bond in enumerate(bond_list):
            bond_list_incre[ids, 0] = bond[0]
            bond_list_incre[ids, 1] = bond[1]
            bond_list_incre[ids, 2] = bond[2] + ALPHA
            bond_list_incre[ids, 3] = bond[3]

        print("NMA_calculation1")
        v_, w_, error_ksp = NMA(N, bond_list, fluctuation_MD, T, traj4, mass_weights)
        print("NMA_calculation2")
        v_, w_, error_ksp_incre = NMA(N, bond_list_incre, fluctuation_MD, T, traj4, mass_weights)

        K_new = xp.zeros(len(bond_list))
        square_of_error = xp.zeros(len(bond_list))

        for ids, bond in enumerate(bond_list):
            if iterative_method == 'NR_method':
                K_new[ids] = bond[2] - (ALPHA * beta * error_ksp[ids] ** 2) / (
                        error_ksp_incre[ids] ** 2 - error_ksp[ids] ** 2)
            elif iterative_method == 'Chu_and_Voth_method':
                K_new[ids] = bond[2] + ALPHA * error_ksp[ids]
            elif iterative_method == 'Lyman_method':
                inv_K = 1 / bond[2] - ALPHA * error_ksp[ids]
                K_new[ids] = 1 / inv_K

            square_of_error[ids] = error_ksp[ids] ** 2

        del error_ksp
        del error_ksp_incre

        step = 0
        for ids, bond in enumerate(bond_list):
            if bond[2] > 0:
                step += abs(K_new[ids] - bond[2]) / bond[2]
        STEP.append(step)
        sum_of_sq_error = xp.sum(square_of_error)
        SUM_OF_SQUARE_ERROR.append(sum_of_sq_error.item())
        itr.append(iteration)

        plt.plot(itr, SUM_OF_SQUARE_ERROR, linestyle='solid')
        plt.xlabel('iteration')
        plt.ylabel('sum of square of error')
        plt.title('CONVERGENCE PLOT')

        xp.savetxt(path + '/POST_PROCESSING/STEP%s.txt' % run, STEP)

        for ids, bond in enumerate(bond_list):
            bond_list[ids][2] = K_new[ids].item()

        if flag_identical_bonds == "yes":
            if system == 'mspa':
                k_sp = func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(bond_list, path)
            elif system == 'fas':
                k_sp = func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(bond_list, path)

        if iteration % 100 == 0:
            xp.savetxt(path + '/POST_PROCESSING/K_values_%d.csv' % iteration, bond_list, delimiter=",")
            plt.plot(itr, STEP, linestyle='solid')
            plt.xlabel('iteration')
            plt.ylabel('STEP')
            plt.title('SPRING CONSTANT CONVERGENCE PLOT')

        counting = 0
        for ids, bond in enumerate(bond_list):
            i, j = int(bond[0]) - 1, int(bond[1]) - 1
            if bond[2] <= 0:
                bond[2] = 0
                counting += 1
                print(i, j, iteration, file=sourceFile)

        print("NUMBER OF k's equal to zero:", counting)

        bonds_k_non_zeros = []
        for ids, bond in enumerate(bond_list):
            i, j = int(bond[0]) - 1, int(bond[1]) - 1
            if bond[2] > 0:
                bonds_k_non_zeros.append([i + 1, j + 1])

        if iteration % 100 == 0:
            with open(path + '/POST_PROCESSING/bonds_k_non_zeros_%d.csv' % iteration, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(bonds_k_non_zeros)

        if sum_of_sq_error < tolerance or abs(sum_of_sq_error - start_SOE) < tolerance:
            print("Converged. Final sum of square of error:", sum_of_sq_error)
            break

        start_SOE = sum_of_sq_error
        gc.collect()

        func_lammps_data.lammps_data_file(N, bond_list, mass_weights, traj4, box_size, path, system, flag_identical_bonds)
        print("Running LAMMPS...")
        lmp = lammps()
        lmp.file(path + "/INPUT/energy_min.in")
        lmp.command("log log_%d.lammps")
        print("LAMMPS finished.")

        traj4 = md.load_lammpstrj(path + '/POST_PROCESSING/em.lammpstrj', top=path + '/INPUT/REFERENCE.pdb')
        traj4.xyz[:, :, :] *= 10

        if flag_identical_bonds == 'no':
            for ids, bond in enumerate(bond_list):
                i, j = int(bond[0]) - 1, int(bond[1]) - 1
                bond_list[ids][3] = math.sqrt(xp.sum((traj4.xyz[0, i, :] - traj4.xyz[0, j, :]) ** 2))

        elif flag_identical_bonds == 'yes':
            dr_sq1 = xp.zeros((N, N))
            for bond in bond_list:
                i, j = int(bond[0]) - 1, int(bond[1]) - 1
                dr_sq1[i][j] = math.sqrt(xp.sum((traj4.xyz[0, i, :] - traj4.xyz[0, j, :]) ** 2))

    return bond_list
