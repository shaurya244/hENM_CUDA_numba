import numpy as xp

from derivative_hessian_updated import derivative_hessian  # to calculate the partial derivative of the hessian
from func_force_field_determination import NMA

# k represents the kth eigenvalue whose derivative we are calculating
# i, j represent the spring constant K_ij with respect to which the derivative is taken
def deri_eign_val(i, j, eign_vec, k, traj4, mass_weights, bond_list):
    eigenvector_k = eign_vec.real[:, k]
    v_ij = eigenvector_k[xp.r_[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]].reshape((6, 1))  # [6x1] column vector
    deri_H = derivative_hessian(bond_list, i, j, traj4, mass_weights)  # [6x6]

    # Matrix multiplication: [1x6] x [6x6] x [6x1] → scalar (1x1)
    val = (v_ij.T @ deri_H @ v_ij)  # extract scalar from 1x1 matrix
    return val


# k represents the eigenvector index whose partial derivative we want
# i,j = bond for which we're taking the derivative
def deri_eign_vec(eign_vec, eign_val, i, j, k, traj4, mass_weights, bond_list, N):
    deri_eign_vec = xp.zeros((3 * N, 1)).astype(xp.complex128)
    deri_H = derivative_hessian(bond_list, i, j, traj4, mass_weights)
    for r in range(0, 3 * N - 6):
        if r == k:
            deri_eign_vec += 0 * eign_vec.real[:, r].reshape((3 * N, 1))
        else:
            a_kij_r = coff_k_ij_r(i, j, eign_val.real[r], eign_val.real[k],
                                  eign_vec.real[:, r].reshape((3 * N, 1)),
                                  eign_vec.real[:, k].reshape((3 * N, 1)),
                                  deri_H, k)
            deri_eign_vec += a_kij_r * eign_vec[:, r].reshape((3 * N, 1))
    return deri_eign_vec


def coff_k_ij_r(i, j, eign_val_r, eign_val_k, eign_vec_r, eign_vec_k, deri_H, k):
    v_r_i = eign_vec_r[3 * i : 3 * i + 3]  # shape (3,1)
    v_r_j = eign_vec_r[3 * j : 3 * j + 3]  # shape (3,1)
    v_r_ij = xp.concatenate((v_r_i, v_r_j))  # shape (6,1)

    v_k_i = eign_vec_k[3 * i : 3 * i + 3]  # shape (3,1)
    v_k_j = eign_vec_k[3 * j : 3 * j + 3]  # shape (3,1)
    v_k_ij = xp.concatenate((v_k_i, v_k_j))  # shape (6,1)

    if eign_val_k - eign_val_r == 0:
        a_k_ij_r = 0
    else:
        a_k_ij_r = (1 / (eign_val_k - eign_val_r)) * (v_r_ij.T @ deri_H @ v_k_ij)

    return a_k_ij_r
