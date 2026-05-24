import numpy as xp

from func_force_field_determination import NMA
from func_deri_eigen_val_vec import deri_eign_val
from func_deri_eigen_val_vec import deri_eign_vec
from jacobian import get_jacobian
import matplotlib.pyplot as plt
from func_error import NMA_error
from scipy.linalg import solve
from joblib import Parallel, delayed
from derivative_hessian_updated import derivative_hessian
from gpu_eigen_value_derivative import eigenvalue_jacob_gpu
from gpu_eigen_vector_derivative import eigenvector_jacobian_gpu
from func_error import NMA_fluctuations

def force_constant_jacobian(bond_list,N,fluctuation_MD,traj4,T,mass_weights,max_itr):
    K_new = xp.zeros((len(bond_list), 1), dtype=xp.float64)
    K_new_test = xp.zeros((len(bond_list), 1), dtype=xp.float64)
    v_old,w_old,error_old ,hessian= NMA(N,bond_list,  fluctuation_MD, T, traj4, mass_weights)
    print("w_old:", w_old)
    k = 0
    w_new = xp.zeros((w_old.shape[0],1))
    v_new = xp.zeros(v_old.shape, dtype=xp.complex128)
    # sq_error_old holds the signed error (fluct_NMA - fluct_MD), NOT squared
    sq_error_old = error_old
    SUM_OF_SQUARE_ERROR = []
    SUM_of_sq_error_nma = []
    SUM_OF_SQUARE_ERROR.append(xp.sum(error_old**2))
    SUM_of_sq_error_nma.append(xp.sum(error_old**2))
    itr = []
    itr.append(k)
    M = len(bond_list)
    deri_H_per_bond = xp.zeros((M, 6, 6), dtype=xp.float64)
    for idx in range(M):
        d = int(bond_list[idx, 0] - 1)
        f = int(bond_list[idx, 1] - 1)
        H6 = derivative_hessian(bond_list, d, f, traj4, mass_weights)  # must be numeric 6x6
        deri_H_per_bond[idx] = xp.array(H6, dtype=xp.float64)
    while k < max_itr:
        # get_jacobian returns J where J[m,n] = d(fluct_NMA_m)/d(K_n)
        J_fluct = get_jacobian(bond_list, N, v_old, w_old, traj4, mass_weights, fluctuation_MD, T, deri_H_per_bond)
        J_fluct = xp.real(xp.asarray(J_fluct))   # (M, M) real

        # Bug 2 fix: form d(sq_error_m)/d(K_n) = 2 * error_m * d(fluct_NMA_m)/d(K_n)
        # sq_error_old is the signed error vector (fluct_NMA - fluct_MD)
        Jacobian = 2.0 * xp.diag(sq_error_old) @ J_fluct   # shape (M, M)

        # Regularise to avoid singular matrix when some errors are near zero
        M_bonds = len(bond_list)
        eps = 1e-8 * xp.max(xp.abs(xp.diag(Jacobian))) if xp.any(xp.diag(Jacobian) != 0) else 1e-8
        Jacobian_reg = Jacobian + eps * xp.eye(M_bonds)
        pertub = xp.squeeze(solve(Jacobian_reg, (sq_error_old**2).reshape((M_bonds, 1))))

        # Bug 5 fix: start with beta=1 (pure Newton); clamp to avoid huge steps
        beta = 1.0
        K_new = bond_list[:,2] - beta * pertub
        K_new = xp.asarray(K_new, dtype=xp.complex128)
        K_new = xp.asarray(xp.real(K_new), dtype=xp.float64)
        K_new[:] = xp.where(K_new < 0, 1e-2, K_new)
        bond_list[:, 2] = K_new
        k = k + 1

        # Bug 4 & 6 fix: recompute exact NMA and use those eigenpairs/error for next iteration
        v_new_nma, w_new_nma, error_new_nma, hessian = NMA(N, bond_list, fluctuation_MD, T, traj4, mass_weights)
        v_old = v_new_nma
        w_old = w_new_nma
        sq_error_old = error_new_nma   # true error, not the linearised estimate

        print("sum_sq_error_nma:", xp.sum(error_new_nma**2))
        SUM_OF_SQUARE_ERROR.append(xp.sum(sq_error_old**2))
        SUM_of_sq_error_nma.append(xp.sum(error_new_nma**2))
        itr.append(k)

    plt.figure()
    # plt.plot(itr, SUM_OF_SQUARE_ERROR, linestyle='solid')
    plt.plot(itr, SUM_of_sq_error_nma, linestyle='dashed')
    plt.xlabel('iteration')
    plt.ylabel('sum_sq_error')
    plt.title('Covergence of Force Constant')

    return bond_list    
# this function calculate the derivative of every eigen value with respect to every parameter (spring constant)
#so this used as the derivative to multiply while updating the eigen values 
#  lambda_new = lambda_old + eigenvalue_jacob @ difference between the old and new spring constants

def compute_eigen_val_row_cpu(m, bond_list, N, traj4, mass_weights, v_old):
    row = xp.zeros(len(bond_list), dtype=xp.complex128)
    for n in range(len(bond_list)):
        i = int(bond_list[n][0]) - 1
        j = int(bond_list[n][1]) - 1
        row[n] = deri_eign_val(i, j, v_old, m, traj4, mass_weights, bond_list)
    return m, row


def eigenvalue_jacob(bond_list, N, traj4, mass_weights, v_old):
    M = 3 * N - 6

    results = Parallel(n_jobs=-1)(
        delayed(compute_eigen_val_row_cpu)(m, bond_list, N, traj4, mass_weights, v_old)
        for m in range(M)
    )

    jacobian = xp.zeros((M, len(bond_list)), dtype=xp.complex128)
    for m, row in results:
        jacobian[m, :] = row

    return jacobian
# this function calculate the derivative of one eigen vector with respect to every parameter (spring constant)
#so this used as the derivative to multiply while updating the eigen vector in 
# vector_new(i) = vector_old(i) - eigenvector_jacob @ difference between the old and new spring constants
# def eigenvector_jacob(v_old, w_old, k, traj4, mass_weights, bond_list, N):
#     eigen_vec_jacob_k = xp.zeros((3 * N, len(bond_list)), dtype=xp.complex128)
#     for n in range(eigen_vec_jacob_k.shape[1]):
#         i = int(bond_list[n][0]) - 1
#         j = int(bond_list[n][1]) - 1
#         result = deri_eign_vec(v_old, w_old, i, j, k, traj4, mass_weights, bond_list, N)
#         eigen_vec_jacob_k[:, n] = xp.asarray(xp.squeeze(result))  # Ensure shape and type match
#     return eigen_vec_jacob_k


def compute_eigenvector_column(n, v_old, w_old, k, traj4, mass_weights, bond_list, N):
    i = int(bond_list[n][0]) - 1
    j = int(bond_list[n][1]) - 1

    # Compute with your bottleneck
    result = deri_eign_vec(v_old, w_old, i, j, k, traj4, mass_weights, bond_list, N)

    # Convert result to CPU numpy (joblib requires picklable CPU objects)
    col = xp.asarray(xp.squeeze(result))

    return n, col
def eigenvector_jacob(v_old, w_old, k, traj4, mass_weights, bond_list, N):
    M = 3 * N
    B = len(bond_list)

    # Launch parallel jobs (one per column)
    results = Parallel(n_jobs=-1)(
        delayed(compute_eigenvector_column)(
            n, v_old, w_old, k, traj4, mass_weights, bond_list, N
        )
        for n in range(B)
    )

    # Allocate final Jacobian (NumPy on CPU)
    eigen_vec_jacob_k = xp.zeros((M, B), dtype=xp.complex128)

    # Fill the columns
    for n, col in results:
        eigen_vec_jacob_k[:, n] = col

    return eigen_vec_jacob_k
