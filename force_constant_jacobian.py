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
    v_old,w_old,error_old = NMA(N,bond_list,  fluctuation_MD, T, traj4, mass_weights)
    k = 0
    w_new = xp.zeros((w_old.shape[0],1))
    v_new = xp.zeros(v_old.shape, dtype=xp.complex128)
    sq_error_old = error_old**2
    SUM_OF_SQUARE_ERROR = []
    SUM_OF_SQUARE_ERROR.append(xp.sum(sq_error_old))
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
        Jacobian = get_jacobian(bond_list, N, v_old, w_old, traj4, mass_weights, fluctuation_MD, T, error_old, deri_H_per_bond)
        print("jacobian calculated")
        pertub = xp.squeeze(solve(Jacobian, sq_error_old.reshape((len(bond_list), 1))))
        beta =  0.1/xp.max(xp.abs(pertub/bond_list[:, 2]))
        K_new = bond_list[:,2]  - beta * pertub
        delta_w = eigenvalue_jacob_gpu(bond_list, N, traj4, mass_weights, v_old, deri_H_per_bond) @ ((K_new - bond_list[:, 2]).reshape((len(bond_list), 1)))
        # w_new = w_old + xp.squeeze(delta_w)
        # for i in range (v_old.shape[1]):
        #     delta_v = eigenvector_jacobian_gpu(bond_list, v_old, w_old, deri_H_per_bond, i, N) @ ((K_new - bond_list[:, 2]).reshape((len(bond_list), 1)))
        #     v_new[:,i] = v_old[:,i] + xp.squeeze(delta_v)
        v_new, w_new,e= NMA(N, bond_list, fluctuation_MD, T, traj4, mass_weights)
        K_new = xp.asarray(K_new, dtype=xp.complex128)
        K_new = xp.asarray(xp.real(K_new), dtype=xp.float64)
        K_new[:] = xp.where(K_new< 0, 0, K_new)
        bond_list[:, 2] = K_new
        print("updated K values real", K_new.dtype)
        v_old = v_new   
        w_old = w_new
        k = k + 1
        error_old= NMA_error(bond_list, traj4, mass_weights, fluctuation_MD, T, N, v_old, w_old)
        sq_error_old = error_old**2 
        SUM_OF_SQUARE_ERROR.append(xp.sum(sq_error_old))
        itr.append(k)

    plt.plot(itr, SUM_OF_SQUARE_ERROR, linestyle='solid')
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
