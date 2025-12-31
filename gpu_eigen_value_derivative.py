import numpy as np
from numba import cuda
import math

from numba import cuda
import math

@cuda.jit(device=True)
def gpu_deri_eign_val_only(v_real, bond_idx, i_atom, j_atom, k_mode, deri_H):
    v11 = v_real[3*i_atom+0, k_mode]
    v12 = v_real[3*i_atom+1, k_mode]
    v13 = v_real[3*i_atom+2, k_mode]
    v21 = v_real[3*j_atom+0, k_mode]
    v22 = v_real[3*j_atom+1, k_mode]
    v23 = v_real[3*j_atom+2, k_mode]

    H = deri_H[bond_idx]

    t0 = H[0,0]*v11 + H[0,1]*v12 + H[0,2]*v13 + H[0,3]*v21 + H[0,4]*v22 + H[0,5]*v23
    t1 = H[1,0]*v11 + H[1,1]*v12 + H[1,2]*v13 + H[1,3]*v21 + H[1,4]*v22 + H[1,5]*v23
    t2 = H[2,0]*v11 + H[2,1]*v12 + H[2,2]*v13 + H[2,3]*v21 + H[2,4]*v22 + H[2,5]*v23
    t3 = H[3,0]*v11 + H[3,1]*v12 + H[3,2]*v13 + H[3,3]*v21 + H[3,4]*v22 + H[3,5]*v23
    t4 = H[4,0]*v11 + H[4,1]*v12 + H[4,2]*v13 + H[4,3]*v21 + H[4,4]*v22 + H[4,5]*v23
    t5 = H[5,0]*v11 + H[5,1]*v12 + H[5,2]*v13 + H[5,3]*v21 + H[5,4]*v22 + H[5,5]*v23

    return v11*t0 + v12*t1 + v13*t2 + v21*t3 + v22*t4 + v23*t5

@cuda.jit
def eigenvalue_jacobian_kernel(bond_list, v_real, deri_H, jac_out):
    """
    jac_out[k, n] = d lambda_k / d K_n
    """
    k, n = cuda.grid(2)

    modes = jac_out.shape[0]
    Nbonds = jac_out.shape[1]

    if k >= modes or n >= Nbonds:
        return

    i = bond_list[n, 0] - 1
    j = bond_list[n, 1] - 1

    jac_out[k, n] = gpu_deri_eign_val_only(
        v_real, n, i, j, k, deri_H
    )

def eigenvalue_jacob_gpu(bond_list, N, traj4, mass_weights, v_old, deri_H):
    bond_list = np.asarray(bond_list, dtype=np.int32)
    Nbonds = len(bond_list)
    modes = 3*N - 6
    ndof = 3*N

    # v_old.real -> (ndof, modes)
    v_real = np.real(v_old)
    if v_real.shape == (modes, ndof):
        v_real = v_real.T

    # --- Precompute derivative Hessians on CPU


    # --- Move to GPU
    bond_dev = cuda.to_device(bond_list)
    v_dev = cuda.to_device(v_real.astype(np.float64))
    H_dev = cuda.to_device(deri_H)
    jac_dev = cuda.device_array((modes, Nbonds), dtype=np.float64)

    threads = (16, 16)
    blocks = (math.ceil(modes/16), math.ceil(Nbonds/16))

    eigenvalue_jacobian_kernel[blocks, threads](
        bond_dev, v_dev, H_dev, jac_dev
    )

    jac = jac_dev.copy_to_host()

    # match original API (complex128)
    out = np.zeros((modes, Nbonds), dtype=np.complex128)
    out.real = jac
    return out
