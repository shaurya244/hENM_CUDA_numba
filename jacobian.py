# gpu_jacobian.py
import math
import numpy as np
from numba import cuda, float64, int32
from derivative_hessian_updated import derivative_hessian  # to calculate the partial derivative of the hessian

# -----------------------
# Device helper functions
# -----------------------

@cuda.jit(device=True)
def gpu_deri_eign_val(v_real, bond_idx, i_atom, j_atom, k_mode, deri_H_per_bond):
    """
    Computes derivative eigenvalue scalar for mode k with respect to bond bond_idx.
    v_real: (ndof, modes) float64
    deri_H_per_bond: (M,6,6) float64
    i_atom, j_atom: 0-based atom indices (integers)
    returns: float64
    """
    # gather v components for atoms i and j for mode k (6 components)
    v11 = v_real[3 * i_atom + 0, k_mode]
    v12 = v_real[3 * i_atom + 1, k_mode]
    v13 = v_real[3 * i_atom + 2, k_mode]
    v21 = v_real[3 * j_atom + 0, k_mode]
    v22 = v_real[3 * j_atom + 1, k_mode]
    v23 = v_real[3 * j_atom + 2, k_mode]

    # H6 is 6x6 numeric array for bond bond_idx
    H = deri_H_per_bond[bond_idx]

    # temp = H @ v_ij  (six components)
    t0 = H[0,0]*v11 + H[0,1]*v12 + H[0,2]*v13 + H[0,3]*v21 + H[0,4]*v22 + H[0,5]*v23
    t1 = H[1,0]*v11 + H[1,1]*v12 + H[1,2]*v13 + H[1,3]*v21 + H[1,4]*v22 + H[1,5]*v23
    t2 = H[2,0]*v11 + H[2,1]*v12 + H[2,2]*v13 + H[2,3]*v21 + H[2,4]*v22 + H[2,5]*v23
    t3 = H[3,0]*v11 + H[3,1]*v12 + H[3,2]*v13 + H[3,3]*v21 + H[3,4]*v22 + H[3,5]*v23
    t4 = H[4,0]*v11 + H[4,1]*v12 + H[4,2]*v13 + H[4,3]*v21 + H[4,4]*v22 + H[4,5]*v23
    t5 = H[5,0]*v11 + H[5,1]*v12 + H[5,2]*v13 + H[5,3]*v21 + H[5,4]*v22 + H[5,5]*v23

    # dot product v_ij.T @ temp
    val = v11*t0 + v12*t1 + v13*t2 + v21*t3 + v22*t4 + v23*t5
    return val


@cuda.jit(device=True)
def gpu_deri_eign_vec_partial(v_real, w_real, bond_idx, i_atom, j_atom, k_mode, modes, deri_H_per_bond):
    """
    Computes only the 6 components of derivative eigenvector needed by your projection:
    returns (delv_i0, delv_i1, delv_i2, delv_j0, delv_j1, delv_j2)
    v_real: (ndof, modes)
    w_real: (modes,)
    deri_H_per_bond: (M,6,6)
    """
    # initialize accumulators
    delv_i0 = 0.0
    delv_i1 = 0.0
    delv_i2 = 0.0
    delv_j0 = 0.0
    delv_j1 = 0.0
    delv_j2 = 0.0

    H6 = deri_H_per_bond[bond_idx]

    # loop over all modes r != k_mode
    for r in range(modes):
        if r == k_mode:
            continue

        denom = w_real[k_mode] - w_real[r]
        if denom == 0.0:
            continue

        # v_r_ij components (6)
        vr0 = v_real[3 * i_atom + 0, r]; vr1 = v_real[3 * i_atom + 1, r]; vr2 = v_real[3 * i_atom + 2, r]
        vr3 = v_real[3 * j_atom + 0, r]; vr4 = v_real[3 * j_atom + 1, r]; vr5 = v_real[3 * j_atom + 2, r]

        # v_k_ij components (6)
        vk0 = v_real[3 * i_atom + 0, k_mode]; vk1 = v_real[3 * i_atom + 1, k_mode]; vk2 = v_real[3 * i_atom + 2, k_mode]
        vk3 = v_real[3 * j_atom + 0, k_mode]; vk4 = v_real[3 * j_atom + 1, k_mode]; vk5 = v_real[3 * j_atom + 2, k_mode]

        # temp = H6 @ v_k_ij
        t0 = H6[0,0]*vk0 + H6[0,1]*vk1 + H6[0,2]*vk2 + H6[0,3]*vk3 + H6[0,4]*vk4 + H6[0,5]*vk5
        t1 = H6[1,0]*vk0 + H6[1,1]*vk1 + H6[1,2]*vk2 + H6[1,3]*vk3 + H6[1,4]*vk4 + H6[1,5]*vk5
        t2 = H6[2,0]*vk0 + H6[2,1]*vk1 + H6[2,2]*vk2 + H6[2,3]*vk3 + H6[2,4]*vk4 + H6[2,5]*vk5
        t3 = H6[3,0]*vk0 + H6[3,1]*vk1 + H6[3,2]*vk2 + H6[3,3]*vk3 + H6[3,4]*vk4 + H6[3,5]*vk5
        t4 = H6[4,0]*vk0 + H6[4,1]*vk1 + H6[4,2]*vk2 + H6[4,3]*vk3 + H6[4,4]*vk4 + H6[4,5]*vk5
        t5 = H6[5,0]*vk0 + H6[5,1]*vk1 + H6[5,2]*vk2 + H6[5,3]*vk3 + H6[5,4]*vk4 + H6[5,5]*vk5

        dot = vr0*t0 + vr1*t1 + vr2*t2 + vr3*t3 + vr4*t4 + vr5*t5
        a_k = dot / denom

        if a_k != 0.0:
            delv_i0 += a_k * v_real[3 * i_atom + 0, r]
            delv_i1 += a_k * v_real[3 * i_atom + 1, r]
            delv_i2 += a_k * v_real[3 * i_atom + 2, r]
            delv_j0 += a_k * v_real[3 * j_atom + 0, r]
            delv_j1 += a_k * v_real[3 * j_atom + 1, r]
            delv_j2 += a_k * v_real[3 * j_atom + 2, r]

    return delv_i0, delv_i1, delv_i2, delv_j0, delv_j1, delv_j2


# -----------------------
# Main kernel: jacobian
# -----------------------

@cuda.jit
def jacobian_kernel_core(bond_list, positions, v_real, w_real, deri_H_per_bond,fluctuation_MD,
                         mass_weights, error_old, T, jac_out):
    """
    Each thread computes jac_out[m, n] as in your CPU version.
    bond_list: int32[:,] (M, >=4)  columns: atom1, atom2, ..., column 3 used for scaling
    positions: float64[:,3] (N,3)
    v_real: float64[:, :] (ndof, modes) where ndof=3*N
    w_real: float64[:] (modes,)
    deri_H_per_bond: float64[:,6,6] (M,6,6)
    mass_weights: float64[:] (N,)
    error_old: float64[:] (M,)
    T: float scalar
    jac_out: float64[:, :] (M, M)   output
    """
    m, n = cuda.grid(2)
    M = bond_list.shape[0]
    if m >= M or n >= M:
        return

    modes = w_real.shape[0]

    # atom indices for bond m (i,j) and bond n (d,f) - convert 1-based -> 0-based
    i = bond_list[m, 0] - 1
    j = bond_list[m, 1] - 1
    d = bond_list[n, 0] - 1
    f = bond_list[n, 1] - 1

    # precompute mass sqrt factors
    inv_mass_i = math.sqrt(mass_weights[i])
    inv_mass_j = math.sqrt(mass_weights[j])

    # displacements between i and j (positions are float64)
    disp0 = positions[i, 0] - positions[j, 0]
    disp1 = positions[i, 1] - positions[j, 1]
    disp2 = positions[i, 2] - positions[j, 2]

    DEL_error_i_j = 0.0
    fluctuation_NMA = 0.0
    # loop over modes
    for k in range(modes):
        # derivative eigenvalue wrt bond n for mode k
        delw_k_delP_r = gpu_deri_eign_val(v_real, n, d, f, k, deri_H_per_bond)

        w_k_real = w_real[k]
        sqrt_wk = math.sqrt(abs(w_k_real))
        if sqrt_wk == 0.0:
            continue

        v_i0 = v_real[3 * i + 0, k]; v_i1 = v_real[3 * i + 1, k]; v_i2 = v_real[3 * i + 2, k]
        v_j0 = v_real[3 * j + 0, k]; v_j1 = v_real[3 * j + 1, k]; v_j2 = v_real[3 * j + 2, k]

        vterm0 = (v_i0 / inv_mass_i) - (v_j0 / inv_mass_j)
        vterm1 = (v_i1 / inv_mass_i) - (v_j1 / inv_mass_j)
        vterm2 = (v_i2 / inv_mass_i) - (v_j2 / inv_mass_j)

        proj = disp0 * vterm0 + disp1 * vterm1 + disp2 * vterm2

        dv_i0, dv_i1, dv_i2, dv_j0, dv_j1, dv_j2 = gpu_deri_eign_vec_partial(
            v_real, w_real, n, d, f, k, modes, deri_H_per_bond
        )

        dvterm0 = (dv_i0 / inv_mass_i) - (dv_j0 / inv_mass_j)
        dvterm1 = (dv_i1 / inv_mass_i) - (dv_j1 / inv_mass_j)
        dvterm2 = (dv_i2 / inv_mass_i) - (dv_j2 / inv_mass_j)

        dproj_dv = (disp0 * dvterm0 + disp1 * dvterm1 + disp2 * dvterm2) / sqrt_wk

        denom_abs = abs(w_k_real)
        if denom_abs == 0.0:
            dproj_dw = 0.0
        else:
            dproj_dw = (-0.5) * (denom_abs ** (-1.5)) * delw_k_delP_r * proj

        DEL_error_i_j += (proj / sqrt_wk) * (dproj_dw + (1/sqrt_wk)*dproj_dv)*(8.314462618 * 0.001 *T*((proj /((bond_list[m, 3])*sqrt_wk)) ** 2)- fluctuation_MD[m])
         
    denom_b = bond_list[m, 3]
   
    scale = ((1.0 / denom_b) ** 2)

    kB = 8.314462618 * 0.001  
    deri_sq_error = (4.0 * kB * T ) * scale * DEL_error_i_j 

    jac_out[m, n] = deri_sq_error

def get_jacobian(bond_list, N, v, w, traj4, mass_weights, fluctuation_MD, T, error_old, deri_H_per_bond):
    """
    GPU-backed replacement for get_jacobian.
    Returns jacobian as complex128 of shape (M, M) to match original API.
    Requirements:
      - derivative_hessian(bond_list, i, j, traj4, mass_weights) must be available and return numeric 6x6 arrays.
      - numba.cuda and a CUDA-capable GPU.
    """
    # -- basic conversions and checks
    bond_list = np.asarray(bond_list)
    M = len(bond_list)
    modes = 3 * N - 6
    ndof = 3 * N

    # Ensure v_real shape is (ndof, modes)
    v_real = np.real(v)
    if v_real.shape == (modes, ndof):
        v_real = v_real.T
    if v_real.shape != (ndof, modes):
        raise ValueError(f"v.real shape {v_real.shape} not (ndof, modes) or transposable.")

    w_real = np.real(w).astype(np.float64).reshape(-1)
    if w_real.shape[0] != modes:
        raise ValueError("w.real length mismatch with modes.")

    positions = np.asarray(traj4.xyz[0], dtype=np.float64)
    if positions.shape != (N, 3):
        raise ValueError("traj4.xyz[0] must have shape (N,3).")

    mass_weights = np.asarray(mass_weights, dtype=np.float64).reshape(-1)
    error_old = np.asarray(error_old, dtype=np.float64).reshape(-1)
    bond_list_i32 = bond_list.astype(np.int32)

    # -- move to device
    bond_dev = cuda.to_device(bond_list_i32)
    pos_dev = cuda.to_device(positions)
    v_dev = cuda.to_device(v_real.astype(np.float64))
    w_dev = cuda.to_device(w_real)
    H_dev = cuda.to_device(deri_H_per_bond)
    mass_dev = cuda.to_device(mass_weights)
    err_dev = cuda.to_device(error_old)
    fluctuation_MD_dev = cuda.to_device(fluctuation_MD)
    jac_dev = cuda.device_array((M, M), dtype=np.float64)

    # launch config
    threadsperblock = (16, 16)
    blockspergrid = (math.ceil(M / threadsperblock[0]), math.ceil(M / threadsperblock[1]))

    # launch kernel
    jacobian_kernel_core[blockspergrid, threadsperblock](
        bond_dev, pos_dev, v_dev, w_dev, H_dev, fluctuation_MD_dev, mass_dev, err_dev, T, jac_dev
    )

    # copy back
    jac_real = jac_dev.copy_to_host()
    # return complex128 to match original API
    jac_complex = np.zeros((M, M), dtype=np.complex128)
    jac_complex.real = jac_real
    return jac_complex

