# gpu_jacobian.py - OPTIMIZED VERSION
import math
import numpy as np
from numba import cuda, float64, int32

# -----------------------
# Device helper functions
# -----------------------

@cuda.jit(device=True)
def matmul_6x6(A, B, C):
    """Optimized 6x6 matrix-vector multiplication (unrolled)"""
    for i in range(6):
        s = 0.0
        for j in range(6):
            s += A[i, j] * B[j]
        C[i] = s

@cuda.jit(device=True)
def matmul_full(A, B, C, n):
    """General matrix-vector multiplication for arbitrary sizes"""
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * B[j]
        C[i] = s

@cuda.jit(device=True)
def make_identity_local(I, n):
    """Create identity matrix (n x n) locally"""
    for i in range(n):
        for j in range(n):
            I[i, j] = 1.0 if i == j else 0.0

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
def gpu_deri_eign_vec_partial(v_real, w_real, bond_idx, i_atom, j_atom, k_mode, modes, ndof, delw_k_delP_r, deri_H_per_bond):
    """
    Computes only the 6 components of derivative eigenvector needed by projection.
    FIXED: Now uses dynamic ndof instead of hardcoded 120
    returns (delv_i0, delv_i1, delv_i2, delv_j0, delv_j1, delv_j2)
    """
    # initialize accumulators
    delv_i0 = 0.0
    delv_i1 = 0.0
    delv_i2 = 0.0
    delv_j0 = 0.0
    delv_j1 = 0.0
    delv_j2 = 0.0
    
    # Create identity matrix with correct size
    I = cuda.local.array((ndof, ndof), dtype=float64)
    make_identity_local(I, ndof)
    
    # Compute (dH - delw*I)
    dH_vec = cuda.local.array((ndof, ndof), dtype=float64)
    for ii in range(ndof):
        for jj in range(ndof):
            dH_vec[ii, jj] = deri_H_per_bond[bond_idx][ii, jj] - (delw_k_delP_r if ii == jj else 0.0)

    # loop over all modes r != k_mode
    for r in range(modes):
        if r == k_mode:
            continue

        denom = w_real[r]
        if denom == 0.0:
            continue

        # Get full mode vectors
        vr = v_real[:, r]
        vk = v_real[:, k_mode]
        
        # Compute H_vk = dH_vec @ vk
        H_vk = cuda.local.array(ndof, dtype=float64)
        matmul_full(dH_vec, vk, H_vk, ndof)
        
        # Compute dot product: vr.T @ H_vk
        dot_prod = 0.0
        for ii in range(ndof):
            dot_prod += vr[ii] * H_vk[ii]
        
        a_k = dot_prod / denom

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
def jacobian_kernel_core(bond_list, positions, v_real, w_real, H_eigen_value_dev, H_eigen_vector_dev, 
                         fluctuation_MD, mass_weights, T, jac_out, ndof):
    """
    Each thread computes jac_out[m, n].
    bond_list: int32[:,] (M, >=4)  columns: atom1, atom2, scaling_factor, ...
    positions: float64[:,3] (N,3)
    v_real: float64[:, :] (ndof, modes)
    w_real: float64[:] (modes,)
    H_eigen_value_dev: float64[:,6,6] (M,6,6)
    H_eigen_vector_dev: float64[:,ndof,ndof] (M,ndof,ndof)
    mass_weights: float64[:] (N,)
    T: float scalar (temperature)
    jac_out: float64[:, :] (M, M) output
    ndof: int scalar (3*N)
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

    # Guard against invalid atom indices
    if i < 0 or j < 0 or d < 0 or f < 0:
        return
    
    # Precompute reciprocals (avoid divisions in loop)
    sqrt_mass_i = math.sqrt(mass_weights[i])
    sqrt_mass_j = math.sqrt(mass_weights[j])
    inv_sqrt_mass_i = 1.0 / sqrt_mass_i if sqrt_mass_i > 0.0 else 0.0
    inv_sqrt_mass_j = 1.0 / sqrt_mass_j if sqrt_mass_j > 0.0 else 0.0

    # displacements between i and j
    disp0 = positions[i, 0] - positions[j, 0]
    disp1 = positions[i, 1] - positions[j, 1]
    disp2 = positions[i, 2] - positions[j, 2]
    
    # Precompute scaling factor and constants
    denom_b = bond_list[m, 3]
    if denom_b == 0.0:
        return
    
    scale = 1.0 / (denom_b * denom_b)
    kB = 8.314462618 * 0.001  # Boltzmann constant (precomputed)
    prefactor = 4.0 * kB * T * scale  # Precompute this constant
    
    DEL_error_i_j = 0.0
    
    # loop over modes
    for k in range(modes):
        # derivative eigenvalue wrt bond n for mode k
        delw_k_delP_r = gpu_deri_eign_val(v_real, n, d, f, k, H_eigen_value_dev)

        w_k_real = w_real[k]
        
        # Guard against zero/negative eigenvalues
        if w_k_real <= 0.0:
            continue
        
        sqrt_wk = math.sqrt(w_k_real)
        inv_sqrt_wk = 1.0 / sqrt_wk

        # Load eigenvector components
        v_i0 = v_real[3 * i + 0, k]
        v_i1 = v_real[3 * i + 1, k]
        v_i2 = v_real[3 * i + 2, k]
        v_j0 = v_real[3 * j + 0, k]
        v_j1 = v_real[3 * j + 1, k]
        v_j2 = v_real[3 * j + 2, k]

        # Compute v term (mass-weighted eigenvector difference)
        vterm0 = v_i0 * inv_sqrt_mass_i - v_j0 * inv_sqrt_mass_j
        vterm1 = v_i1 * inv_sqrt_mass_i - v_j1 * inv_sqrt_mass_j
        vterm2 = v_i2 * inv_sqrt_mass_i - v_j2 * inv_sqrt_mass_j

        # Projection
        proj = disp0 * vterm0 + disp1 * vterm1 + disp2 * vterm2

        # Get derivative eigenvector components
        dv_i0, dv_i1, dv_i2, dv_j0, dv_j1, dv_j2 = gpu_deri_eign_vec_partial(
            v_real, w_real, n, d, f, k, modes, ndof, delw_k_delP_r, H_eigen_vector_dev
        )

        # Compute dv term
        dvterm0 = dv_i0 * inv_sqrt_mass_i - dv_j0 * inv_sqrt_mass_j
        dvterm1 = dv_i1 * inv_sqrt_mass_i - dv_j1 * inv_sqrt_mass_j
        dvterm2 = dv_i2 * inv_sqrt_mass_i - dv_j2 * inv_sqrt_mass_j

        # Derivative of projection wrt eigenvector
        dproj_dv = (disp0 * dvterm0 + disp1 * dvterm1 + disp2 * dvterm2) * inv_sqrt_wk

        # Derivative wrt eigenvalue
        w_k_abs = abs(w_k_real)
        dproj_dw = (-0.5) * (w_k_abs ** (-1.5)) * delw_k_delP_r * proj

        # Compute squared bond fluctuation (cached computation)
        proj_normalized = proj * inv_sqrt_wk / denom_b
        proj_sq = proj_normalized * proj_normalized
        
        # Accumulated error term
        fluctuation_diff = kB * T * proj_sq - fluctuation_MD[m]
        DEL_error_i_j += (proj * inv_sqrt_wk) * (dproj_dw + inv_sqrt_wk * dproj_dv) * fluctuation_diff
         
    deri_sq_error = prefactor * DEL_error_i_j
    jac_out[m, n] = deri_sq_error


def get_jacobian(bond_list, N, v, w, traj4, mass_weights, fluctuation_MD, T, 
                 deri_H_per_bond_eigen_value, deri_H_per_bond_eigen_vector):
    """
    GPU-backed jacobian computation.
    Returns jacobian as complex128 of shape (M, M) to match original API.
    """
    # -- basic conversions and checks
    bond_list = np.asarray(bond_list)
    M = len(bond_list)
    modes = 3 * N - 6
    ndof = 3 * N

    # Ensure v_real shape is (ndof, modes)
    v_real = np.real(v).astype(np.float64)
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
    if mass_weights.shape[0] != N:
        raise ValueError("mass_weights shape mismatch with N.")
    
    # Validate bond_list
    if bond_list.shape[1] < 4:
        raise ValueError("bond_list must have at least 4 columns.")
    
    # Validate deri_H arrays
    if deri_H_per_bond_eigen_value.shape[0] != M or deri_H_per_bond_eigen_value.shape != (M, 6, 6):
        raise ValueError(f"deri_H_per_bond_eigen_value shape mismatch: expected ({M}, 6, 6)")
    
    if deri_H_per_bond_eigen_vector.shape[0] != M or deri_H_per_bond_eigen_vector.shape[1] != ndof:
        raise ValueError(f"deri_H_per_bond_eigen_vector shape mismatch: expected ({M}, {ndof}, {ndof})")
    
    fluctuation_MD = np.asarray(fluctuation_MD, dtype=np.float64).reshape(-1)
    if fluctuation_MD.shape[0] != M:
        raise ValueError("fluctuation_MD length mismatch with M bonds.")
    
    bond_list_i32 = bond_list.astype(np.int32)

    # -- move to device
    bond_dev = cuda.to_device(bond_list_i32)
    pos_dev = cuda.to_device(positions)
    v_dev = cuda.to_device(v_real)
    w_dev = cuda.to_device(w_real)
    H_eigen_value_dev = cuda.to_device(deri_H_per_bond_eigen_value.astype(np.float64))
    H_eigen_vector_dev = cuda.to_device(deri_H_per_bond_eigen_vector.astype(np.float64))
    mass_dev = cuda.to_device(mass_weights)
    fluctuation_MD_dev = cuda.to_device(fluctuation_MD)
    jac_dev = cuda.device_array((M, M), dtype=np.float64)

    # launch config
    threadsperblock = (16, 16)
    blockspergrid = (math.ceil(M / threadsperblock[0]), math.ceil(M / threadsperblock[1]))

    # launch kernel with ndof parameter
    jacobian_kernel_core[blockspergrid, threadsperblock](
        bond_dev, pos_dev, v_dev, w_dev, H_eigen_value_dev, H_eigen_vector_dev, 
        fluctuation_MD_dev, mass_dev, T, jac_dev, ndof
    )

    # copy back
    jac_real = jac_dev.copy_to_host()
    # return complex128 to match original API
    jac_complex = np.zeros((M, M), dtype=np.complex128)
    jac_complex.real = jac_real
    return jac_complex