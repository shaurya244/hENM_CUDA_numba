import numpy as np
from numba import cuda

# ============================================================
# DEVICE FUNCTION
# ============================================================
@cuda.jit
def Atomic_add_core(Vr, Vk, a, acc):
    i = cuda.grid(1)

    # local partial
    tmp = 0.0

    if i < Vr.shape[0]:
        tmp = a * Vr[i] * Vk[i]

    # accumulate into global scalar
    cuda.atomic.add(acc, 0, tmp)
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
@cuda.jit(device=True)
def deri_eign_vec_device(p, i, j, k, v, w, deri_H, n_modes, delw_k_delP_r):
    acc = 0.0

    for r in range(n_modes):
        if r == k:
            continue
        a = 0.0
        for m in range(v.shape[0]):
            a += -delw_k_delP_r * v[m, r] * v[m, k]
        vr0 = v[3*i+0, r]; vr1 = v[3*i+1, r]; vr2 = v[3*i+2, r]
        vr3 = v[3*j+0, r]; vr4 = v[3*j+1, r]; vr5 = v[3*j+2, r]

        vk0 = v[3*i+0, k]; vk1 = v[3*i+1, k]; vk2 = v[3*i+2, k]
        vk3 = v[3*j+0, k]; vk4 = v[3*j+1, k]; vk5 = v[3*j+2, k]
        b = -delw_k_delP_r*(vr0*vk0 + vr1*vk1 + vr2*vk2 + vr3*vk3 + vr4*vk4 + vr5*vk5)
        num = (
            vr0*((deri_H[0,0]-delw_k_delP_r)*vk0 + deri_H[0,1]*vk1 + deri_H[0,2]*vk2 + deri_H[0,3]*vk3 + deri_H[0,4]*vk4 + deri_H[0,5]*vk5) +
            vr1*(deri_H[1,0]*vk0 + (deri_H[1,1]-delw_k_delP_r)*vk1 + deri_H[1,2]*vk2 + deri_H[1,3]*vk3 + deri_H[1,4]*vk4 + deri_H[1,5]*vk5) +
            vr2*(deri_H[2,0]*vk0 + deri_H[2,1]*vk1 + (deri_H[2,2]-delw_k_delP_r)*vk2 + deri_H[2,3]*vk3 + deri_H[2,4]*vk4 + deri_H[2,5]*vk5) +
            vr3*(deri_H[3,0]*vk0 + deri_H[3,1]*vk1 + deri_H[3,2]*vk2 + (deri_H[3,3]-delw_k_delP_r)*vk3 + deri_H[3,4]*vk4 + deri_H[3,5]*vk5) +
            vr4*(deri_H[4,0]*vk0 + deri_H[4,1]*vk1 + deri_H[4,2]*vk2 + deri_H[4,3]*vk3 + (deri_H[4,4]-delw_k_delP_r)*vk4 + deri_H[4,5]*vk5) +
            vr5*(deri_H[5,0]*vk0 + deri_H[5,1]*vk1 + deri_H[5,2]*vk2 + deri_H[5,3]*vk3 + deri_H[5,4]*vk4 + (deri_H[5,5]-delw_k_delP_r)*vk5)+a-b
        )

        denom = w[k] - w[r]   # Fox-Kapoor eq. 5: denominator is (λ_k − λ_r)
        if denom != 0.0:
            acc += (num / denom) * v[p, r]

    return acc


# ============================================================
# CUDA KERNEL
# ============================================================

@cuda.jit
def eigenvector_jacobian_kernel(
    bond_i, bond_j,
    v, w,
    deri_H_all,
    k,
    n_modes,
    M, B,
    J
):
    p, n = cuda.grid(2)

    if p < M and n < B:
        i = bond_i[n]
        j = bond_j[n]
        delw_k_dleP_r = gpu_deri_eign_val_only(
        v, n, i, j, k, deri_H_all
    )

        J[p, n] = deri_eign_vec_device(
            p, i, j, k,
            v, w,
            deri_H_all[n],
            n_modes,delw_k_dleP_r
        )


# ============================================================
# HOST LAUNCHER
# ============================================================

def eigenvector_jacobian_gpu(bond_list, v, w, deri_H_all, k, N):
    bond_i = bond_list[:, 0].astype(np.int32) - 1
    bond_j = bond_list[:, 1].astype(np.int32) - 1

    M = v.shape[0]
    B = bond_i.shape[0]
    n_modes = 3*N - 6

    bond_i_d = cuda.to_device(bond_i)
    bond_j_d = cuda.to_device(bond_j)
    v_d = cuda.to_device(v.astype(np.float64))
    w_d = cuda.to_device(w.astype(np.float64))
    deri_H_d = cuda.to_device(deri_H_all.astype(np.float64))

    J_d = cuda.device_array((M, B), dtype=np.float64)

    threads = (16, 16)
    blocks = ((M + 15)//16, (B + 15)//16)

    eigenvector_jacobian_kernel[blocks, threads](
        bond_i_d, bond_j_d,
        v_d, w_d,
        deri_H_d,
        k,
        n_modes,
        M, B,
        J_d
    )

    return J_d.copy_to_host()
