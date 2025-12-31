import numpy as np
from numba import cuda

# ============================================================
# DEVICE FUNCTION
# ============================================================

@cuda.jit(device=True)
def deri_eign_vec_device(p, i, j, k, v, w, deri_H, n_modes):
    acc = 0.0

    for r in range(n_modes):
        if r == k:
            continue

        vr0 = v[3*i+0, r]; vr1 = v[3*i+1, r]; vr2 = v[3*i+2, r]
        vr3 = v[3*j+0, r]; vr4 = v[3*j+1, r]; vr5 = v[3*j+2, r]

        vk0 = v[3*i+0, k]; vk1 = v[3*i+1, k]; vk2 = v[3*i+2, k]
        vk3 = v[3*j+0, k]; vk4 = v[3*j+1, k]; vk5 = v[3*j+2, k]

        num = (
            vr0*(deri_H[0,0]*vk0 + deri_H[0,1]*vk1 + deri_H[0,2]*vk2 + deri_H[0,3]*vk3 + deri_H[0,4]*vk4 + deri_H[0,5]*vk5) +
            vr1*(deri_H[1,0]*vk0 + deri_H[1,1]*vk1 + deri_H[1,2]*vk2 + deri_H[1,3]*vk3 + deri_H[1,4]*vk4 + deri_H[1,5]*vk5) +
            vr2*(deri_H[2,0]*vk0 + deri_H[2,1]*vk1 + deri_H[2,2]*vk2 + deri_H[2,3]*vk3 + deri_H[2,4]*vk4 + deri_H[2,5]*vk5) +
            vr3*(deri_H[3,0]*vk0 + deri_H[3,1]*vk1 + deri_H[3,2]*vk2 + deri_H[3,3]*vk3 + deri_H[3,4]*vk4 + deri_H[3,5]*vk5) +
            vr4*(deri_H[4,0]*vk0 + deri_H[4,1]*vk1 + deri_H[4,2]*vk2 + deri_H[4,3]*vk3 + deri_H[4,4]*vk4 + deri_H[4,5]*vk5) +
            vr5*(deri_H[5,0]*vk0 + deri_H[5,1]*vk1 + deri_H[5,2]*vk2 + deri_H[5,3]*vk3 + deri_H[5,4]*vk4 + deri_H[5,5]*vk5)
        )

        denom = w[k] - w[r]
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
        J[p, n] = deri_eign_vec_device(
            p, i, j, k,
            v, w,
            deri_H_all[n],
            n_modes
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
