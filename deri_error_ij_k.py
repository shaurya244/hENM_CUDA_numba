import numpy as xp

from func_deri_eigen_val_vec import deri_eign_val
from func_deri_eigen_val_vec import deri_eign_vec
from func_force_field_determination import NMA

def deri_error_ij_df(bond_list, i, j, d, f, N, v, w, traj4, mass_weights, fluctuation_MD, T, error_old):
    kB = 8.314462618 * 0.001  # kJ/mol K
    DEL_error_i_j = 0

    for k in range(0, (3 * N) - 6):
        delw_k_delP_r = deri_eign_val(d, f, v, k, traj4, mass_weights, bond_list)
        delv_k_delP_r = deri_eign_vec(v, w, d, f, k, traj4, mass_weights, bond_list, N)

        w_k_real = xp.real(w[k])
        sqrt_wk = xp.sqrt(xp.abs(w_k_real))
        inv_mass_i = xp.sqrt(mass_weights[i])
        inv_mass_j = xp.sqrt(mass_weights[j])

        def displacement(dim):
            return traj4.xyz[0, i, dim] - traj4.xyz[0, j, dim]

        def vterm(dim):
            return (v.real[3 * i + dim][k] / inv_mass_i) - (v.real[3 * j + dim][k] / inv_mass_j)

        def dvterm(dim):
            return (delv_k_delP_r.real[3 * i + dim] / inv_mass_i) - (delv_k_delP_r.real[3 * j + dim] / inv_mass_j)

        proj = sum(displacement(dim) * vterm(dim) for dim in range(3))

        dproj_dw = ((-0.5) * (w_k_real) ** (-1.5)) * delw_k_delP_r * proj
        dproj_dv = sum(displacement(dim) * dvterm(dim) for dim in range(3)) / sqrt_wk

        DEL_error_i_j += (proj / sqrt_wk) * (dproj_dw + dproj_dv)

    row = xp.where((bond_list[:, 0] == i + 1) & (bond_list[:, 1] == j + 1))[0]
    deri_sq_error = (4 * kB * T * error_old[row[0]]) * ((1 / bond_list[row[0], 3]) ** 2) * DEL_error_i_j
    return deri_sq_error
