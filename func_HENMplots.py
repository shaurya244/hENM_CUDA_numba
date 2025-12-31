import numpy as xp
   

def HENM_plots(N, k_sp, bo_md, bond_list, dr_sq, path):
    # Ensure inputs are moved to CPU for printing/saving
    k_sp = xp.asnumpy(k_sp) if hasattr(k_sp, 'get') else k_sp
    dr_sq = xp.asnumpy(dr_sq) if hasattr(dr_sq, 'get') else dr_sq

    ######### RESULTS.txt ##############
    with open(f"{path}/OUTPUT/RESULTS.txt", 'w') as f:
        f.write("Bond\n")
        f.write("i\tj\tk_ij (kJ/mol A^2)\t(bo_ij)(A)\tbond\n")
        bond = 1
        for i in range(N):
            for j in range(i + 1, N):
                bond_type = [i + 1, j + 1]
                if bond_type in bond_list:
                    f.write(f"{i+1}\t{j+1}\t{k_sp[i][j]}\t{dr_sq[i][j]}\t{bond}\n")
                    bond += 1

    ######## SPRING CONSTANT.txt ########
    with open(f"{path}/POST_PROCESSING/SPRING CONSTANT.txt", 'w') as f:
        for i in range(N):
            for j in range(i + 1, N):
                if [i + 1, j + 1] in bond_list:
                    f.write(f"{k_sp[i][j]}\n")

    #### SPRING CONSTANT NON ZERO.txt ###
    with open(f"{path}/POST_PROCESSING/SPRING CONSTANT NON ZERO.txt", 'w') as f:
        for i in range(N):
            for j in range(i + 1, N):
                if [i + 1, j + 1] in bond_list and k_sp[i][j] > 0:
                    f.write(f"{k_sp[i][j]}\n")

    #### SPRING CONSTANT_full.txt #######
    with open(f"{path}/POST_PROCESSING/SPRING CONSTANT_full.txt", 'w') as f:
        for i in range(N):
            for j in range(i + 1, N):
                f.write(f"{k_sp[i][j]}\n")
