import numpy as xp

# i and j represents the atom numbers for which we will be calculating the derivative of the hessian like dH_dkij
# so i and j here help us to locate the postion in the hessian correctly 
def derivative_hessian(bond_list,i,j,traj4,mass_weights):
    H_ij = xp.zeros((3,3))
    deri_H = xp.zeros((6,6))
    for k in range(0,3):
        for l in range (0,3):
            H_ij[k,l]=(-1/ (((bond_list[(bond_list[:,0]==(i+1)) & (bond_list[:,1]==(j+1)) ,3])**2)* 
                            (xp.sqrt(mass_weights[i]))*(xp.sqrt(mass_weights[j]))))*(traj4.xyz[0,i,k]-traj4.xyz[0,j,k])*(traj4.xyz[0,i,l]-traj4.xyz[0,j,l])
    deri_H[0:3,3:6] = H_ij
    deri_H[3:6,0:3] = H_ij
    deri_H[0:3,0:3] = -H_ij
    deri_H[3:6,3:6] = -H_ij
    return deri_H
