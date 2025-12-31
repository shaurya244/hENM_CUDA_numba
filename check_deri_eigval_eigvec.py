import numpy as np 
from func_deri_eigen_val_vec import deri_eign_val
from func_deri_eigen_val_vec import deri_eign_vec
from func_force_field_determination import NMA
import matplotlib.pyplot as plt
def check_deri_eigen_val(N,bond_list,traj4,mass_weights,i,j,k,fluctuation_MD,T):
    perturbations = np.linspace(0.001,0.0000001,100)
    delta = np.zeros((len(perturbations),1))
    for z in range (len(perturbations)):
        v_old,w_old,error = NMA(N,bond_list,fluctuation_MD,T,traj4,mass_weights)
        print("v_old dimensions : ",v_old.shape)
        bond_list[:,2] = bond_list[:,2]+perturbations[z]
        v_new,w_new,error = NMA(N,bond_list,fluctuation_MD,T,traj4,mass_weights)
        del_W_del_K_ana = (w_new[k]-w_old[k])/perturbations[z]
        del_W_del_K_num = deri_eign_val(i,j,v_old,k,traj4,mass_weights,bond_list)
        delta[z] = np.abs(del_W_del_K_num-del_W_del_K_ana)
    plt.plot(perturbations,delta)
    plt.xlabel("perturbations")
    plt.ylabel("diff b\w analytical and numerical")
    plt.title("convergence of eigen value derivative")
    plt.show()
    print("delta",delta[z])
def check_deri_eigen_vec(N,bond_list,traj4,mass_weights,i,j,k,fluctuation_MD,T):
    perturbations = np.linspace(0.001,0.0000001,100)
    delta = np.zeros((len(perturbations),1))
    for z in range (len(perturbations)):
        v_old,w_old,hessian = NMA(N,bond_list,fluctuation_MD,T,traj4,mass_weights)
        print("v_old dimensions : ",v_old.shape)
        bond_list[:,2] = bond_list[:,2]+perturbations[z]
        v_new,w_new,hessian = NMA(N,bond_list,fluctuation_MD,T,traj4,mass_weights)
        del_v_del_K_ana = (v_new[:,k]-v_old[:,k])/perturbations[z]
        del_v_del_K_num = deri_eign_vec(v_old,w_old,i,j,k,traj4,mass_weights,bond_list,N)
        delta[z] = np.linalg.norm(del_v_del_K_num-del_v_del_K_ana)    
    plt.plot(perturbations,delta)
    plt.xlabel("perturbations")
    plt.ylabel("diff b\w analytical and numerical")
    plt.title("convergence of eigen vectors derivative")
    plt.show()


