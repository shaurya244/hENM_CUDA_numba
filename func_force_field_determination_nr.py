#!/usr/bin/env python
# coding: utf-8

# In[1]:

import func_fas_identical_bonds_for_any_quantity
from lammps import lammps
import gc 
#import time
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import linalg as LA
import csv
#import xlsxwriter
import func_mspa_identical_bonds_for_any_quantity
import func_lammps_data
import mdtraj as md
import func_fas_identical_bonds_for_any_quantity



collected = gc.collect() 
def NMA(N,bond_list,fluctuation_MD,T,traj4,mass_weights):
    hessian=np.zeros((3*N,3*N))
    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1                      
        for k in range(0,3):                    
            for l in range(0,3): 
                hessian[(3*i)+k][(3*j)+l]=(((-  bond_list[ids][2]))/ ((bond_list[ids][3]**2)*(math.sqrt(mass_weights[i]))*(math.sqrt(mass_weights[j]))))*(traj4.xyz[0,i,k]-traj4.xyz[0,j,k])*(traj4.xyz[0,i,l]-traj4.xyz[0,j,l])         
                hessian[(3*j)+l][(3*i)+k]=hessian[(3*i)+k][(3*j)+l]
    print("off diagonal")
    for i in range(0,N):    
        for k in range(0,3):     #along rows
            for l in range(0,3): #along columns
                value=0
                for j in range(0,N):

                    if i != j:                                 
                        value=value+hessian[(3*i)+k][(3*j)+l]  
                hessian[(3*i)+k][(3*i)+l]=-value               
    print("diagonal")
    w,v=LA.eig(hessian)      
    v_edit=[]
    w_edit=[]
    for i in range(len(w)):
        #print(w[i])
        if w[i] > 0.00001:  #eliminating any eigen value less than 10^-5
            v_edit.append(v[:,i])
            w_edit.append(w[i])
    w=np.array(w_edit)
    v=np.array(v_edit)
    v = np.transpose(v)
    del v_edit
    del w_edit
    del hessian
    if len(w)!= (3*N)-6:
        print("MORE THAN 6 EIGEN VALUES ARE LESS SMALLER THAN 10^-5")
    fluctuation_NMA=np.zeros((len(bond_list)))
    kB=8.314462618*0.001  
    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1  
        DELTA_i_j=0
        for k in range (0,(3*N)-6):
            #mass_weighted
            DELTA_i_j=DELTA_i_j+   (   (1/math.sqrt(w.real[k]))* (
                                        (traj4.xyz[0,i,0]-traj4.xyz[0,j,0])*((v.real[3*i][k]/(math.sqrt(mass_weights[i])))-(v.real[3*j][k]/(math.sqrt(mass_weights[j]))))+
                                        (traj4.xyz[0,i,1]-traj4.xyz[0,j,1])*((v.real[3*i+1][k]/(math.sqrt(mass_weights[i])))-(v.real[3*j+1][k]/(math.sqrt(mass_weights[j]))))+
                                        (traj4.xyz[0,i,2]-traj4.xyz[0,j,2])*((v.real[3*i+2][k]/(math.sqrt(mass_weights[i])))-(v.real[3*j+2][k]/(math.sqrt(mass_weights[j]))))
                                                                  )    
                                    )**2 
        
        fluctuation_NMA[ids]=kB*T*DELTA_i_j* ((1/bond_list[ids][3])**2)

    error=np.zeros((len(bond_list)))
    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1      

        error[ids]=((fluctuation_NMA[ids]-fluctuation_MD[ids]))   
    collected = gc.collect()

    del fluctuation_NMA
    
    return v,w,error

def force_constants_nma(N,T,ALPHA,beta,max_itr,bond_list,fluctuation_MD,run,tolerance,path,flag_identical_bonds,flag_automate_ALPHA_beta_values,iterative_method,system,traj4,box_size,mass_weights):
    SUM_OF_SQUARE_ERROR=[]
    #RESIDUAL=[]
    STEP=[]
    itr=[]
    iteration=0
    count_itr=0
    var=0
    start_SOE=0
    sourceFile = open(path+'/POST_PROCESSING/array_kzeros%s.txt'%(run), 'w')


    for iteration in range(0,max_itr):
         #start = time.time()
        if flag_automate_ALPHA_beta_values=='yes':
            ALPHA=1/(max_itr+1)
            beta=1/(max_itr+1)
        
        bond_list_incre=np.zeros((len(bond_list),4))
        # appending lines take into account that no spring constant is zero and have some finite value
        for ids,bond in enumerate (bond_list):
            bond_list_incre[ids][0]=bond_list[ids][0]
            bond_list_incre[ids][1]=bond_list[ids][1]
            bond_list_incre[ids][2]=bond_list[ids][2]
            bond_list_incre[ids][3]=bond_list[ids][3]

        print("NMA_calculation1")
        v_,w_,error_ksp=NMA_GPU(N,bond_list,fluctuation_MD,T,traj4,mass_weights)

          
        K_new=np.zeros((len(bond_list)))
        square_of_error=np.zeros((len(bond_list)))
        print("shape of K_new", K_new.shape)

        print("The iteration number is",iteration)

        for ids,bond in enumerate (bond_list):
            print(ids)
            i,j=int(bond[0])-1,int(bond[1])-1                 
            if iterative_method=='NR_method':
                bond_list_incre[ids][2]=bond_list_incre[ids][2]+ALPHA
                v_incre,w_incre,error_ksp_incre=NMA_GPU(N,bond_list_incre,fluctuation_MD,T,traj4,mass_weights)        
                K_new[ids]=bond_list[ids][2]-(ALPHA*beta*error_ksp[ids]**2)/(error_ksp_incre[ids]**2-error_ksp[ids]**2)   



        for ids,bond in enumerate (bond_list):
            bond_list[ids][2]=K_new[ids]

               
            square_of_error[ids]=(error_ksp[ids])**2

              

        ##################################################################################################################
        
        # SUM OF SQUARE OF ERROR
        sum_of_sq_error=np.sum(square_of_error)
        print("sum of square of error=",sum_of_sq_error)
        
        SUM_OF_SQUARE_ERROR.append(sum_of_sq_error)
        count_itr=iteration
        itr.append(count_itr)

        ######################################## PLOTING SUM OF SQUARE OF ERROR ###################################################

        plt.plot(np.array(itr), np.array(SUM_OF_SQUARE_ERROR),linestyle='solid')

        
        # naming the x axis
        plt.xlabel('iteration')
        # naming the y axis
        plt.ylabel('sum of square of error')
        # giving a title to graph
        plt.title('CONVERGENCE PLOT')
        #saving the figure       

                        
        #######################################  SAVING TEXT FILES ###############################################################

        file=open(path+'/POST_PROCESSING/sum_of_square_of_error%s.txt'%(run),"w")
        np.savetxt(file,SUM_OF_SQUARE_ERROR)
        file.close()

        file=open(path+'/POST_PROCESSING/no_of_iteration%s.txt'%(run),"w")
        np.savetxt(file,itr)
        file.close()

          
        ############################################################################################################################



        if flag_identical_bonds=="yes":
            if system=='mspa':
                k_sp=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
                #dr_sq=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds(dr_sq,path)
            if system=='fas':
                k_sp=func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
                #dr_sq=func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds(dr_sq,path)
            
        if iteration % 10 == 0:  # Save every 100 iterations
            np.savetxt(path+'/POST_PROCESSING/K_values_%d.csv'%(iteration),bond_list, delimiter = ",")    
            plt.plot(itr, SUM_OF_SQUARE_ERROR,linestyle='solid')
            
            # naming the x axis
            plt.xlabel('iteration')
            # naming the y axis
            plt.ylabel('sum of square of error')
            # giving a title to graph
            plt.title('CONVERGENCE PLOT')
            #saving the figure       
  

  
        #appending lines take into account if any spring constant is negative and make it zero after each iteration
        counting=0
        #bonds_k_zeros=[]
        for ids,bond in enumerate (bond_list):            
            i,j=int(bond[0])-1,int(bond[1])-1 
            if bond_list[ids][2]<=0:    
 
                bond_list[ids][2]=0
                bond_list[ids][2]=0
                counting=counting+1
                #bonds_k_zeros.append(bond_type)
                print(i,j,iteration,file=sourceFile)
                   

        print("NUMBER OF k's equal to zero", counting)

        #saving non zero spring constant
        bonds_k_non_zeros=[]
        for ids,bond in enumerate (bond_list):            
            i,j=int(bond[0])-1,int(bond[1])-1 
            bond_type=[i+1,j+1]
            if bond_list[ids][2]>0:      
                bonds_k_non_zeros.append(bond_type)

        if iteration % 10 == 0:  # Save every 100 iterations
           
        	
            file = open(path+'/POST_PROCESSING/bonds_k_non_zeros_%d.csv'%(iteration), 'w+', newline ='')
            with file:       
                write = csv.writer(file)
                write.writerows(bonds_k_non_zeros)


        
        if sum_of_sq_error<tolerance:   #Tolerance can be used after running multiple run for a system as it changes with ALPHA, beta values                                            choosen
            print("sum_of_sq_error", sum_of_sq_error)
            break
        start_SOE=sum_of_sq_error
            
#24 NOV 2024   
   
        del sum_of_sq_error
        

        func_lammps_data.lammps_data_file(N,bond_list,mass_weights,traj4,box_size,path,system,flag_identical_bonds )
        print("lammps_start")
        lmp = lammps()
        lmp.file(path+"/INPUT/energy_min.in")
        #lmp.command("run 1000")
        lmp.command("log log_%d.lammps")
        print("lammps_end")
        
        traj4 = md.load_lammpstrj(path+'/POST_PROCESSING/em.lammpstrj',top=path+'/INPUT/REFERENCE.pdb')
        traj4.xyz[:,:,:]=traj4.xyz[:,:,:]*10         


        
          # list storing values of all Rij's i.e. Rij= sqrt ((xi -xj)^2 + (yi -yj)^2 +(zi -zj)^2)) 

        if flag_identical_bonds=='no':
            print("flag_identical_bonds=no")
            for ids,bond in enumerate (bond_list):
                i,j=int(bond[0])-1,int(bond[1])-1   
                bond_list[ids][3]=math.sqrt(np.sum((traj4.xyz[0,i,:]-traj4.xyz[0,j,:])**2))



        if flag_identical_bonds=='yes':

            print("flag_identical_bonds=yes")
            dr_sq1=np.zeros((N,N))
            for bond in bond_list:
                i,j=bond[0]-1,bond[1]-1
                dr_sq1[i][j]=math.sqrt(np.sum((traj4.xyz[0,i,:]-traj4.xyz[0,j,:])**2))        


    return bond_list


################################# PLOTS AND SAVING DATA ########################################################################
# import numpy as np
# from numba import cuda
# def NMA_GPU(
#     N,
#     bond_list,
#     fluctuation_MD,
#     T,
#     traj4,
#     mass_weights
# ):
#     """
#     GPU-accelerated Normal Mode Analysis
#     Returns: v, w, error
#     """

#     # -------------------------
#     # Extract & move data
#     # -------------------------
#     coords = np.asarray(traj4.xyz[0], dtype=np.float64)
    
#     mass = np.asarray(mass_weights, dtype=np.float64)

#     bond_i = np.asarray(bond_list[:,0]-1, dtype=np.int32)
#     bond_j = np.asarray(bond_list[:,1]-1, dtype=np.int32)
#     k_spring = np.asarray(bond_list[:,2], dtype=np.float64)
#     k_spring = np.ascontiguousarray(k_spring, dtype=np.float64)
#     r0 = np.asarray(bond_list[:,3], dtype=np.float64)
#     r0 = np.ascontiguousarray(r0, dtype=np.float64)
#     coords_dev = cuda.to_device(coords)
#     mass_dev = cuda.to_device(mass)
#     bond_i_dev = cuda.to_device(bond_i)
#     bond_j_dev = cuda.to_device(bond_j)
#     k_spring_dev = cuda.to_device(k_spring)
#     r0_dev = cuda.to_device(r0)

#     B = bond_list.shape[0]
#     M = 3 * N

#     hessian = np.zeros((M, M), dtype=np.float64)
#     hessian_dev = cuda.to_device(hessian)
#     # -------------------------
#     # Hessian off-diagonal
#     # -------------------------
#     threads = 128
#     blocks = (B + threads - 1) // threads

#     hessian_offdiag_kernel[blocks, threads](
#         bond_i_dev, bond_j_dev, k_spring_dev, r0_dev,
#         coords_dev, mass_dev, hessian_dev
#     )

#     # -------------------------
#     # Hessian diagonal
#     # -------------------------
#     threads = 128
#     blocks = (M + threads - 1) // threads

#     hessian_diagonal_kernel[blocks, threads](hessian_dev, N)

#     # -------------------------
#     # Eigen decomposition (GPU)
#     # -------------------------
#     hessian_cpu = np.array(hessian_dev.copy_to_host(), dtype=np.float64)
#     w, v = np.linalg.eigh(hessian_cpu)

#     # -------------------------
#     # Remove zero modes
#     # -------------------------
#     mask = w > 1e-5
#     w = w[mask]
#     v = v[:, mask]
    
#     if w.shape[0] != (3*N - 6):
#         print("WARNING: unexpected number of modes")
#     w_dev = cuda.to_device(w)
#     v_dev = cuda.to_device(v)
#     # -------------------------
#     # Fluctuation computation
#     # -------------------------
#     fluct_gpu = np.zeros(B, dtype=np.float64)
#     fluct_gpu_dev = cuda.to_device(fluct_gpu)
#     threads = 128
#     blocks = (B + threads - 1) // threads

#     fluctuation_kernel[blocks, threads](
#         bond_i_dev, bond_j_dev,
#         coords_dev, mass_dev,
#         v_dev, w_dev,
#         r0_dev, T,
#         fluct_gpu_dev
#     )

#     # -------------------------
#     # Error
#     # -------------------------
#     fluct_cpu = np.array(fluct_gpu_dev.copy_to_host(), dtype=np.float64)
#     error = np.zeros(B, dtype=np.float64)
#     error = fluct_cpu - np.asarray(fluctuation_MD)

#     return v, w, error

# @cuda.jit
# def fluctuation_kernel(
#     bond_i, bond_j, coords, mass,
#     v, w, r0, T, out
# ):
#     bid = cuda.grid(1)
#     if bid >= bond_i.shape[0]:
#         return

#     i = bond_i[bid]
#     j = bond_j[bid]

#     delta = 0.0
#     for k in range(w.shape[0]):
#         inv = 1.0 / math.sqrt(w[k])
#         s = 0.0
#         for d in range(3):
#             s += (coords[i,d]-coords[j,d]) * (
#                 v[3*i+d,k]/math.sqrt(mass[i]) -
#                 v[3*j+d,k]/math.sqrt(mass[j])
#             )
#         delta += (inv * s) ** 2

#     kB = 8.314462618e-3
#     out[bid] = kB * T * delta * (1.0 / r0[bid])**2
# @cuda.jit
# def hessian_diagonal_kernel(hessian, N):
#     idx = cuda.grid(1)
#     if idx >= 3*N:
#         return

#     i = idx // 3
#     k = idx % 3

#     for l in range(3):
#         val = 0.0
#         for j in range(N):
#             if j != i:
#                 val += hessian[3*i+k, 3*j+l]
#         hessian[3*i+k, 3*i+l] = -val
# @cuda.jit
# def hessian_offdiag_kernel(
#     bond_i, bond_j, k_spring, r0,
#     coords, mass, hessian
# ):
#     bid = cuda.grid(1)
#     if bid >= bond_i.shape[0]:
#         return

#     i = bond_i[bid]
#     j = bond_j[bid]

#     mi = math.sqrt(mass[i])
#     mj = math.sqrt(mass[j])

#     for k in range(3):
#         dxk = coords[i, k] - coords[j, k]
#         for l in range(3):
#             dxl = coords[i, l] - coords[j, l]
#             val = (-k_spring[bid]) / (r0[bid]**2 * mi * mj) * dxk * dxl

#             hessian[3*i+k, 3*j+l] = val
#             hessian[3*j+l, 3*i+k] = val

import numpy as np
from numba import cuda
import math

def NMA_GPU(N, bond_list, fluctuation_MD, T, traj4, mass_weights):
    M = 3 * N
    # Ensure data types are strictly correct for the GPU
    coords = np.ascontiguousarray(traj4.xyz[0], dtype=np.float64)
    mass = np.ascontiguousarray(mass_weights, dtype=np.float64)
    
    # Crucial: Ensure indices are 0-based and within [0, N-1]
    bond_i = (bond_list[:, 0] - 1).astype(np.int32)
    bond_j = (bond_list[:, 1] - 1).astype(np.int32)
    k_spring = bond_list[:, 2].astype(np.float64)
    r0 = bond_list[:, 3].astype(np.float64)

    # Initialize Hessian on device to avoid unnecessary transfers
    hessian_dev = cuda.to_device(np.zeros((M, M), dtype=np.float64))
    
    # Transfers
    coords_dev = cuda.to_device(coords)
    mass_dev = cuda.to_device(mass)
    bond_i_dev = cuda.to_device(bond_i)
    bond_j_dev = cuda.to_device(bond_j)
    k_spring_dev = cuda.to_device(k_spring)
    r0_dev = cuda.to_device(r0)

    # 1. Compute Hessian
    threads = 128
    blocks = (bond_i.shape[0] + threads - 1) // threads
    hessian_combined_kernel[blocks, threads](
        bond_i_dev, bond_j_dev, k_spring_dev, r0_dev,
        coords_dev, mass_dev, hessian_dev
    )

    # 2. Eigen-decomposition on CPU
    hessian_cpu = hessian_dev.copy_to_host()
    w, v = np.linalg.eigh(hessian_cpu)

    # 3. Filter modes (Keep only vibrations)
    mask = w > 1e-5
    w_filtered = w[mask]
    v_filtered = v[:, mask]
    
    w_dev = cuda.to_device(w_filtered)
    v_dev = cuda.to_device(v_filtered)

    # 4. Compute Fluctuations
    fluct_dev = cuda.device_array(bond_i.shape[0], dtype=np.float64)
    fluctuation_kernel_optimized[blocks, threads](
        bond_i_dev, bond_j_dev, coords_dev, mass_dev,
        v_dev, w_dev, r0_dev, T, fluct_dev
    )

    fluct_cpu = fluct_dev.copy_to_host()
    error = fluct_cpu - np.asarray(fluctuation_MD)

    return v_filtered, w_filtered, error
@cuda.jit
def hessian_combined_kernel(bond_i, bond_j, k_spring, r0, coords, mass, hessian):
    """
    Computes off-diagonal and diagonal components in one pass.
    Uses atomics to ensure thread safety.
    """
    bid = cuda.grid(1)
    if bid >= bond_i.shape[0]:
        return

    i = bond_i[bid]
    j = bond_j[bid]

    # Pre-calculate mass scaling
    inv_mass_prod = 1.0 / math.sqrt(mass[i] * mass[j])
    # The negative sign and constants
    coeff = -k_spring[bid] / (r0[bid]**2) * inv_mass_prod

    for k in range(3):
        dxk = coords[i, k] - coords[j, k]
        for l in range(3):
            dxl = coords[i, l] - coords[j, l]
            val = coeff * dxk * dxl

            # 1. Off-diagonal elements (i, j)
            cuda.atomic.add(hessian, (3*i + k, 3*j + l), val)
            cuda.atomic.add(hessian, (3*j + l, 3*i + k), val)

            # 2. Diagonal accumulation: H_ii = sum(-H_ij)
            # We subtract val here because val is already negative
            cuda.atomic.add(hessian, (3*i + k, 3*i + l), -val)
            cuda.atomic.add(hessian, (3*j + k, 3*j + l), -val)

@cuda.jit
def fluctuation_kernel_optimized(bond_i, bond_j, coords, mass, v, w, r0, T, out):
    bid = cuda.grid(1)
    if bid >= bond_i.shape[0]:
        return

    i = bond_i[bid]
    j = bond_j[bid]
    num_modes = w.shape[0]

    delta = 0.0
    for k in range(num_modes):
        # 1/w_k * (projection)^2
        # s = (Ri - Rj) dot (vi/sqrt(mi) - vj/sqrt(mj))
        s = 0.0
        for d in range(3):
            diff_vec = (coords[i, d] - coords[j, d])
            mode_diff = (v[3*i + d, k] / math.sqrt(mass[i]) - 
                         v[3*j + d, k] / math.sqrt(mass[j]))
            s += diff_vec * mode_diff
        
        delta += (s * s) / w[k]

    kB = 8.314462618e-3
    # Resulting fluctuation
    out[bid] = (kB * T * delta) / (r0[bid]**2)