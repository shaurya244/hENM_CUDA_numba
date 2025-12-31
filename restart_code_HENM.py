#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARIES

import math
import numpy as np
import mdtraj as md 
import matplotlib.pyplot as plt

#IMPORTING FUNCTIONS
import func_read_spring_constant_excel_file
import func_force_field_determination
import func_bond_fluctuation
import func_read_bondlist
import func_boltzmann_inversion
import func_HENMplots
import func_lammps_data
import func_lammps_input_file
import func_read_parameter_files
import sys
import func_fas_identical_bonds_for_any_quantity
import func_mspa_identical_bonds_for_any_quantity
from force_constant_jacobian import force_constant_jacobian
path=sys.argv[1]
run=sys.argv[2]
#FOR IMPORTING GROMACS TRAJECTORY AND TOPOLOGY FILE

input_file_format=sys.argv[3]

if input_file_format=="xtc":
    traj2=md.load(path+'/INPUT/COARSE_GRAINED_MAPPED_TRAJECTORY.xtc', top=path+'/INPUT/REFERENCE.pdb')        ################## INPUT ##########################

if input_file_format=="lammpstrj":
    traj2=md.load(path+'/INPUT/COARSE_GRAINED_MAPPED_TRAJECTORY.lammpstrj', top=path+'/INPUT/REFERENCE.pdb')        ################## INPUT 
    
no_frames_CG=traj2.n_frames             # no. of frames in MD trajectory
traj2.xyz=traj2.xyz*10                  # ALL DISTANCE UNITS ARE IN ANGSTROM
N=traj2.n_atoms                         # NUMBER OF CG SITES

parameters=func_read_parameter_files.value_of_parameters(path)
iterative_method=parameters["iterative_method"] ################## INPUT ######### 
flag_automate_ALPHA_beta_values=parameters["flag_automate_ALPHA_beta_values"] ################## INPUT ######### 
ALPHA=parameters["ALPHA"]              ################## INPUT ##########################
beta=parameters["beta"]                ################## INPUT ##########################
max_itr=parameters["max_itr"]   
#print(max_itr)################## INPUT ##########################
tolerance=parameters["tolerance"]      ################## INPUT ########################## 
T=parameters["T"]                      ################## INPUT ########################## TEMPERATURE 
name=parameters["bondlist_name"]       ################## INPUT ########################## NAME OF THE BONDLIST
flag_identical_bonds=parameters["flag_identical_bonds"]  ####### INPUT #### ANALYSIS BASED ON IDENTICAL BONDS (average bondlength and average spring constant in each iteration)########
count_run_value=parameters["count_run_value"]          ################## INPUT ##########################
initial_guess=parameters["initial_guess"]##################################################### FOR CG_HENM LAMMPS INPUT FILE ##########################
##################################################### FOR CG_HENM LAMMPS INPUT FILE ##########################
system=parameters["system"]  
damp=parameters["lammps_damp"]                              ################## INPUT ############# DAMP VALUE FOR LAMMPS INPUT FILE 
dump_value=parameters["lammps_dump_value"]                  ################## INPUT ############# FREQUENCY OF DUMPING COORDINATES FOR LAMMPS INPUT FILE 
run_value=parameters["lammps_run_value"]                    ################## INPUT ############# RUN FOR LAMMPS INPUT FILE 
box_size=parameters["lammps_box_size"]                      ################## INPUT ############# DIMENSION OF BOX SIZE FOR LAMMPS INPUT FILE
method=parameters["method"]  
############### CALCULATING CG_mapped_MD FLUCTUATION ########################################### 
print("calculate MD fluctuation start")
mass_weights = np.loadtxt(path+'/INPUT/mass_weights.txt', dtype=float)

bond_list,no_of_bonds=func_read_bondlist.gen_bondlist(N,name,path)

fluctuation_MD,bo_md,SIG_md=func_bond_fluctuation.bond_fluctuations(traj2,no_frames_CG,no_of_bonds,flag_identical_bonds,path,system,bond_list)
print("end")
################ CREATING/READING BONDLIST #######################################################################
print("The number of bonds in run 1",no_of_bonds) 

print("MD OVER")

################################### NOTE ###########################################################################################################

# The input for normal mode analysis is average structure of the CG mapped coordinates.

#############################################################################################################################################

traj4 = md.load(path+'/INPUT/REFERENCE.pdb') #### input ######              # reading the energy minimized coordinates 

no_frames_em=traj4.n_frames              #number of frames   


############## CALCULATING EQUILIBRIUM BOND LENGTH FOR ALL FRAMES #############################################3

# list storing values of all Rij's i.e. Rij= sqrt ((xi -xj)^2 + (yi -yj)^2 +(zi -zj)^2))
traj4.xyz[:,:,:]=traj4.xyz[:,:,:]*10  # Distance in Angstrom #

bond_list=func_read_spring_constant_excel_file.read_spring_constant(path,bond_list)


#convergence step 
#calculating the bond distance in 3D for each bond based identical bonds or without
 # list storing values of all Rij's i.e. Rij= sqrt ((xi -xj)^2 + (yi -yj)^2 +(zi -zj)^2)) 
if flag_identical_bonds=='no':
    print("flag_identical_bonds=no")
    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1   
        # dr_sq[i][j]=math.sqrt((R[i][0]-R[j][0])**2+(R[i][1]-R[j][1])**2+(R[i][2]-R[j][2])**2)
        bond_list[ids][3]=math.sqrt(np.sum((traj4.xyz[0,i,:]-traj4.xyz[0,j,:])**2))
        
        
if flag_identical_bonds=='yes':

    print("flag_identical_bonds=yes")
    dr_sq1=np.zeros((N,N))
    for bond in bond_list:
        i,j=bond[0]-1,bond[1]-1
        dr_sq1[i][j]=math.sqrt(np.sum((traj4.xyz[0,i,:]-traj4.xyz[0,j,:])**2))
        #dr_sq1[j][i]=dr_sq1[i][j]
    #np.savetxt('C:/Users/HHK LAB/WORK CODE HENM/DESKTOP_HENM/MSPA_1_micros/POST_PROCESSING/distance1.csv', dr_sq1, delimiter = ",")    
    
    #k_sp=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
    #dr_sq=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(dr_sq1,path)
    #np.savetxt('C:/Users/HHK LAB/WORK CODE HENM/DESKTOP_HENM/MSPA_1_micros/POST_PROCESSING/distance2.csv', dr_sq, delimiter = ",")
    dr_sq=dr_sq1
    if system=='mspa':
        print(system)
        k_sp=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
        #dr_sq=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(dr_sq1,path)
        #print(k_sp)
        #print(dr_sq)


    if system=='fas':
        print(system)
        k_sp=func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
        #dr_sq=func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(dr_sq1,path)  




#convergence step 1
#run=
if method =="iterative":
    bond_list=func_force_field_determination.force_constants(N,T,ALPHA,beta,max_itr,bond_list,fluctuation_MD,run,tolerance,path,flag_identical_bonds,flag_automate_ALPHA_beta_values,iterative_method,system,traj4, box_size,mass_weights)
if method =="jacobian":
    bond_list=force_constant_jacobian(bond_list,N,fluctuation_MD,traj4,T,mass_weights,max_itr)
np.savetxt(path+'/POST_PROCESSING/K_values_final.csv', bond_list, delimiter = ",") 

#HENM PLOTS
#func_HENMplots.HENM_plots(N,k_sp,bo_md,bond_list,dr_sq,path)

#GENERATE LAMMPS INPUT FILE
func_lammps_input_file.lammps_input_file_gen(T,damp,run_value,dump_value,path,count_run_value)

#LAMMPS DATA FILE
func_lammps_data.lammps_data_file(N,bond_list,mass_weights,traj4,box_size,path,system,flag_identical_bonds)

