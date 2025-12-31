#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARIES

import math
import numpy as np
import mdtraj as md 
import matplotlib.pyplot as plt

#IMPORTING FUNCTIONS
import func_read_parameter_files
import func_bond_fluctuation
import func_read_bondlist
import func_lammps_data
import func_comparison_MDvsCG
import func_rmsf_plots
import sys

path=sys.argv[1]   ############# path of working directory #############################

input_file_format=sys.argv[2]

if input_file_format=="xtc":
    traj2=md.load(path+'/INPUT/COARSE_GRAINED_MAPPED_TRAJECTORY.xtc', top=path+'/INPUT/REFERENCE.pdb')        ################## INPUT ##########################

if input_file_format=="lammpstrj":
    traj2=md.load(path+'/INPUT/COARSE_GRAINED_MAPPED_TRAJECTORY.lammpstrj', top=path+'/INPUT/REFERENCE.pdb')        ################## INPUT ##########################
no_frames_CG=traj2.n_frames             # no. of frames in MD trajectory
traj2.xyz=traj2.xyz*10
N=traj2.n_atoms   
parameters=func_read_parameter_files.value_of_parameters(path)
name=parameters["bondlist_name"]       ################## INPUT ########################## NAME OF THE BONDLIST
flag_identical_bonds=parameters["flag_identical_bonds"]  ###################### INPUT ############# ANALYSIS BASED ON IDENTICAL BONDS (average bondlength and average spring constant in each iteration)########
stride=parameters["stride"]          ################## INPUT ##########################
system=parameters["system"]  
############### CALCULATING MD FLUCTUATION ########################################### 
### CREATING/READING BONDLIST ########
bond_list,no_of_bonds=func_read_bondlist.gen_bondlist(N,name,path)
fluctuation_MD,bo_md,SIG_md=func_bond_fluctuation.bond_fluctuations(traj2,no_frames_CG,no_of_bonds,flag_identical_bonds,path,system,bond_list)

print("MD OVER")



traj_md_reference=md.load(path+'/INPUT/REFERENCE.pdb')

# COARSE GRAINED SIMULATION

#FOR IMPORTING LAMMPS TRAJECTORY AND TOPOLOGY FILE
######################################################  NOTE #############################################################

# The trajectories in LAMMPS are written in Angstroms (Ao) 1 Ao = 10^-10 m
# The trajectories uploaded using MDTRAJ are in nanometers
# We will convert the trajectory to nanometers to Angstroms by multiplying each coordinates by factor of 10

#########################################################################################################################

#traj5=md.load(path+'/POST_PROCESSING/COARSE_GRAINED_MAPPED_TRAJECTORY.xtc', top=path+'/INPUT/REFERENCE.pdb') 
print("readingMDTRAJECTORY")
traj5=md.load_lammpstrj(path+'/POST_PROCESSING/hENM_CG_trajectory.lammpstrj', top=path+'/INPUT/REFERENCE.pdb') 

traj5=traj5.superpose(traj5, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#traj5=traj5[::stride]
traj5=traj5[::100000]
traj5.xyz[:,:,:]=traj5.xyz[:,:,:]*10      # coordinates in Angstroms
                        
no_frames_CG_lmp=traj5.n_frames           # no. of frames in MD trajectory

#saving CG trajectory in PDB format
traj_CG_reference=traj5[0]
for i in range(0,N):
    traj_CG_reference.xyz[0,i,:]=np.mean(traj5.xyz[:,i,:],axis=0) 
       
traj_CG_reference.save(path+'/POST_PROCESSING/REFERENCE_CG.pdb')   # saving average structure

# CG FLUCTUATIONS (VARIANCE)

#flag_identical_bonds="no"
fluctuation_CG,bo_cg,SIG_cg=func_bond_fluctuation.bond_fluctuations(traj5,no_frames_CG_lmp,no_of_bonds,flag_identical_bonds,path,system,bond_list)

# COMPARISON MD VS CG
func_comparison_MDvsCG.comparison(fluctuation_MD,fluctuation_CG,traj2,traj5,no_of_bonds,N,no_frames_CG,bo_md,bo_cg,path)

#CALCULATION OF RMSF OF MD MAPPED SIMUATION AND CG SIMULATION

#RMSF PLOTS
# rmsf_superpose='yes'
# if rmsf_superpose=='yes':
#     print('rmsf_superpose')
#     MD_chain_A=traj2.atom_slice(traj2.topology.select('residue 1 to 10 and chainid 0') )
#     MD_chain_B=traj2.atom_slice(traj2.topology.select('residue 1 to 10 and chainid 1') )
#     MD_chain_C=traj2.atom_slice(traj2.topology.select('residue 1 to 10 and chainid 2') )
#     MD_chain_D=traj2.atom_slice(traj2.topology.select('residue 1 to 10 and chainid 3') )
#     MD_chain_E=traj2.atom_slice(traj2.topology.select('residue 1 to 10 and chainid 4') )
#     MD_chain_F=traj2.atom_slice(traj2.topology.select('residue 1 to 10 and chainid 5') )

#     CG_chain_A=traj5.atom_slice(traj5.topology.select('residue 1 to 10 and chainid 0') )
#     CG_chain_B=traj5.atom_slice(traj5.topology.select('residue 1 to 10 and chainid 1') )
#     CG_chain_C=traj5.atom_slice(traj5.topology.select('residue 1 to 10 and chainid 2') )
#     CG_chain_D=traj5.atom_slice(traj5.topology.select('residue 1 to 10 and chainid 3') )
#     CG_chain_E=traj5.atom_slice(traj5.topology.select('residue 1 to 10 and chainid 4') )
#     CG_chain_F=traj5.atom_slice(traj5.topology.select('residue 1 to 10 and chainid 5') )

#     MD_aligned_chain_A=MD_chain_A.superpose(MD_chain_A, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     MD_aligned_chain_B=MD_chain_B.superpose(MD_chain_B, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     MD_aligned_chain_C=MD_chain_C.superpose(MD_chain_C, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     MD_aligned_chain_D=MD_chain_D.superpose(MD_chain_D, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     MD_aligned_chain_E=MD_chain_E.superpose(MD_chain_E, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     MD_aligned_chain_F=MD_chain_F.superpose(MD_chain_F, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)

#     CG_aligned_chain_A=CG_chain_A.superpose(CG_chain_A, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     CG_aligned_chain_B=CG_chain_B.superpose(CG_chain_B, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     CG_aligned_chain_C=CG_chain_C.superpose(CG_chain_C, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     CG_aligned_chain_D=CG_chain_D.superpose(CG_chain_D, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     CG_aligned_chain_E=CG_chain_E.superpose(CG_chain_E, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
#     CG_aligned_chain_F=CG_chain_F.superpose(CG_chain_F, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)

#     MD_rmsf_chainA=md.rmsf(MD_aligned_chain_A, MD_aligned_chain_A, frame=0, atom_indices=None, parallel=True, precentered=False)
#     MD_rmsf_chainB=md.rmsf(MD_aligned_chain_B, MD_aligned_chain_B, frame=0, atom_indices=None, parallel=True, precentered=False)
#     MD_rmsf_chainC=md.rmsf(MD_aligned_chain_C, MD_aligned_chain_C, frame=0, atom_indices=None, parallel=True, precentered=False)
#     MD_rmsf_chainD=md.rmsf(MD_aligned_chain_D, MD_aligned_chain_D, frame=0, atom_indices=None, parallel=True, precentered=False)
#     MD_rmsf_chainE=md.rmsf(MD_aligned_chain_E, MD_aligned_chain_E, frame=0, atom_indices=None, parallel=True, precentered=False)
#     MD_rmsf_chainF=md.rmsf(MD_aligned_chain_F, MD_aligned_chain_F, frame=0, atom_indices=None, parallel=True, precentered=False)

#     CG_rmsf_chainA=md.rmsf(CG_aligned_chain_A, CG_aligned_chain_A, frame=0, atom_indices=None, parallel=True, precentered=False)
#     CG_rmsf_chainB=md.rmsf(CG_aligned_chain_B, CG_aligned_chain_B, frame=0, atom_indices=None, parallel=True, precentered=False)
#     CG_rmsf_chainC=md.rmsf(CG_aligned_chain_C, CG_aligned_chain_C, frame=0, atom_indices=None, parallel=True, precentered=False)
#     CG_rmsf_chainD=md.rmsf(CG_aligned_chain_D, CG_aligned_chain_D, frame=0, atom_indices=None, parallel=True, precentered=False)
#     CG_rmsf_chainE=md.rmsf(CG_aligned_chain_E, CG_aligned_chain_E, frame=0, atom_indices=None, parallel=True, precentered=False)
#     CG_rmsf_chainF=md.rmsf(CG_aligned_chain_F, CG_aligned_chain_F, frame=0, atom_indices=None, parallel=True, precentered=False)
    
#     ch_md_rmsf=[]
#     for ch in range(0,10):
#         arr1=[MD_rmsf_chainA[ch],MD_rmsf_chainB[ch],MD_rmsf_chainC[ch],MD_rmsf_chainD[ch],MD_rmsf_chainE[ch],MD_rmsf_chainF[ch]]
#         ch_md_rmsf.append(np.mean(arr1))


#     ch_cg_rmsf=[]

#     for ch_cg in range(0,10):
#         arr2=[CG_rmsf_chainA[ch_cg],CG_rmsf_chainB[ch_cg],CG_rmsf_chainC[ch_cg],CG_rmsf_chainD[ch_cg],CG_rmsf_chainE[ch_cg],CG_rmsf_chainF[ch_cg]]
#         ch_cg_rmsf.append(np.mean(arr2))

#     CG_sites_x_axis=np.arange(1,len(MD_rmsf_chainA)+1,1)
#     plt.plot(CG_sites_x_axis,ch_md_rmsf)
#     plt.plot(CG_sites_x_axis,ch_cg_rmsf)
#     plt.xlabel("CG_site")
#     plt.ylabel("RMSF_avg_over_time_over_all_six_chains")
#     plt.legend(['MD','CG'])
#     plt.plot(CG_sites_x_axis,MD_rmsf_chainA,color='red')
#     plt.plot(CG_sites_x_axis,MD_rmsf_chainB,color='red')
#     plt.plot(CG_sites_x_axis,MD_rmsf_chainC,color='red')
#     plt.plot(CG_sites_x_axis,MD_rmsf_chainD,color='red')
#     plt.plot(CG_sites_x_axis,MD_rmsf_chainE,color='red')
#     plt.plot(CG_sites_x_axis,MD_rmsf_chainF,color='red')

#     plt.plot(CG_sites_x_axis,CG_rmsf_chainA,color='black')
#     plt.plot(CG_sites_x_axis,CG_rmsf_chainB,color='black')
#     plt.plot(CG_sites_x_axis,CG_rmsf_chainC,color='black')
#     plt.plot(CG_sites_x_axis,CG_rmsf_chainD,color='black')
#     plt.plot(CG_sites_x_axis,CG_rmsf_chainE,color='black')
#     plt.plot(CG_sites_x_axis,CG_rmsf_chainF,color='black')

#     plt.show()


# rmsf_superpose='no'
# if rmsf_superpose=='no':

rmsf_value_md=md.rmsf(traj2, traj_md_reference, frame=0, atom_indices=None, parallel=True, precentered=False)

rmsf_value_cg=md.rmsf(traj5, traj_CG_reference, frame=0, atom_indices=None, parallel=True, precentered=False)

func_rmsf_plots.rmsf_plots(rmsf_value_md,(rmsf_value_cg),N,path)








