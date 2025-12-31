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

    #FOR ELEMENTS OF OFF DIAGONAL SUB MATRICES
    
    #NO OF COLUMNS AND ROWS OF HESSIAN MATRIX (3NX3N) where N is the number if particles
    
    #HESSIAN MATRIX (3NX3N)  where N is the number if particles
    
    hessian=np.zeros((3*N,3*N))

    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1                      
        for k in range(0,3):     #ALONG THE ROWS                
            for l in range(0,3): #ALONG THE COLUMNS
                #off diagonal elements= (-kij/drij^2)*(x_i-xj)*(x_i-xj)
                #hessian[(3*i)+k][(3*j)+l]=(((-spring_constant[i][j]))/ (dr_sq[i][j]**2))*(traj4.xyz[0,i,k]-traj4.xyz[0,j,k])*(traj4.xyz[0,i,l]-traj4.xyz[0,j,l])                #*(R[i][k]-R[j][k])*(R[i][l]-R[j][l]) 
                hessian[(3*i)+k][(3*j)+l]=(((-  bond_list[ids][2]))/ ((bond_list[ids][3]**2)*(math.sqrt(mass_weights[i]))*(math.sqrt(mass_weights[j]))))*(traj4.xyz[0,i,k]-traj4.xyz[0,j,k])*(traj4.xyz[0,i,l]-traj4.xyz[0,j,l])                #*(R[i][k]-R[j][k])*(R[i][l]-R[j][l]) 

                hessian[(3*j)+l][(3*i)+k]=hessian[(3*i)+k][(3*j)+l]
    # print("off diagonal")

    #ADDING ELEMENTS OF DIAGONAL SUB MATRICES IN HESSIAN MATRIX
    for i in range(0,N):    
        for k in range(0,3):     #along rows
            for l in range(0,3): #along columns
                value=0
                for j in range(0,N):

                    if i != j:                                     #ELEMINATING THE TERMS WHERE i=j as k11,k22..... are not considered
                        value=value+hessian[(3*i)+k][(3*j)+l]      #summation of corresponding off diagonal to obtain the respective diagonal elements of sub matrices
                hessian[(3*i)+k][(3*i)+l]=-value               # negative of summation value is the diagonal elements of diagonal sub matrix    
    # print("diagonal")
                    #diagonal elements=- summation (off -diagonal elements)

    ############################################  OUTPUT ########################################################################
    # WRITING THE HESSIAN MATRIX IN AN EXCEL FILE
#     with xlsxwriter.Workbook('Hessian_mspa.xlsx') as workbook:
#         worksheet = workbook.add_worksheet()
#         for row_num, data in enumerate(hessian):
#             worksheet.write_row(row_num, 0, data)
#    np.savetxt('C:/Users/HHK LAB/WORK CODE HENM/DESKTOP_HENM/MSPA_1_micros/POST_PROCESSING/distance.csv', dr_sq, delimiter = ",")
#    np.savetxt('C:/Users/HHK LAB/WORK CODE HENM/DESKTOP_HENM/MSPA_1_micros/POST_PROCESSING/hessian.csv', hessian, delimiter = ",")     
    ###############################################################################################################################

    #FOR EIGEN VALUES AND EIGEN VECTORS OF HESSIAN MATRIX 
    # The output from this library: w[i] is the eigen value and v[:,i] is the corresponding eigen vector
    #print(hessian)
    w,v=LA.eig(hessian)      
    #print(w)
    #print(v)
    ################################################ NOTE ############################################################################
    
    # Eliminating the eigen values (and corresponding eigen vector) which are nearly or equal to zero
    # These six eigen values corresponds to rotational and traslation motion of the system
    #There are 3N-6 NON-ZERO eigen values.
    
    ##################################################################################################################################  
    
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
    #print(w.shape)
    #print(v.shape)
    #print(v)
    ######################################## NOTE #######################################################

    # w[i] is the eigen value and v[i] is the corresponding vector.

    ######################################################################################################

    if len(w)!= (3*N)-6:
        print("MORE THAN 6 EIGEN VALUES ARE LESS SMALLER THAN 10^-5")


    #sum_over_modes=[]
    fluctuation_NMA=np.zeros((len(bond_list))) #NMA FLUCTUATION MATRIX
    kB=8.314462618*0.001  #kJ/mol K
    #kB=1.987204259*0.001 #kCal/mol K

    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1  
        DELTA_i_j=0
          # The normal mode analysis is carried out based on the bondlist that is provided by the user
        for k in range (0,(3*N)-6):
            #mass_weighted
            DELTA_i_j=DELTA_i_j+   (   (1/math.sqrt(w.real[k]))* (
                                        (traj4.xyz[0,i,0]-traj4.xyz[0,j,0])*((v.real[3*i][k]/(math.sqrt(mass_weights[i])))-(v.real[3*j][k]/(math.sqrt(mass_weights[j]))))+
                                        (traj4.xyz[0,i,1]-traj4.xyz[0,j,1])*((v.real[3*i+1][k]/(math.sqrt(mass_weights[i])))-(v.real[3*j+1][k]/(math.sqrt(mass_weights[j]))))+
                                        (traj4.xyz[0,i,2]-traj4.xyz[0,j,2])*((v.real[3*i+2][k]/(math.sqrt(mass_weights[i])))-(v.real[3*j+2][k]/(math.sqrt(mass_weights[j]))))
                                                                  )    
                                    )**2 





#              DELTA_i_j=DELTA_i_j+((R[i][0]-R[j][0])*(v.real[k][3*i]-v.real[k][3*j])*(1/math.sqrt(w.real[k]))*(1/dr_sq[i][j])+(R[i][1]-R[j][1])*(v.real[k][3*i+1]-v.real[k][3*j+1])*(1/math.sqrt(w.real[k]))*(1/dr_sq[i][j])+(R[i][2]-R[j][2])*(v.real[k][3*i+2]-v.real[k][3*j+2])*(1/math.sqrt(w.real[k]))*(1/dr_sq[i][j]))**2                                              
#         fluctuation_NMA[i][j]=kB*T*DELTA_i_j                    
        #fluctuation_NMA[j][i]=fluctuation_NMA[i][j]
                #sum_over_modes.append(DELTA_i_j)

        
             # DELTA_i_j=DELTA_i_j+   (   (1/math.sqrt(w.real[k]))* ((R[i][0]-R[j][0])*(v.real[k][3*i]-v.real[k][3*j])+(R[i][1]-R[j][1])*(v.real[k][3*i+1]-v.real[k][3*j+1])+(R[i][2]-R[j][2])*(v.real[k][3*i+2]-v.real[k][3*j+2])       )    )**2    
            
            #DELTA_i_j=DELTA_i_j+   (   (1/math.sqrt(w.real[k]))* ( (traj4.xyz[0,i,0]-traj4.xyz[0,j,0])*(v.real[k][3*i]-v.real[k][3*j])+(traj4.xyz[0,i,1]-traj4.xyz[0,j,1])*(v.real[k][3*i+1]-v.real[k][3*j+1])+(traj4.xyz[0,i,2]-traj4.xyz[0,j,2])*(v.real[k][3*i+2]-v.real[k][3*j+2])       )    )**2    
        
        
        fluctuation_NMA[ids]=kB*T*DELTA_i_j* ((1/bond_list[ids][3])**2)

    # del w
    # del v
    
    error=np.zeros((len(bond_list)))
    for ids,bond in enumerate (bond_list):
        i,j=int(bond[0])-1,int(bond[1])-1      

        error[ids]=((fluctuation_NMA[ids]-fluctuation_MD[ids])**2)   # minimize the sum of square of error 
    #     #sq_error[j][i]=sq_error[i][j]

    #sq_error=(np.subtract(fluctuation_NMA,fluctuation_MD)**2)



    collected = gc.collect()
 
    #print("Garbage collector: collected %d objects." % (collected))
    del fluctuation_NMA
    
    return v,w,error

def force_constants(N,T,ALPHA,beta,max_itr,bond_list,fluctuation_MD,run,tolerance,path,flag_identical_bonds,flag_automate_ALPHA_beta_values,iterative_method,system,traj4,box_size,mass_weights):


    #SUM_OF_SQUARE_ERROR_RI=[]
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
        print(bond_list)
        bond_list_incre=bond_list 
        #print(bond_list_incre)      
        # bond_list_incre=np.zeros((len(bond_list),4))
        # # appending lines take into account that no spring constant is zero and have some finite value
        # for ids,bond in enumerate (bond_list):
        #     bond_list_incre[ids][0]=bond_list[ids][0]
        #     bond_list_incre[ids][1]=bond_list[ids][1]
        #     bond_list_incre[ids][2]=bond_list[ids][2]
        #     bond_list_incre[ids][3]=bond_list[ids][3]

        print("NMA_calculation1")
        v_,w_,error_ksp=NMA(N,bond_list,fluctuation_MD,T,traj4,mass_weights)
        
          
        # K_new=np.zeros((len(bond_list)))
        square_of_error=np.zeros((len(bond_list)))
        print("shape of bond_list", bond_list.shape)

        print("The iteration number is",iteration)


        Knew  = np.zeros((len(bond_list), 1))
        Kold  = bond_list[:, 2].reshape(len(bond_list), 1) 
        #print(Kold)
        error_old = np.array(error_ksp).reshape(len(bond_list), 1)
        J     = np.zeros((len(bond_list), len(bond_list)))
        #print(error_old)
        print("shape of error_ksp", error_ksp.shape)
        print("shape of error_old", error_old.shape)    
        print("shape of Knew", Knew.shape)  
        print("shape of Kold", Kold.shape)  
        print("shape of Jacobian", J.shape) 


        # error_old = np.array(error_ksp).reshape(len(bond_list),1)    # n×1
        if iterative_method=='Jacobian_NR_method':



            for ids,bond in enumerate (bond_list):
                print(ids)
                for ids_1,bond in enumerate (bond_list):
                    bond_list_incre=bond_list                   
                    bond_list_incre[ids_1][2]=bond_list[ids_1][2]+ALPHA
                    v_incre,w_incre,error_ksp_incre=NMA(N,bond_list_incre,fluctuation_MD,T,traj4,mass_weights)



                    finite_diff_error=(error_ksp_incre[ids]-error_ksp[ids])/ALPHA

                    J[ids][ids_1]=finite_diff_error


            #print(J)
            print(np.sum(J, axis=0))  # check if any column is all zeros

            print("Rank of J:", np.linalg.matrix_rank(J))
            print("Shape of J:", J.shape)




            # solve J * delta = -error_old
            delta = np.linalg.solve(J, -(error_old))      # n×1




            #epsilon = 1e-3
            #delta = np.linalg.solve(J + epsilon*np.eye(len(J)), -error_old)


            #delta = np.linalg.pinv(J) @ ((-error_old))
            print("Shape of delta:", delta.shape)
            Knew = Kold + delta
            print("Shape of Knew:", Knew.shape)
            Kold=Knew
            print("Shape of Kold:", Kold.shape)
            #print(Knew)
        #print(Kold)
        for ids,bond in enumerate (bond_list):
            #print(Kold[ids])
            bond_list[ids][2]=Kold[ids,0]

               
            square_of_error[ids]=(error_ksp[ids])

        ###################### other convergence methods #############################################################################
        # del error_ksp
        # if iterative_method=='NR_method':
        #     del error_ksp_incre
        # step=0       
        # for ids,bond in enumerate (bond_list):
        #     if bond_list[ids][2]>0:
        #         step=step+ (abs(K_new[ids]-bond_list[ids][2])/bond_list[ids][2])
        # STEP.append(step)                 
 
#         step=0
#         for i in range(0,N):
#             for j in range(i+1,N):
#                 bond_type=[i+1,j+1]
#                 if bond_type in  bond_list and k_sp[i][j]>0 :
#                     step=step+ (abs(K_new[i][j]-k_sp[i][j])/k_sp[i][j])
#         STEP.append(step)
        #print("STEP",STEP)

#         residual=0
#         for i in range(0,N):
#             for j in range(i+1,N):
#                 bond_type=[i+1,j+1]
#                 if bond_type in  bond_list and  k_sp[i][j]>0 :
#                     residual=residual+ (math.sqrt(abs(error_ksp[i][j]))/fluctuation_MD[i][j])
#         RESIDUAL.append(residual)  
        #print("residual",RESIDUAL)               

        ##################################################################################################################
        
        # SUM OF SQUARE OF ERROR
        sum_of_sq_error=np.sum(square_of_error)
        print("sum of square of error=",sum_of_sq_error)
        
        SUM_OF_SQUARE_ERROR.append(sum_of_sq_error)
        count_itr=iteration
        itr.append(count_itr)

        ######################################## PLOTING SUM OF SQUARE OF ERROR ###################################################

        plt.plot(itr, SUM_OF_SQUARE_ERROR,linestyle='solid')

        
        # naming the x axis
        plt.xlabel('iteration')
        # naming the y axis
        plt.ylabel('sum of square of error')
        # giving a title to graph
        plt.title('CONVERGENCE PLOT')
        #saving the figure       
        #plt.savefig(path+'/POST_PROCESSING/SUM OF SQUARE OF ERROR PLOT %s.eps'%(run))


        #################################################### RESIDUAL ##############################################################
        ######################################## PLOTING RESIDUAL ###################################################

#         plt.plot(itr, RESIDUAL,linestyle='solid')
#         plt.yscale("log")
#         # naming the x axis
#         plt.xlabel('iteration')
#         # naming the y axis
#         plt.ylabel('RESIDUAL')
#         # giving a title to graph
#         plt.title('RESIDUAL CONVERGENCE PLOT')
#         #saving the figure       
#         #plt.savefig(path+'/POST_PROCESSING/RESIDUAL PLOT %s.eps'%(run))
                
#         #################################################### step #############################################################
#         ######################################## PLOTING step ###################################################

#         plt.plot(itr, STEP ,linestyle='solid')
#         plt.yscale("log")
#         # naming the x axis
#         plt.xlabel('iteration')
#         # naming the y axis
#         plt.ylabel('STEP')
#         # giving a title to graph
#         plt.title('SPRING CONSTANT CONVERGENCE PLOT')
#         #saving the figure       
#         #plt.savefig(path+'/POST_PROCESSING/SPRING CONSTANT CONVERGENCE %s.eps'%(run))
                        
        #######################################  SAVING TEXT FILES ###############################################################

        # file=open(path+'/POST_PROCESSING/sum_of_square_of_error%s.txt'%(run),"w")
        # np.savetxt(file,SUM_OF_SQUARE_ERROR)
        # file.close()

        # file=open(path+'/POST_PROCESSING/no_of_iteration%s.txt'%(run),"w")
        # np.savetxt(file,itr)
        # file.close()
        #del itr
#         file=open(path+'/POST_PROCESSING/residual%s.txt'%(run),"w")
#         np.savetxt(file,RESIDUAL)
#         file.close()

        # file=open(path+'/POST_PROCESSING/STEP%s.txt'%(run),"w")
        # np.savetxt(file,STEP)
        # file.close()
          
        ############################################################################################################################
        # for ids,bond in enumerate (bond_list):            
        #     bond_list[ids][2]=K_new[ids]  


        # if flag_identical_bonds=="yes":
        #     if system=='mspa':
        #         k_sp=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
        #         #dr_sq=func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds(dr_sq,path)
        #     if system=='fas':
        #         k_sp=func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(k_sp,path)
        #         #dr_sq=func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds(dr_sq,path)
            
        # if iteration % 10 == 0:  # Save every 100 iterations
        #     # np.savetxt(path+'/POST_PROCESSING/K_values_%d.csv'%(iteration),bond_list, delimiter = ",")    
        #     np.savetxt(path + f'/POST_PROCESSING/K_values_{iteration}.csv',bond_list.astype(float), delimiter=",")


        #     plt.plot(itr, SUM_OF_SQUARE_ERROR,linestyle='solid')
            
        #     # naming the x axis
        #     plt.xlabel('iteration')
        #     # naming the y axis
        #     plt.ylabel('sum of square of error')
        #     # giving a title to graph
        #     plt.title('CONVERGENCE PLOT')
            #saving the figure       
            #plt.savefig(path+'/POST_PROCESSING/SUM OF SQUARE OF ERROR PLOT %s.eps'%(run))


            # plt.plot(itr, STEP ,linestyle='solid')
            
            # # naming the x axis
            # plt.xlabel('iteration')
            # # naming the y axis
            # plt.ylabel('STEP')
            # # giving a title to graph
            # plt.title('SPRING CONSTANT CONVERGENCE PLOT')
            #saving the figure       
            #plt.savefig(path+'/POST_PROCESSING/SPRING CONSTANT CONVERGENCE %s.eps'%(run))


        #end = time.time()    
        #print("The time of execution of above program is :",(end-start) * (1/60), "min")

#         if step<tolerance:
#             print("step", step)
#             break

    # opening the csv file in 'w+' mode
     	#file = open(path+'/POST_PROCESSING/bonds_k_zeros.csv', 'w+', newline ='')
     
     # writing the data into the file
     	#with file:   
     		#write = csv.writer(file)       
        	#write.writerows(bonds_k_zeros)
        	
#         	file = open(path+'/POST_PROCESSING/bonds_k_non_zeros_%d.csv'%(iteration), 'w+', newline ='')
#         	with file:       
#         		write = csv.writer(file)
#         		write.writerows(bonds_k_non_zeros)


  
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
                   

        # print("NUMBER OF k's equal to zero", counting)

        # #saving non zero spring constant
        # bonds_k_non_zeros=[]
        # for ids,bond in enumerate (bond_list):            
        #     i,j=int(bond[0])-1,int(bond[1])-1 
        #     bond_type=[i+1,j+1]
        #     if bond_list[ids][2]>0:      
        #         bonds_k_non_zeros.append(bond_type)

        # if iteration % 10 == 0:  # Save every 100 iterations
           
        	
        #     file = open(path+'/POST_PROCESSING/bonds_k_non_zeros_%d.csv'%(iteration), 'w+', newline ='')
        #     with file:       
        #         write = csv.writer(file)
        #         write.writerows(bonds_k_non_zeros)


        
        if sum_of_sq_error<tolerance:   #Tolerance can be used after running multiple run for a system as it changes with ALPHA, beta values                                            choosen
            print("sum_of_sq_error", sum_of_sq_error)
            break

#24 NOV 2024            
        if abs(sum_of_sq_error-start_SOE)<tolerance :#and sum_of_sq_error<start_SOE:   #Tolerance can be used after running multiple run for a system                                                                                  as it changes with ALPHA, beta values choosen
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


