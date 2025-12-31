import math                           
import numpy as np                
import mdtraj as md


def average_for_identical_bonds_fluctuation(quantity,path):
    
    traj2=md.load(path+'/INPUT/REFERENCE.pdb') 
    N=traj2.n_atoms
    n=int(N/traj2.n_chains)  # no. of CG SITES PER CHAIN IN FAS
    #print("no. of CG SITES PER CHAIN IN FAS",n)
    i=0
    quantity_average=np.zeros(((N),(N)))
    #identical bonds in single chain

    for j in range(1,n+1):
        quantity_identical_bonds=[]
        for k in range(j+1,n+1):
            if j!=k:
                
                id1=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i))
                quantity_identical_bonds.append(quantity[id1[0]][id1[1]])
                
                id2=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+1))
                quantity_identical_bonds.append(quantity[id2[0]][id2[1]])

                id3=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+2))
                quantity_identical_bonds.append(quantity[id3[0]][id3[1]])

                id4=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+3))
                quantity_identical_bonds.append(quantity[id4[0]][id4[1]])

                id5=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+4))
                quantity_identical_bonds.append(quantity[id5[0]][id5[1]])

                id6=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+5))
                quantity_identical_bonds.append(quantity[id6[0]][id6[1]])

                id7=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+6))
                quantity_identical_bonds.append(quantity[id7[0]][id7[1]])

                id8=traj2.topology.select('residue %s %s and chainid==%s'%(j,k,i+7))
                quantity_identical_bonds.append(quantity[id8[0]][id8[1]])

                quantity_average[id1[0]][id1[1]]=np.average(quantity_identical_bonds)
                quantity_average[id2[0]][id2[1]]=np.average(quantity_identical_bonds)
                quantity_average[id3[0]][id3[1]]=np.average(quantity_identical_bonds)
                quantity_average[id4[0]][id4[1]]=np.average(quantity_identical_bonds)
                quantity_average[id5[0]][id5[1]]=np.average(quantity_identical_bonds)
                quantity_average[id6[0]][id6[1]]=np.average(quantity_identical_bonds)
                quantity_average[id7[0]][id7[1]]=np.average(quantity_identical_bonds)
                quantity_average[id8[0]][id8[1]]=np.average(quantity_identical_bonds)
            
    ##print(len(avg_quantity))   45X6

    #first adjacent    

    i=0
    
    for j in range(1,n+1):
        quantity_identical_bonds=[]

        for k in range(1,n+1):
                id1_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+1)))
                quantity_identical_bonds.append(quantity[id1_dome[0]][id1_dome[1]])

                id2_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+7)))
                quantity_identical_bonds.append(quantity[id2_dome[0]][id2_dome[1]])
                #print(id2_dome[0],id2_dome[1])

                id3_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+1,k,i+2)))
                quantity_identical_bonds.append(quantity[id3_dome[0]][id3_dome[1]])

                id4_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+2,k,i+3)))
                quantity_identical_bonds.append(quantity[id4_dome[0]][id4_dome[1]])

                id5_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+3,k,i+4))) 
                quantity_identical_bonds.append(quantity[id5_dome[0]][id5_dome[1]])

                id6_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+4,k,i+5)))
                quantity_identical_bonds.append(quantity[id6_dome[0]][id6_dome[1]])

                id7_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+5,k,i+6))) 
                quantity_identical_bonds.append(quantity[id7_dome[0]][id7_dome[1]])

                id8_dome=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+6,k,i+7)))
                quantity_identical_bonds.append(quantity[id8_dome[0]][id8_dome[1]])

                
                quantity_average[id1_dome[0]][id1_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id2_dome[0]][id2_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id3_dome[0]][id3_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id4_dome[0]][id4_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id5_dome[0]][id5_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id6_dome[0]][id6_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id7_dome[0]][id7_dome[1]]=np.average(quantity_identical_bonds)
                quantity_average[id8_dome[0]][id8_dome[1]]=np.average(quantity_identical_bonds)                
    #second adjacent  

                id1_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+2)))
                quantity_identical_bonds.append(quantity[id1_dome_OPP[0]][id1_dome_OPP[1]])
                
                id2_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+6)))
                quantity_identical_bonds.append(quantity[id2_dome_OPP[0]][id2_dome_OPP[1]])
                
                id3_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+1,k,i+7)))
                quantity_identical_bonds.append(quantity[id3_dome_OPP[0]][id3_dome_OPP[1]])
                
                id4_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+1,k,i+3)))
                quantity_identical_bonds.append(quantity[id4_dome_OPP[0]][id4_dome_OPP[1]])
                
                id5_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+2,k,i+4)))
                quantity_identical_bonds.append(quantity[id5_dome_OPP[0]][id5_dome_OPP[1]])
                
                id6_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+3,k,i+5)))
                quantity_identical_bonds.append(quantity[id6_dome_OPP[0]][id6_dome_OPP[1]])
                
                id7_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+4,k,i+6)))
                quantity_identical_bonds.append(quantity[id7_dome_OPP[0]][id7_dome_OPP[1]])
                
                id8_dome_OPP=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+5,k,i+7)))
                quantity_identical_bonds.append(quantity[id8_dome_OPP[0]][id8_dome_OPP[1]])
                
                quantity_average[id1_dome_OPP[0]][id1_dome_OPP[1]]=np.average(quantity_identical_bonds)
                quantity_average[id2_dome_OPP[0]][id2_dome_OPP[1]]=np.average(quantity_identical_bonds)
                quantity_average[id3_dome_OPP[0]][id3_dome_OPP[1]]=np.average(quantity_identical_bonds)
                quantity_average[id4_dome_OPP[0]][id4_dome_OPP[1]]=np.average(quantity_identical_bonds)
                quantity_average[id5_dome_OPP[0]][id5_dome_OPP[1]]=np.average(quantity_identical_bonds)
                quantity_average[id6_dome_OPP[0]][id6_dome_OPP[1]]=np.average(quantity_identical_bonds)  
                quantity_average[id7_dome_OPP[0]][id7_dome_OPP[1]]=np.average(quantity_identical_bonds)
                quantity_average[id8_dome_OPP[0]][id8_dome_OPP[1]]=np.average(quantity_identical_bonds)  
 #third adjacent                               
    i=0
    for j in range(1,n+1):
        quantity_identical_bonds=[]

        for k in range(1,n+1): 

                id1_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+3)))
                quantity_identical_bonds.append(quantity[id1_dome_same[0]][id1_dome_same[1]])
                
                id2_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+5)))
                quantity_identical_bonds.append(quantity[id2_dome_same[0]][id2_dome_same[1]])
                
                id3_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+1,k,i+4)))
                quantity_identical_bonds.append(quantity[id3_dome_same[0]][id3_dome_same[1]])
                
                id4_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+2,k,i+5)))
                quantity_identical_bonds.append(quantity[id4_dome_same[0]][id4_dome_same[1]])
                
                id5_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+3,k,i+6)))
                quantity_identical_bonds.append(quantity[id5_dome_same[0]][id5_dome_same[1]])
                
                id6_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+4,k,i+7)))
                quantity_identical_bonds.append(quantity[id6_dome_same[0]][id6_dome_same[1]])

                id7_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+1,k,i+6)))
                quantity_identical_bonds.append(quantity[id7_dome_same[0]][id7_dome_same[1]])
                
                id8_dome_same=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+2,k,i+7)))
                quantity_identical_bonds.append(quantity[id8_dome_same[0]][id8_dome_same[1]])
                
                
                quantity_average[id1_dome_same[0]][id1_dome_same[1]]=np.average(quantity_identical_bonds)
                quantity_average[id2_dome_same[0]][id2_dome_same[1]]=np.average(quantity_identical_bonds)
                quantity_average[id3_dome_same[0]][id3_dome_same[1]]=np.average(quantity_identical_bonds)
                quantity_average[id4_dome_same[0]][id4_dome_same[1]]=np.average(quantity_identical_bonds)
                quantity_average[id5_dome_same[0]][id5_dome_same[1]]=np.average(quantity_identical_bonds)
                quantity_average[id6_dome_same[0]][id6_dome_same[1]]=np.average(quantity_identical_bonds)   
                quantity_average[id7_dome_same[0]][id7_dome_same[1]]=np.average(quantity_identical_bonds)
                quantity_average[id8_dome_same[0]][id8_dome_same[1]]=np.average(quantity_identical_bonds)                                  
   
   
    i=0
    for j in range(1,n+1):
        quantity_identical_bonds=[]

        for k in range(1,n+1): 



                id1_dome_same4=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i,k,i+4)))
                quantity_identical_bonds.append(quantity[id1_dome_same4[0]][id1_dome_same4[1]])
                
                id2_dome_same4=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+1,k,i+5)))
                quantity_identical_bonds.append(quantity[id2_dome_same4[0]][id2_dome_same4[1]])
                
                id3_dome_same4=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+2,k,6)))
                quantity_identical_bonds.append(quantity[id3_dome_same4[0]][id3_dome_same4[1]])
                
                id4_dome_same4=(traj2.topology.select('residue %s and chainid==%s or residue %s and chainid==%s '%(j,i+3,k,i+7)))
                quantity_identical_bonds.append(quantity[id4_dome_same4[0]][id4_dome_same4[1]])

                quantity_average[id1_dome_same4[0]][id1_dome_same4[1]]=np.average(quantity_identical_bonds)
                quantity_average[id2_dome_same4[0]][id2_dome_same4[1]]=np.average(quantity_identical_bonds)
                quantity_average[id3_dome_same4[0]][id3_dome_same4[1]]=np.average(quantity_identical_bonds)
                quantity_average[id4_dome_same4[0]][id4_dome_same4[1]]=np.average(quantity_identical_bonds)
 
                                



    for m in range(0,len(quantity_average)):
        for n in range(0,len(quantity_average)):
            quantity_average[n][m]=quantity_average[m][n]
            
            
#    import xlsxwriter
#    ############################################  OUTPUT ########################################################################
#    # WRITING THE HESSIAN MATRIX IN AN EXCEL FILE
#     with xlsxwriter.Workbook('quantity.xlsx') as workbook:
#         worksheet = workbook.add_worksheet()
#         for row_num, data in enumerate(quantity_average):
#             worksheet.write_row(row_num, 0, data)
    ###############################################################################################################################

    return quantity_average
