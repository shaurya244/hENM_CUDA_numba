import random

def lammps_input_file_gen(T,damp,run_value,dump_value,path,count_run_value):

    sourceFile = open(path+'/POST_PROCESSING/lammps_input.in', 'w')
    print("units real",file = sourceFile) 
    print("dimension 3",file = sourceFile)
    print("boundary p p p",file = sourceFile)
    print("atom_style full\n",file = sourceFile)
    print("pair_style none",file = sourceFile)
    print("bond_style harmonic\n",file = sourceFile)
    print("read_data" ' "'+path+"/POST_PROCESSING/LAMMPS_DATA_FILE.data"+'"'+"\n",file = sourceFile)
    print("timestep 1\n",file = sourceFile)
    print("velocity all create",T,random.randint(10000,50000),file = sourceFile)
    print("fix 1 all nve",file = sourceFile)
    print("fix 2 all langevin",T,T,damp,random.randint(50001,99999),file = sourceFile)
    print("dump 1 all custom",dump_value,' "'+path+'/POST_PROCESSING/hENM_CG_trajectory.lammpstrj" '+"id type xu yu zu\n",file = sourceFile)

    for r in range(0,count_run_value):
        print("run",run_value,file = sourceFile)

    sourceFile.close()
    
    
