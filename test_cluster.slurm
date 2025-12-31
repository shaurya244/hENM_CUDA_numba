#!/bin/bash
#
#SBATCH --job-name=henm.py
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH -p test


/home/manishaphd20/anaconda3/envs/HENM_env/bin/python MAPPING_AND_NETWORKING/mapping_and_networking.py "/home/manishaphd20"
/home/manishaphd20/anaconda3/envs/HENM_env/bin/python CREATE_BOND_LIST/bondlist.py "/home/manishaphd20" 
/home/manishaphd20/anaconda3/envs/HENM_env/bin/python code_HENM.py "/home/manishaphd20"
/home/manishaphd20/anaconda3/envs/HENM_env/bin/python restart_code_HENM.py "." run(integer)
/home/ambikagr22/software_main/lammps-2Aug2023/build/lmp -in POST_PROCESSING/system.in 
/home/manishaphd20/anaconda3/envs/HENM_env/bin/python post_henm.py "/home/manishaphd20"
