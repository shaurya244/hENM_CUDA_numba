Input required:

(1) COARSE_GRAINED_MAPPED_TRAJECTORY.xtc        TRAJECTORY
(2) REFERENCE.pdb                               (This file can be Energy_minimized_coordinates.pdb, average structure.pdb or initial_frame_of_production_run.pdb)
(3) bondlist.csv                                BONDLIST CAN BE GENERATED OR CAN BE USER_DEFINED
(4) mass_of_CG_sites_info.txt                   INFORMATION OF OF MASS OF CG SITES
(5) parameter.txt                               PARAMETERS FOR code_HENM.py
(6) mass_weights                                mass of each CG site 
(7) energy_min.in                               energy_minimization file to energy minimize the system at each iteration (provide path/POSTPROCESSING..)





Arguments in parameter.txt

ALPHA
beta                         
max_itr                                         ITERATION
tolerance                                       TOLERANCE VALUE FOR CONVERGENCE
T                                               TEMPERATURE
lammps_damp                                     DAMP VALUE FOR LAMMPS INPUT FILE
lammps_dump_value                               FREQUENCY OF DUMPING COORDINATES FOR LAMMPS INPUT FILE 
lammps_run_value                                RUN FOR LAMMPS INPUT FILE
lammps_box_size                                 DIMENSION OF BOX SIZE FOR LAMMPS INPUT FILE
bondlist_name                                   NAME OF THE BONDLIST
flag_identical_bonds                            AVERAGE SPRING VALUES FOR IDENTICAL BONDS: 'yes'/'no'
flag_automate_ALPHA_beta_values                 AUTOMATE ALPHA AND BETA VALUES: 'yes'/'no'
iterative_method                                METHOD: 'Lyman_method','Chu_and_Voth_method'/'NR_method'
stride                                          DELETE SOME FRAME IN A LARGE TRAJECTORY OF HENMCG SIMULATION
count_run_value                                 FOR "run" COMMAND IN LAMMPS INPUT 
initial_guess                                   INITIAL GUESS FOR HESSIAN 'uniform'/'boltzmann_inversion'

COMMANDS:

STEP1:
path_of_working_directory="pwd"

file_format_of_input_trajecory="xtc" or "lammpstrj"

python code_HENM.py "path_of_working_directory" "file_format_of_input_trajecory"

STEP2:
argument=pwd(path of the working directory) give as a string
python post_HENM.py "path of the working directory" run "input_trajectory_format (xtc/lammpstrj)"

RESTART 

python restart_code_HENM.py "path of the working directory" run "input_trajectory_format (xtc/lammpstrj)"
