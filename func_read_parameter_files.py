
def value_of_parameters(path):

    # Define a dictionary to store parameter values
    parameters = {}

    # Read the text file
    with open(path+'/INPUT/parameter.txt', 'r') as file:
        for line in file:
            # Split each line by whitespace
            parts = line.split()
            if len(parts) == 2:
                parameter_name = parts[0]
                parameter_value = parts[1].strip("'")

                # Store parameter values in the dictionary
                if parameter_name == 'bondlist_name':
                    # Handling bondlist_name separately due to potential whitespace in the file name
                    parameters[parameter_name] = parameter_value
                elif parameter_name == 'flag_identical_bonds':
                    parameters[parameter_name] = parameter_value
                elif parameter_name == 'iterative_method':
                    parameters[parameter_name] = parameter_value
                elif parameter_name == 'flag_automate_ALPHA_beta_values':
                    parameters[parameter_name] = parameter_value
                elif parameter_name == 'ALPHA' :
                    parameters[parameter_name] = float(parameter_value)
                elif parameter_name == 'beta' :
                    parameters[parameter_name] = float(parameter_value)
                elif parameter_name == 'tolerance' :
                    parameters[parameter_name] = float(parameter_value)
                elif parameter_name == "lammps_box_size" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "max_itr" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "T" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "lammps_damp" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "lammps_dump_value" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "lammps_run_value" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "count_run_value" :
                    parameters[parameter_name] = int(float(parameter_value))
                elif parameter_name == "stride" :
                    parameters[parameter_name] = int(float(parameter_value)) 
                elif parameter_name == "initial_guess" :
                    parameters[parameter_name] =  parameter_value
                elif parameter_name == "system" :
                    parameters[parameter_name] =  parameter_value
                elif parameter_name == "method" :
                    parameters[parameter_name] =  parameter_value                    
    #print(parameters)
    return parameters
