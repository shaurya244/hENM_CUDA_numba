import numpy as xp



import func_mspa_identical_bonds_for_any_quantity
import func_fas_identical_bonds_for_any_quantity

def lammps_data_file(N, bond_list, mass_weights, traj_cord, box_size, path, system, flag_identical_bonds):
    atomtypes_mspa = xp.loadtxt(path + '/INPUT/atomtypes_MSPA.txt', dtype=float)
    mass_CG_sites = xp.loadtxt(path + '/INPUT/mass_of_CG_sites_info.txt', dtype=float)

    # Use CuPy if available
    K = xp.zeros((len(bond_list)))
    for ids, bond in enumerate(bond_list):
        K[ids] = bond[2] / (4.184 * 2)

    # Apply identical bond averaging if requested
    if flag_identical_bonds == 'yes':
        print("flag_identical_bonds=yes")
        if system == 'mspa':
            K = func_mspa_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(K, path)
        elif system == 'fas':
            K = func_fas_identical_bonds_for_any_quantity.average_for_identical_bonds_fluctuation(K, path)

    # Move to NumPy for saving
    K = xp.asnumpy(K) if hasattr(K, 'get') else K

    with open(path + '/POST_PROCESSING/LAMMPS_DATA_FILE.data', 'w') as f:
        non_zero_bonds = sum(K[ids] != 0 for ids in range(len(bond_list)))

        f.write("LAMMPS Description\n")
        f.write(f"{N} atoms\n")
        f.write(f"{int(max(atomtypes_mspa) if system == 'mspa' else N)} atom types\n")
        f.write(f"{non_zero_bonds} bonds\n")
        f.write(f"{non_zero_bonds} bond types\n\n")

        f.write(f"0 {box_size} xlo xhi\n")
        f.write(f"0 {box_size} ylo yhi\n")
        f.write(f"0 {box_size} zlo zhi\n\n")

        f.write("Masses\n\n")
        for m in range(len(mass_CG_sites)):
            f.write(f"{m + 1} {mass_CG_sites[m]}\n")

        f.write("\nAtoms#full\n\n")
        for i in range(N):
            x, y, z = traj_cord.xyz[0, i, :]
            atom_type = int(atomtypes_mspa[i]) if system == 'mspa' else i + 1
            f.write(f"{i + 1} 0 {atom_type} 0.000000 {x} {y} {z}\n")

        f.write("\nBonds\n\n")
        bond_id = 1
        for ids, bond in enumerate(bond_list):
            if K[ids] != 0:
                i, j = int(bond[0]), int(bond[1])
                f.write(f"{bond_id} {bond_id} {i} {j}\n")
                bond_id += 1

        f.write("\nBond Coeffs\n\n")
        bond_id = 1
        for ids, bond in enumerate(bond_list):
            if K[ids] != 0:
                f.write(f"{bond_id} {K[ids]} {bond[3]}\n")
                bond_id += 1
