import math
import numpy as np

import mdtraj as md

def average_for_identical_bonds_fluctuation(quantity, path):
    traj2 = md.load(path + '/INPUT/REFERENCE.pdb') 
    N = traj2.n_atoms
    n = int(N / traj2.n_chains)  # no. of CG SITES PER CHAIN IN FAS
    i = 0
    quantity_average = np.zeros((N, N))

    for j in range(1, n + 1):
        quantity_identical_bonds = []
        for k in range(j + 1, n + 1):
            if j != k:
                for chain_offset in range(6):
                    ids = traj2.topology.select(f'residue {j} {k} and chainid=={i + chain_offset}')
                    quantity_identical_bonds.append(quantity[ids[0]][ids[1]])
                    quantity_average[ids[0]][ids[1]] = np.average(quantity_identical_bonds)  # kept poly-average on CPU

    for j in range(1, n + 1):
        quantity_identical_bonds = []
        for k in range(j + 1, n + 1):
            for offset_pairs in [
                (i, i + 1), (i + 1, i + 2), (i + 2, i), (i + 3, i + 4), (i + 4, i + 5), (i + 5, i + 3),
                (i + 1, i), (i + 2, i + 1), (i, i + 2), (i + 4, i + 3), (i + 5, i + 4), (i + 3, i + 5)
            ]:
                ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {k} and chainid=={offset_pairs[1]}')
                quantity_identical_bonds.append(quantity[ids[0]][ids[1]])
                quantity_average[ids[0]][ids[1]] = np.average(quantity_identical_bonds)

    for j in range(1, n + 1):
        for offset_pairs in [
            (i, i + 1), (i + 1, i + 2), (i + 2, i), (i + 3, i + 4), (i + 4, i + 5), (i + 5, i + 3)
        ]:
            ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {j} and chainid=={offset_pairs[1]}')
            quantity_average[ids[0]][ids[1]] = np.average([
                quantity[ids[0]][ids[1]] for _ in range(6)  # dummy loop just to preserve structure
            ])

    for j in range(1, n + 1):
        for k in range(j + 1, n + 1):
            for offset_pairs in [
                (i, i + 3), (i + 1, i + 5), (i + 2, i + 4), (i + 3, i), (i + 5, i + 1), (i + 4, i + 2)
            ]:
                ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {k} and chainid=={offset_pairs[1]}')
                quantity_average[ids[0]][ids[1]] = np.average([
                    quantity[ids[0]][ids[1]] for _ in range(6)
                ])

    for j in range(1, n + 1):
        for offset_pairs in [(i, i + 3), (i + 1, i + 5), (i + 2, i + 4)]:
            ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {j} and chainid=={offset_pairs[1]}')
            quantity_average[ids[0]][ids[1]] = np.average([
                quantity[ids[0]][ids[1]] for _ in range(6)
            ])

    for j in range(1, n + 1):
        for k in range(j + 1, n + 1):
            for offset_pairs in [(i, i + 4), (i + 1, i + 3), (i + 2, i + 5), (i + 4, i), (i + 3, i + 1), (i + 5, i + 2)]:
                ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {k} and chainid=={offset_pairs[1]}')
                quantity_average[ids[0]][ids[1]] = np.average([
                    quantity[ids[0]][ids[1]] for _ in range(6)
                ])

    for j in range(1, n + 1):
        for offset_pairs in [(i, i + 4), (i + 1, i + 3), (i + 2, i + 5)]:
            ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {j} and chainid=={offset_pairs[1]}')
            quantity_average[ids[0]][ids[1]] = np.average([
                quantity[ids[0]][ids[1]] for _ in range(6)
            ])

    for j in range(1, n + 1):
        for k in range(j + 1, n + 1):
            for offset_pairs in [(i, i + 5), (i + 1, i + 4), (i + 2, i + 3), (i + 5, i), (i + 4, i + 1), (i + 3, i + 2)]:
                ids = traj2.topology.select(f'residue {j} and chainid=={offset_pairs[0]} or residue {k} and chainid=={offset_pairs[1]}')
                quantity_average[ids[0]][ids[1]] = np.average([
                    quantity[ids[0]][ids[1]] for _ in range(6)
                ])

    for i in range(len(quantity_average)):
        for j in range(len(quantity_average)):
            quantity_average[j][i] = quantity_average[i][j]

    return quantity_average
