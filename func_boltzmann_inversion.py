import numpy as xp

from numpy.polynomial import polynomial as P  # NumPy polyfit still used; CuPy has no polynomial support

def gaussian(x, mu, sig):
    return 1. / (xp.sqrt(2 * xp.pi) * sig) * xp.exp(-xp.power((x - mu) / sig, 2) / 2)

def boltz_inver_initial_guess(bo_md, SIG_md, bond_list, T, path):
    for ids, bond in enumerate(bond_list):
        x_values = xp.arange(xp.min(bo_md), xp.max(bo_md), 1)
        prob = gaussian(x_values, bo_md[ids], SIG_md[ids])

        # Filter for positive probabilities
        mask = prob > 0
        probe = prob[mask]
        x = x_values[mask]

        kB = 8.314462618 * 0.001
        y = -kB * T * xp.log(probe)

        # Convert to NumPy for polyfit


        c, stats = P.polyfit(x, y, 2, full=True)
        bond_list[ids][2] = c[2]

    return bond_list
