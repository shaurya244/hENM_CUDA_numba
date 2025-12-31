import pandas as pd
import numpy as xp
import csv

def read_spring_constant(path,bond_list):
	df = pd.read_csv(path+'/POST_PROCESSING/K_values_final.csv', header=None)

	k_sp = xp.array(df)
	for ids,bond in enumerate (bond_list):
		i,j=int(bond[0])-1,int(bond[1])-1   
		bond_list[ids][2]=k_sp[i][j]

	return bond_list
