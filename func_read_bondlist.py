import csv
import numpy as xp

def gen_bondlist(N,name,path):

    c1 = [] #column 1 - bond_i
    c2 = [] #column 2 - bond_j

    with open(path+'/INPUT/%s'%name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            c1.append(row[0])
            c2.append(row[1])

    bond_i = []
    for word in c1:
        if word.isdigit():
            bond_i.append(int(word))

    bond_j = []
    for word in c2:
        if word.isdigit():
            bond_j.append(int(word))


    bond_list = xp.zeros((len(c1), 4), dtype=object)
    for i in range(0,len(c1)):
        bond_list[i][0]=int(bond_i[i])
        bond_list[i][1]=int(bond_j[i])

    no_of_bonds=len(bond_list)
    
    return bond_list,no_of_bonds
