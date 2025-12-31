import matplotlib.pyplot as plt
import numpy as xp

def rmsf_plots(rmsf_MD,rmsf_CG,N,path):

    # res = np.arange(1,N+1, 1)
    # plt.figure(figsize=(15, 10))
    # # plotting the points
    # plt.plot(res,rmsf_MD)
    # plt.plot(res,rmsf_value_cg)

    # # naming the x axis
    # plt.xlabel('CG SITES')
    # # naming the y axis
    # plt.ylabel('RMSF ')
    # plt.axvspan(1,11,facecolor='green', alpha=0.3)
    # plt.axvspan(11,21,facecolor='red', alpha=0.3)
    # plt.axvspan(21,31,facecolor='grey', alpha=0.3)
    # plt.axvspan(31,41,facecolor='blue', alpha=0.3)
    # plt.axvspan(41,51,facecolor='yellow', alpha=0.3)
    # plt.axvspan(51,60,facecolor='pink', alpha=0.3)
    # plt.title('FLUCTUATIONS')
    # plt.savefig('FLUCTUATION.eps')
    
    plt.figure(figsize=(10, 5))
    res = xp.arange(1,N+1, 1)
    # plotting the points
    plt.rcParams.update({'font.size':15})  
    plt.plot(res,rmsf_MD)
    plt.plot(res,rmsf_CG)
    # naming the x axis
    plt.xlabel('CG SITES')
    # naming the y axis
    plt.ylabel('FLUCTUATION')
    # giving a title to my graph
    plt.title('RMS FLUCTUATION')
    plt.savefig(path+'/OUTPUT/RMS_FLUCTUATION.eps')
    plt.legend(["CG-MD mapped trajectory","HENM-CG Simulation"])
    # function to show the plot
    #plt.show()


    ratio_rmsf=[]
    for i in range(len(rmsf_MD)):
        ratio_rmsf.append(rmsf_MD[i]/rmsf_CG[i])
    plt.rcParams.update({'font.size': 40})  
    plt.figure(figsize=(60, 20))
    res = xp.arange(1,len(rmsf_MD)+1, 1)
    # plotting the points 
    plt.plot(res,ratio_rmsf,'.',markersize=40)
    # naming the x axis
    plt.xlabel('CG SITES')
    # naming the y axis
    plt.ylabel('RMSF RATIO (MD/CG)')
    # giving a title to my graph
    plt.title('RMSF RATIO')
    plt.savefig(path+'/POST_PROCESSING/RMSF_RATIO.eps')
    # function to show the plot
    #plt.show()   


    percent_ratio_rmsf=[]
    for i in range(len(rmsf_MD)):
        percent_ratio_rmsf.append(abs(rmsf_MD[i]-rmsf_CG[i]/rmsf_MD[i])*100)
    plt.rcParams.update({'font.size': 40})  
    plt.figure(figsize=(60, 20))
    res = xp.arange(1,len(rmsf_MD)+1, 1)
    # plotting the points 
    plt.plot(res,percent_ratio_rmsf,'.',markersize=40)
    # naming the x axis
    plt.xlabel('CG SITES')
    # naming the y axis
    plt.ylabel('% error ABS((MD-CG/MD)*100)')
    # giving a title to my graph
    plt.title('RMSF %ERROR')
    plt.savefig(path+'/POST_PROCESSING/RMSF_%_ERROR.eps')
    # function to show the plot
    #plt.show()  


    #RMSF ERROR
    error_rmsf=[]
    for i in range (0,N): 
        error_rmsf.append((abs(rmsf_MD[i]-rmsf_CG[i])))
    res =xp.arange(1,N+1, 1)
    plt.rcParams.update({'font.size': 50})
    plt.figure(figsize=(80, 30))
    plt.plot(res,error_rmsf,'.',markersize=50)
    # naming the x axis
    plt.xlabel('BOND ij')
    # naming the y axis
    plt.ylabel('error ij')
    # giving a title to my graph
    plt.title('error plot for rmsf')
    plt.savefig(path+'/POST_PROCESSING/RMSF_ERROR.eps')


    ##################################### RMSF MD in (Ang) ###########################################################################
    sourceFile = open(path+'/POST_PROCESSING/RMSF_MD.txt', 'w')
    for i in range(len(rmsf_MD)):
        print(rmsf_MD[i],file=sourceFile)
    sourceFile.close()

    ##################################### RMSF CG in (Ang) ###########################################################################

    sourceFile = open(path+'/POST_PROCESSING/RMSF_CG.txt', 'w')
    for i in range(len(rmsf_CG)):
        print(rmsf_CG[i],file=sourceFile)
    sourceFile.close()

    #################################################################################
