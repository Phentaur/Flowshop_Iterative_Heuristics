# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:17:47 2022

@author: owner
"""

# Importation of necessary packages
import os.path
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from Functions.FlowshopProblemClassDefs import *
from numpy.random import choice,randint
from Functions.FlowshopProblemCreationRoutines_v1 import *
# from cvxopt import matrix
# from cvxopt.glpk import ilp
import pickle

############ Gantt Chart ######################
#A arrival times,Z=waiting times,S=starting times,Y = finishing times, D = due times
def plotSched(A,Z,S,Y,D):
    # Number of jobs, machines
    N,M=Z.shape
    W = np.zeros(Z.shape)
    ig,ax=plt.subplots()
    # Define a specific color for each job
    color = plt.cm.rainbow(np.linspace(0, 1, N))    
    
    # Plot initial arrival and waiting times
    for jn in range(N):
        yVal = M+1 - 2*(jn+1)/(3*N)-1/(3*N)
        W[jn,0] = A[jn]+Z[jn,0]
        esp=(int)(W[jn,0]-A[jn])+1
        ax.scatter(np.linspace(A[jn],W[jn,0],esp),\
         np.linspace(yVal,yVal,esp),marker='.',color=color[jn],\
             linewidth=0.1)
    # Plot lines for machines, machine by machine.
    ind=0
    labl=[]
    for jm in range(M):
        for jn in range(N):
            # Job interval on machine
            if (ind<N):
                ax.plot([S[jn,jm],Y[jn,jm]],[M-jm,M-jm],color=color[jn],linewidth=5,\
                         label='job'+str(ind))
                ind+=1
            else:
                ax.plot([S[jn,jm],Y[jn,jm]],[M-jm,M-jm],color=color[jn],linewidth=5)
            # If start time = prev. availability time, then draw dotted line
            if S[jn,jm] == W[jn,jm]:
                yVal = M-jm+1  - 2*(jn+1)/(3*N)-1/(3*N)  # heights are staggered so different weighting times are visible.
                ax.plot([S[jn,jm],S[jn,jm]],[yVal,M-jm],'c:')

            # Plot waiting time
            if jm<M-1:
                yVal = M-jm  - 2*(jn+1)/(3*N)-1/(3*N)   # heights are staggered so different weighting times are visible.
                ax.plot([Y[jn,jm],Y[jn,jm]],[M-jm,yVal],'k:')
                W[jn,jm+1] = Y[jn,jm]+Z[jn,jm+1]
                esp=(int)(W[jn,jm+1]-Y[jn,jm])
                ax.plot(np.linspace(Y[jn,jm],W[jn,jm+1],esp),\
                      np.linspace(yVal,yVal,esp),color=color[jn],solid_capstyle = 'round',\
                          linewidth=2)
        labl.append(M-jm)
    for jn in range(N):
        ax.scatter(D[jn],1/(3*N),marker='x',color=color[jn])
    plt.xlabel('time')
    ax.set_yticks(labl)
    ax.set_yticklabels(range(len(labl)))
    plt.ylabel('machine')
    plt.title('gantt chart')
    plt.legend()
    plt.show()
    
# ### Function to generate start and finish times at each stage

                
#Evaluate schedule objective functions    
# Given job,A,X,Z find corresponding start, finish, and idle times
def CalcYS(m0,m1,job,Am0,X,Z):
    N,M=X.shape
    m1 = min(m1,M)
    #Starting time(S) and finishing time(Y)
    S=np.empty((N,M))
    Y=np.empty((N,M))
    Idle =np.empty((N,M))

    
    # First Stage:
    S[job[0,m0],m0]=Am0[job[0,m0]]+Z[job[0,m0],m0]
    Y[job[0,m0],m0]=S[job[0,m0],m0]+X[job[0,m0],m0]
    Idle[job[0,m0],m0] = 0

    for jn in range(1,N):
        S[job[jn,m0],m0]=max(Am0[job[jn,m0]]+Z[job[jn,m0],m0],Y[job[jn-1,m0],m0])
        Y[job[jn,m0],m0]=S[job[jn,m0],m0]+X[job[jn,m0],m0]
        Idle[job[jn,m0],m0] = S[job[jn,m0],m0]-Y[job[jn-1,m0],m0]


    for jm in range(m0+1,m1):
        S[job[0,jm],jm]=Y[job[0,jm],jm-1]+Z[job[0,jm],jm]
        Y[job[0,jm],jm]=S[job[0,jm],jm]+X[job[0,jm],jm]
        Idle[job[0,jm],0] = Z[job[0,jm],jm]

        for jn in range(1,N):
            S[job[jn,jm],jm]=max(Y[job[jn-1,jm],jm],Y[job[jn,jm],jm-1]+Z[job[jn,jm],jm])
            Y[job[jn,jm],jm]=S[job[jn,jm],jm]+X[job[jn,jm],jm]
            Idle[job[jn,jm],jm] = S[job[jn,jm],jm]-Y[job[jn-1,jm],jm]
        
    return Y,S,Idle

# In[13]:


def Obj_pq(D,Y,p,q):
    #Total completion time:
    L=Y[:,-1]-D
    #number of late jobs:
    Lp=sum((L>0)*p)
    #Total late time
    L = L*q
    Lq=sum(L[L>0])
    
    return Lp+Lq #w*max(Y[:,-1])+(1-w)*Lk #w*Lj+(1-w)*(Lj>0) + 0.001*NLj

#%% Create and save K_sen_temp random scenarios

def Save_Sen(K_sen_save, filename, params):
    # K_sen_save is the number of scenarios to be saved and filename is the name of the file to save them to.

    data = []
    for q in range(K_sen_save):
        # Creates one scenario
        I=Create_inst(params)
        data.append(I)
    
    # Confirm if you want to save over the file
    if os.path.isfile('./' + filename + '.pkl'):
        ans = input("Do you wish to save a new scenario set over this file? (Y/N)")
        if ans == "Y":
            # Open the file to input scenarios
            file = open(filename + ".pkl", "wb")
            
            # Dump the scenarios
            pickle.dump(data, file)

            # Close the file
            file.close()
    
        else:
            return
    
#%% Code to save results
def Save_Results(Res_filename, results):
   
    # Confirm if you want to save over results
    if os.path.isfile('./' + Res_filename + '.pkl'):
        ans = input("Do you wish to save results over this file? (Y/N)")
        if ans == "Y":
            # Open the file to input results
            file = open(Res_filename + ".pkl", "wb")

            # Dump the scenarios
            pickle.dump(results, file)

            # Close the file
            file.close()
    
        else:
            return
