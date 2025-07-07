# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:31:39 2023

This code uses the exact MILP algorithm to generate and store single-machine instances which can then be used to test other single-machine algorithms.  The pickle files in the "Data" folder were created using this program.

The name of the file that is created is specified in the variable 'pickleFilename'

@author: matth
"""
from Functions.FlowshopProblemClassDefs import *
from Functions.FlowshopProblemCreationRoutines_v1 import *
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pickle
unif = np.random.uniform
geom = np.random.geometric 


##### Pickle  file to write to #### 
pickleFilename = 'single_machine_100N_100int.pickle'
writeToPickle=True
#Really big number for lateness indicator "on/off"
cBig=10000
# Number of instances generated
nInstances=100

pctDev = 0.25 # Percent deviation in parameters.
geomProb = 0.6 # Percent of time there is backlog of more than 1 job

N_jobs,M_stage,K_sen,mu_Ar,mu_Ai,mu_Dint=100,1,1,5,1E10,10
N = N_jobs# legacy variable
    #Flat rate:
p=50
    #Rate penalty:
q=1
   
# Parameters for randomly generating a scheduling scenario
Mu_mu_Xm=np.ones((1,M_stage))*5
Mu_sigma_Xm=np.ones((1,M_stage))*0.0001 # Remove uncertainty
Mu_mu_Zm=np.ones((1,M_stage))*5# 
Mu_sigma_Zm=np.ones((1,M_stage))*0.0001 # Remove uncertainty
uncertainFlag=False





# Set up classes for parameters
params=Params(M_stage,N_jobs,K_sen,mu_Ar,mu_Ai,mu_Dint,Mu_mu_Xm,Mu_sigma_Xm,Mu_mu_Zm,Mu_sigma_Zm)
# Temporary params
paramsTmp=Params(M_stage,N_jobs,K_sen,mu_Ar,mu_Ai,mu_Dint,Mu_mu_Xm,Mu_sigma_Xm,Mu_mu_Zm,Mu_sigma_Zm)


# Save to existing pickle file
# Note only works for 1 scenario
with open(pickleFilename, 'wb') as f:
    for i in range(nInstances): 
        if i>-1:
            print(i)
        #print("Instance no. ",i+1)
        paramsTmp.Mu_mu_Xm = params.Mu_mu_Xm*(1 + unif(-pctDev,pctDev))
        paramsTmp.Mu_mu_Zm = params.Mu_mu_Zm*(1 + unif(-pctDev,pctDev))
        paramsTmp.mu_Ar = params.mu_Ar*(1 + unif(-pctDev,pctDev))
        paramsTmp.mu_Dint = params.mu_Dint*(1 + unif(-pctDev,pctDev))               
        
        [Vec,indVec,mu_X,sigma_X,mu_Z,sigma_Z]=Create_inst(paramsTmp)    
        
        # Push back arrival so that more than one is ready at time 0.
        Vec.A[:,0] = Vec.A[:,0] - Vec.A[min(geom(geomProb),N_jobs)-1]
 
        A = Vec.A[:,0]; D=Vec.D[:,0]; X=Vec.X[:,:,0]; 
        if (A[0]<0):
            A=A-A[0]
            D=D-A[0]
        #res=flow_1(A,D,X,p,q)
        
        # Job Orders (Integers, Sorted)
        #matchMx = res['x'][-N*N-2*N:-2*N].reshape(N,N)
        #matchMx[matchMx > .9] = 1
        #matchMx[matchMx <.9] = 0
        orders= np.arange(N_jobs)
        # print(orders)
    
        instance={'A':A,'X':X,'D':D,'order':orders, 'p':p, 'q':q}
   
        if writeToPickle:
            pickle.dump(instance, f)