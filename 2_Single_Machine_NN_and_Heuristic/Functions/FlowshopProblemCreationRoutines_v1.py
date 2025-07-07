# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:16:11 2022

@author: owner
"""
# Importation of necessary packages
from scipy import stats
import random
import numpy as np
from Functions.FlowshopProblemClassDefs import *
# from FlowshopProblemScheduling_v2 import *
# from MILPfn import MILPfn

# ### Function to generate instances

# In[6]:


def Create_inst(params):
    Ar=np.empty((params.n_jobs,params.k_sen))
    Ai=np.empty((params.n_jobs,params.k_sen))
    A1=np.empty((2*params.n_jobs,params.k_sen))
    A=np.empty((params.n_jobs,params.k_sen))
    D=np.empty((params.n_jobs,params.k_sen))
    Dint=np.empty((params.n_jobs,params.k_sen))
    X=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    Z=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    indVec1=np.empty((2*params.n_jobs,params.k_sen))
    indVec=np.empty((params.n_jobs,params.k_sen))
    
    #For each scenario
    for k in range(params.k_sen):
              #Job arrivals Times
              # Generate n_jobs of each type,
              # then select earliest n_jobs
        #Arrival time of regular jobs (generated using exponential dist.):
        Ar[0,k]=stats.expon.rvs(scale=params.mu_Ar)
        for n in range(1,params.n_jobs):
            Ar[n,k]=Ar[n-1,k]+stats.expon.rvs(scale=params.mu_Ar)
        #Arrival time of irregular jobs:
        Ai[0,k]=stats.expon.rvs(scale=params.mu_Ai)
        for n in range(1,params.n_jobs):
            Ai[n,k]=Ai[n-1,k]+stats.expon.rvs(scale=params.mu_Ai)
        A1[:,k]=np.concatenate((Ar[:,k],Ai[:,k]))
        indVec1[:,k] = np.concatenate((np.ones(params.n_jobs),np.zeros(params.n_jobs)))
        sortVec = np.argsort(A1[:,k])
        A1[:,k] = A1[sortVec,k]
        A[:,k] = A1[:params.n_jobs,k]
        indVec1[:,k] = indVec1[sortVec,k]
        indVec[:,k]= indVec1[:params.n_jobs,k]

        #Generate processing times for this instance
        for jt in range(20):
            # Allow 20 times for failure
            try:
                
                mu_X=stats.expon.rvs(scale=(np.ones((params.n_jobs,1))@params.Mu_mu_Xm))
                sigma_X=stats.expon.rvs(scale=(np.ones((params.n_jobs,1))@params.Mu_sigma_Xm))
                ln_sigma_X = np.sqrt(np.log((sigma_X/mu_X)**2 + 1))
                ln_mu_X = np.log(mu_X) - ln_sigma_X**2/2
        
                # Actual set of processing times
                # first two indices for jobs & machines
                X[:,:,k]=stats.lognorm.rvs(scale=np.exp(ln_mu_X),s=ln_sigma_X)
        
                #Information available time delay
                mu_Z=stats.expon.rvs(scale=(np.ones((params.n_jobs,1))@params.Mu_mu_Zm))
                sigma_Z=stats.expon.rvs(scale=(np.ones((params.n_jobs,1))@params.Mu_sigma_Zm))
                ln_sigma_Z = np.sqrt(np.log((sigma_Z/mu_Z)**2 + 1))
                ln_mu_Z = np.log(mu_Z) - ln_sigma_Z**2/2
                
                Z[:,:,k]=stats.lognorm.rvs(scale=np.exp(ln_mu_Z),s=ln_sigma_Z)
            except:
                continue
            else:
                break

        #Job due additional delay (for margin)
        Dint[:,k]=stats.expon.rvs(scale=(np.ones(params.n_jobs)*params.mu_Dint))
        for n in range(params.n_jobs):
            # Generate actual due time = arrive+tot processing+tot info delay+margin
            D[n,k]=A[n,k]+sum(X[n,:,k])+sum(Z[n,:,k])+Dint[n,k]
    # sigma_X and sigma_Z are when jobs' exact processing / info delays are not known to the scheduler.        
    return [Inst(A,D,X,Z),indVec,mu_X,sigma_X,mu_Z,sigma_Z]


# ### Function to generate inputs for instance
# 
# The values known to the scheduler are
# 
# * A and D-A for jobs already arrived (note D-A is due time interval for regular jobs)
# * means and standard deviations for A and due time interval that haven't already arrived
# * Mean and standard deviation of processing times and information delay times for jobs already arrived
# * Overall mean and standard deviation of processing times and information delay times for jobs not already arrived

# In[7]:


def Create_inpt(A,D,indVec,mu_X,sigma_X,mu_Z,sigma_Z,params):
    
    val=1
    iVec=np.empty((params.n_jobs,params.k_sen))
    muA=np.empty((params.n_jobs,params.m_stage, params.k_sen))
    stdA=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    muDint=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    stdDint=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    muX=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    sigmaX=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    muZ=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    sigmaZ=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    
    #For each scenario
    for k in range(params.k_sen):
        for n in range(params.n_jobs):
            if(indVec[n,k]==0):
                iVec[n,k]=val
                val=val+1
            else:
                iVec[n,k]=0
        muA[:,:,k]=((1-indVec[:,k])*params.mu_Ai*iVec[:,k]+indVec[:,k]*A[:,k]).reshape((-1,1)) \
                    @ np.ones((1,params.m_stage))
        stdA[:,:,k]=((1-indVec[:,k])*np.sqrt(params.mu_Ai)).reshape((-1,1)) \
                    @ np.ones((1,params.m_stage))

        muDintIrr=(params.Mu_mu_Xm).sum()+(params.Mu_mu_Zm).sum()+params.mu_Dint
        muDint[:,:,k]=((1-indVec[:,k])*muDintIrr*iVec[:,k]+indVec[:,k]*(D[:,k]-A[:,k])).reshape((-1,1)) \
                    @ np.ones((1,params.m_stage))
        stdDint[:,:,k]=((1-indVec[:,k])*np.sqrt(muDintIrr)).reshape((-1,1)) \
                    @ np.ones((1,params.m_stage))

        # @@@ Note Mu_mu_Xm, Mu_sigma_Xm,Mu_mu_Zm et Mu_sigma_mu are the same for all k.
        for m in range(params.m_stage):
            muX[:,m,k]=(1-indVec[:,k])*params.Mu_mu_Xm[0,m]+indVec[:,k]*mu_X[:,m]

        for m in range(params.m_stage):
            sigmaX[:,m,k]=(1-indVec[:,k])*(np.sqrt(params.Mu_mu_Xm[0,m]+(params.Mu_sigma_Xm[0,m])**2))+indVec[:,k]*sigma_X[:,m]

        for m in range(params.m_stage):
            muZ[:,m,k]=(1-indVec[:,k])*params.Mu_mu_Zm[0,m]+indVec[:,k]*mu_Z[:,m]
            
        for m in range(params.m_stage):
            sigmaZ[:,m,k]=(1-indVec[:,k])*(np.sqrt(params.Mu_mu_Zm[0,m]+(params.Mu_sigma_Zm[0,m])**2))+indVec[:,k]*sigma_Z[:,m]

    return Inpt(muA,stdA,muDint,stdDint,muX,sigmaX,muZ,sigmaZ)


# ### Function to generate Jobs order

# In[8]:


def Job_order(params):
    job=np.empty((params.n_jobs,params.m_stage,params.k_sen))
    
    #For each scenario
    for k in range(params.k_sen):
        job[:,:,k]=np.array([np.random.permutation(range(params.n_jobs)) for m in range(params.m_stage)]).T
    return job.astype(int)

# In[9]:

def generData5(params,nData,gpno):
    
    params.k_sen=1
    n_channel = 3
    n=params.n_jobs
    m=params.m_stage
    l=8*n
    Data=np.empty((nData,l,m))
    Labels=np.empty((nData,m,n))
    for k in range(nData):
        print("group number"+str(gpno)+", run number "+str(k))
        [I,indVec,mu_X,sigma_X,mu_Z,sigma_Z]=Create_inst(params)
        Inp=Create_inpt(I.A,I.D,indVec,mu_X,sigma_X,mu_Z,sigma_Z,params)
        
        Data[k,0:n,:]=Inp.muX[:,:,0]
        Data[k,n:2*n,:]=Inp.stdX[:,:,0]
        Data[k,2*n:3*n,:]=Inp.muZ[:,:,0]
        Data[k,3*n:4*n,:]=Inp.stdZ[:,:,0]
        Data[k,4*n:5*n,:]=Inp.muA[:,0].reshape((n,1))*np.ones((1,m))
        Data[k,5*n:6*n,:]=Inp.stdA[:,0].reshape((n,1))*np.ones((1,m))
        Data[k,6*n:7*n,:]=Inp.muDint[:,0].reshape((n,1))*np.ones((1,m))
        Data[k,7*n:,:]=Inp.stdDint[:,0].reshape((n,1))*np.ones((1,m))
        
        
        [Job,Y]=IG(I.A[:,0],I.D[:,0],I.X[:,:,0],I.Z[:,:,0],0.4,5)
        J=np.empty(Job.shape)
        for i in range(n):
            for j in range(m):
                for e in range(n):
                    if Job[e,j]==i:
                        J[i,j]=(e+1)/max(Job[:,j])
            
        for j in range(m):
            Labels[k,j,:]  = J[:,j]
    
    return Data,Labels

