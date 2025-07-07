

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:17:47 2022

@author: owner
"""

# Importation of necessary packages
import numpy as np
from cvxopt import matrix
from cvxopt.glpk import ilp,options
import pickle

def  MILPsched1M(availTimes,dueTimes,X,pVec,qVec,rVec):
    
## Optimized schedule for single machine
# Constants:  
# a_1,…a_N (arrival time, including information delay ); 
# d_1,…d_N  (due times), 
# x_1,…x_N (processing times)
# p_1,…p_N (penalty (cost)  if job is late);  
# q_1,…q_N (lateness penalty coefficient).

    # Remove messaging from cvxopt.
    options['msg_lev'] = 'GLP_MSG_OFF'
    aVec = availTimes # arrival times in vector
    xVec = X # processing times in vector
    dVec = dueTimes # due times in vector
    if type(pVec)!="numpy.ndarray":
        pVec=pVec+0*aVec
        qVec=qVec+0*aVec
        rVec=rVec+0*aVec
    Cbig = np.sum(X) + max(aVec) - min(dVec) # This is the largest possible lateness (used for lateness indicator)
    
    nJ = len(aVec)
    nV = (nJ+3)*nJ # the total number of variables
    
    # order of variables: w, s, v, u
    
    # Define objective function
    objVec = np.zeros(nV) # arrays counting zeros of nV
    objVec[:nJ] = qVec # all arrival times until nJ-1
    objVec[nJ:2*nJ] = rVec 
    objVec[2*nJ:3*nJ] = pVec 
    
    # Define equality and inequality constraints
    Aeq = np.zeros((2*nJ,nV))
    Aineq = np.zeros((5*nJ-1,nV))
    #Construct the row and column constraints one by one
    for jr in range(nJ):
        Aeq[jr,(3+jr)*nJ:(4+jr)*nJ] = 1 # row eq. constraints
        Aeq[jr+nJ,3*nJ+jr:3*nJ+nJ**2:nJ] = 1# col eq. constraints
        
        # Positive lateness inequality constraint
        Aineq[jr,jr] = -1
        
        # Lateness indicator
        Aineq[nJ+jr,jr] = 1 # coef of w
        Aineq[nJ+jr,jr+2*nJ] = -Cbig # coef of v
        
        # Lateness calculation
        Aineq[2*nJ+jr,jr] = -1 # coef of w
        Aineq[2*nJ+jr,nJ+jr] = 1 # coef of s
        Aineq[2*nJ+jr,(3*nJ+jr):(3+nJ)*nJ:nJ] =xVec - dVec # coef of u
        
        
        #starting time after arrival time inequality constraints
        Aineq[3*nJ+jr , nJ+jr] =-1 # coef. of s
        Aineq[3*nJ+jr , 3*nJ+jr:(3+nJ)*nJ:nJ] =aVec  
        
        #starting time after completion time inequality constraints
        if jr<nJ-1:
            Aineq[4*nJ+jr,nJ+jr] = 1
            Aineq[4*nJ+jr,nJ+jr+1] = -1
            Aineq[4*nJ+jr , 3*nJ+jr:(3+nJ)*nJ:nJ] =xVec  
        
    objVec = matrix(np.array(objVec, dtype=float))
    Aeq = matrix(np.array(Aeq, dtype=float))
    bEq = matrix(np.ones(2*nJ))
    
    Aineq = matrix(Aineq, tc='d')
    bIneq = matrix(np.zeros((5*nJ-1, 1), dtype=float), tc='d')
    
    # pickle.dump([Aineq,bIneq,Aeq,bEq], open("trial.pkl", "wb"))
    
    status, xNew = ilp(objVec, Aineq, bIneq, Aeq, bEq, I=set(), B=set(range(2*nJ,(nJ+3)*nJ)))
    # print(" status ", status)
    jobMx = np.reshape(xNew[-nJ*nJ:],(nJ,nJ))
    J = np.arange(nJ) @ jobMx# ### Jobs from 0 to N-1
    J = J.astype(int)
    return J


def  MILPsched2M(availTimes,dueTimes,X,Z,pVec,qVec):
## Optimized schedule for two machines
# Constants:  
# a_1,…a_N (arrival time, including information delay ); 
# d_1,…d_N  (due times), 
# x_1i,x_2i…x_Ni: Processing times, where i=1,2 and x_ni is the processing time of the job with index n on machine i.
# z_1i  z_2i….z_(Ni)Waiting times, where i=1,2 and z_ni is the waiting time of the job with index n on machine i.
# p_1,…p_N (penalty (cost)  if job is late);  
# q_1,…q_N (lateness penalty coefficient).

    if type(pVec)!="numpy.ndarray":
        pVec=pVec+0*availTimes
        qVec=qVec+0*availTimes
    
    nJ = len(availTimes)
    objW = qVec
    objS1=np.zeros(nJ**2)
    objS2=np.zeros(nJ**2)
    objV=qVec
    objU1=np.zeros(nJ**2)
    objU2=np.zeros(nJ**2)
    objVec = np.concatenate((objW,objS1,objS2,objV,objU1,objU2))

    Aeq,Aineq,bIneq = MILPcreate2M_a(availTimes,dueTimes,X,Z) 
    Aeq_b,Aineq_b,bIneq_b = MILPcreate2M_b(availTimes,dueTimes,X,Z) 
            
    objVec = matrix(np.array(objVec, dtype=float))
    Aeq = matrix(np.array(Aeq, dtype=float))
    bEq = matrix(np.ones(4*nJ))
    
    Aineq = matrix(Aineq, tc='d')
    bIneq = matrix(bIneq, tc='d')
    
    pickle.dump([Aineq,bIneq,Aeq,bEq], open("trial.pkl", "wb"))
    
    status, xNew = ilp(objVec, Aineq, bIneq, Aeq, bEq, I=set(), B=set(range(nJ+2*nJ**2,2*nJ+4*nJ**2)))
    print(" status ", status)
    jobMx = np.reshape(xNew[-nJ*nJ:],(nJ,nJ))
    J = np.arange(nJ) @ jobMx # ### Jobs from 0 to N-1
    J = J.astype(int)
    return J

def  MILPcreate2M_a(availTimes,dueTimes,xMat,zMat):
## Create matrix for MILP (initial version)
# Constants:  
# a_1,…a_N (arrival time, including information delay ); 
# d_1,…d_N  (due times), 
# x_1i,x_2i…x_Ni: Processing times, where i=1,2 and x_ni is the processing time of the job with index n on machine i.
# z_1i  z_2i….z_(Ni)Waiting times, where i=1,2 and z_ni is the waiting time of the job with index n on machine i.

    aVec = availTimes
    dVec = dueTimes

    Cbig = np.sum(X) +np.sum(zMat) + max(aVec) 
    # This is the largest possible lateness or starting time (used for lateness indicator and for starting time constraint)
    
    nJ = len(aVec)
    nV = (4*nJ+2)*nJ # Number of variables
    
    # Define equality and inequality constraints
    Aeq = np.zeros((4*nJ,nV)) 
    Aineq = np.zeros((2*nJ**2+(7*nJ-2),nV)) 
    bIneq = np.zeros((2*nJ**2+(7*nJ-2), 1))
    #Construct the row and column sconstraints one by one
    for jr in range(nJ):
        Aeq[jr,(2*nJ+2*nJ**2+jr):(3*nJ+2*nJ**2+jr)] = 1 # row eq. constraints for machine 1
        Aeq[jr+nJ,(2*nJ+2*nJ**2+jr):(2*nJ+3*nJ**2+jr):nJ] = 1# col eq. constraints for machine 1
        Aeq[jr+2*nJ, (2*nJ+3*nJ**2+jr):(3*nJ+3*nJ**2+jr)] = 1 # row eq. constraints for machine 2
        Aeq[jr+3*nJ,(2*nJ+3*nJ**2+jr):(2*nJ+4*nJ**2+jr):nJ] = 1# col eq. constraints for machine 2
        
        # Positive lateness inequality constraint
        Aineq[jr,jr] = -1
        
        # Lateness indicator
        Aineq[nJ+jr,jr] = 1 # coef of w
        Aineq[nJ+jr,jr+nJ+2*nJ**2] = -Cbig # coef of v # 
        
        # Lateness calculation
        Aineq[2*nJ+jr,jr] = -1 # coef of w
        Aineq[2*nJ+jr,nJ+nJ**2+jr:nJ+2*nJ**2+jr:nJ] = 1 # coef of s
        Aineq[2*nJ+jr,(2*nJ+3*nJ**2+jr)::nJ] =xMat[:,1] - dVec # coef of u
        
        #starting time after arrival time inequality constraints
        Aineq[3*nJ+jr , nJ+jr*nJ:2*nJ+jr*nJ] =-1 # coef. of ss for machine 1 
        bIneq[3*nJ+jr, 0] = -(aVec[jr] + zMat[jr,0])
        
        Aineq[4*nJ+jr , nJ+jr*nJ+nJ**2:2*nJ+jr*nJ+nJ**2 ] =-1 # coef. of s for machine 2
        Aineq[4*nJ+jr , nJ+jr*nJ:2*nJ+jr*nJ] = 1 # coef. of s for machine 2  
        bIneq[4*nJ+jr, 0] = -(xMat[jr,0] + zMat[jr,1])
        
        #starting time after completion time inequality constraints
        if jr<nJ-1:
            Aineq[5*nJ+jr,nJ+jr:nJ+jr+nJ**2:nJ] = 1 # coef. of S_nk1
            Aineq[5*nJ+jr,nJ+jr+1:nJ+jr+1+nJ**2:nJ] = -1 # coef. of S_nk1
            Aineq[5*nJ+jr , 2*nJ+nJ**2+jr:2*nJ+jr+2*nJ**2:nJ] =xMat[:,0] 
        
            Aineq[6*nJ+jr-1,nJ+nJ**2+jr:nJ+nJ**2+jr+nJ**2:nJ] = 1 # coef. of S_nk2
            Aineq[6*nJ+jr-1,nJ+nJ**2+jr+1:nJ+nJ**2+jr+1+nJ**2:nJ] = -1 # coef. of S_nk2
            Aineq[6*nJ+jr-1 , 2*nJ+nJ**2+nJ**2+jr:2*nJ+nJ**2+jr+2*nJ**2:nJ] =xMat[:,1]  
        for js in range(nJ):
            Aineq[7*nJ-1+jr+js*nJ,nJ+jr+js*nJ]=1
            Aineq[7*nJ-1+jr+js*nJ,2*nJ+2*nJ**2+jr+js*nJ]=-Cbig
            
            Aineq[8*nJ-1+nJ**2,nJ+nJ**2+jr+js*nJ]=1
            Aineq[8*nJ-1+nJ**2,2*nJ+3*nJ**2+jr+js*nJ]=-Cbig
            
    return Aeq,Aineq,bIneq

def  MILPcreate2M_b(availTimes,dueTimes,xMat,zMat):
## Create matrix for MILP (alternative version which is simpler to understand)
# Constants:  
# a_1,…a_N (arrival time, including information delay ); 
# d_1,…d_N  (due times), 
# x_1i,x_2i…x_Ni: Processing times, where i=1,2 and x_ni is the processing time of the job with index n on machine i.
# z_1i  z_2i….z_(Ni)Waiting times, where i=1,2 and z_ni is the waiting time of the job with index n on machine i.

    aVec = availTimes
    dVec = dueTimes
    Cbig = np.sum(xMat) + np.sum(zMat) + max(aVec) 
    # This is the largest possible lateness or starting time (used for lateness indicator and for starting time constraint)
    
    N = len(aVec)
    
     
 # Equality constraints    
 # ∑_n u_nk1 = 1 ∀ k; # Matching condition for matrix 1
    W=np.zeros((N,N))
    S1=np.zeros((N,N**2))
    S2=np.zeros((N,N**2))
    V=np.zeros((N,N))
    U2=np.zeros((N,N**2))
    U1=np.zeros((N,N**2))
    for k in range(N):
        U1[k,k::N]=1
    # Put matrices together as a single matrix
    Aeq1 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1)
   
 # ∑_k u_nk1 = 1 ∀ n; #  Matching condition for matrix 2  
    U1=np.zeros((N,N**2))
    for k in range(N):
        U1[k,N*k:N*k+N]=1
    # Put matrices together as a single matrix
    Aeq2 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1)
   
 # ∑_n u_nk2 = 1 ∀ k;  #  Matching condition for matrix 3
    U1=np.zeros((N,N**2))
    for k in range(N):
        U2[k,k::N]=1
    # Put matrices together as a single matrix
    Aeq3 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1)
     
 # ∑_k u_nk2 = 1 ∀ n;  #  Matching condition for matrix 4  
    U2=np.zeros((N,N**2))
    for k in range(N):
        U2[k,N*k:N*k+N]=1
    # Put matrices together as a single matrix
    Aeq4 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1)
   
    Aeq = np.concatenate((Aeq1,Aeq2,Aeq3,Aeq4), axis = 0)
    
  # Inequality constraints
    # Positive lateness inequality constraint
    U2=np.zeros((N,N**2))
    for k in range(N):
        W[k,k] = -1
    Aineq1 =  np.concatenate((W,S1,S2,V,U2,U1), axis =1)
    bineq1 = np.zeros(N)
    
 # Lateness indicator
    W=np.zeros((N,N)) 
    V=np.zeros((N,N))
    for k in range(N):
        W[k,k]= 1
        V[k,k]= -Cbig
    Aineq2 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1)
    bineq2 = np.zeros(N)

# Lateness calculation
    W=np.zeros((N,N))  
    V=np.zeros((N,N))
    print(np.shape(xMat))
    for k in range(N):
        W[k,k] = -1
        S2[k,k::N]= 1
        U2[k,k::N]= xMat[:,1]- dVec[:]
    Aineq3 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1) 
    bineq3 = np.zeros(N)

# starting time after arrival time inequality constraints
    W=np.zeros((N,N)) 
    S2=np.zeros((N,N**2))
    U2=np.zeros((N,N**2))
    for k in range(N):
        S1[k,N*k:N*k+N]=-1
    Aineq4 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1) # for machine 1
    bineq4 = -zMat[:,0] - aVec
    
    S1=np.zeros((N,N**2))
    for k in range(N):
        S1[k,N*k:N*k+N]= 1
        S2[k,N*k:N*k+N]=-1
    Aineq5 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1) # for machine 2
    bineq5 = -zMat[:,1] - xMat[:,0]
    
# starting time after completion time inequality constraints       
    S1=np.zeros((N,N**2))
    S2=np.zeros((N,N**2))
    for k in range(N):
        S1[k,k::N] = 1
        S1[k,k+1::N] = -1
        U1[k,k::N] = xMat[:,0] 
    Aineq6 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1) # for machine 1
    bineq6 = np.zeros(N)
    
    U1=np.zeros((N,N**2))  
    S1=np.zeros((N,N**2))
    for k in range(N):
        S2[k,k::N] = 1
        S2[k,k+1::N] = -1
        U2[k,k::N] = xMat[:,1] # for machine 2
    Aineq7 = np.concatenate((W,S1,S2,V,U2,U1), axis = 1) # for machine 2
    bineq7 = np.zeros(N)
    Aineq= np.concatenate((Aineq1,Aineq2,Aineq3,Aineq4,Aineq5,Aineq6,Aineq7), axis = 0)
    bIneq= np.concatenate((bineq1,bineq2,bineq3,bineq4,bineq5,bineq6,bineq7))
    
    return Aeq, Aineq,bIneq




def MILP_Sched(A,D,X,Z,pVec,qVec,rVec,S,Y):
    # New scheduling using machinewise MILP scheduling
    J=np.empty(X.shape).astype(int)
    Y=np.empty(X.shape)
    S=np.empty(X.shape)
    Idle =np.empty(X.shape)
    N,M=X.shape
    # First stage scheduling
    availTimes = A + Z[:,0]
    for m in range(M):
        print("stage",m)
        # Estimate due times at this stage
        # Actual due time minus execution and waiting times for subsequent machines.
        if len(Idle)==0:
            dueTimes = D - np.sum(X[:,m+1:],axis=1)-np.sum(Z[:,m+1:],axis=1)
        else:
            # If previous schedule exists, estimated due time is equal to start time on next machine plus the waiting intevals plus the lateness.
            if m < M-2:
                dueTimes = S[:,m+1] + np.sum(S[:,m+2:]-Y[:,m+1:-1],axis=1) + (D[:]-Y[:,-1])
            elif m == M-2:
                dueTimes = S[:,m+1]  + (D[:]-Y[:,-1])
            else:
                dueTimes = D
            
        J[:,m] = MILPsched1M(availTimes,dueTimes,X[:,m],pVec,qVec,rVec)
        # Compute S,Y,Idle for this machine
        S[J[0,m],m]=availTimes[J[0,m]]
        Y[J[0,m],m] =  S[J[0,m],m]+X[J[0,m],m]
        Idle[J[0,m],m] =  S[J[0,m],m]
        for n in range(1,N):
            S[J[n,m],m] = max(Y[J[n-1,m],m],availTimes[J[n,m]])
            Y[J[n,m],m] =S[J[n,m],m]+X[J[n,m],m]
            Idle[J[n,m],m] = S[J[n,m],m]- Y[J[n-1,m],m]
        if m<M-1:
            availTimes=Y[:,m]+Z[:,m+1] 
        # plotSched(A,Z[:,:m+1],S[:,:m+1],Y[:,:m+1],D)
            
    return J,S,Y,Idle


    
debug = False
if debug:
    A = np.array([ 1.63137284,  5.44267173, 12.75002244])
    D = np.array([82.70786561,  6.62613236, 44.9384129 ])
    X = np.array([[79.14671014],
           [ 0.71659951],
           [17.88842656]])
    Z = np.array([[8.83326597e-14],
           [1.64407158e-14],
           [4.59892195e-11]])
    [Jnew,Snew,Ynew,IdleNew]=MILP_Sched(A,D,X,Z,0.5,0.5,[],[])