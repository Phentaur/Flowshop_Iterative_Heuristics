# -*- coding: utf-8 -*-
'''
This code contains the functions needed to run the NN multimachine
'''
#
# import libraries
import time
from Functions.FlowshopMILPschedRoutines import  MILPsched1M
from Functions.FlowshopBaseSchedRoutines import Obj_pq
import numpy as np
import copy as cp
# Initialize random number generator
from numpy.random import default_rng
import pickle
import statistics as stats
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from Functions.FlowshopStochastic import newStochSolnFn
import tensorflow as tf
from Functions.InsertionHeuristicFunctions import *

useWeights=False #Set to True for NN Weights, False for Heuristic. Current Heuristic: insertHeurFn (line 269)
P=50
Q=1

model = load_model("G:/.shortcut-targets-by-id/17tL5VZqY-TmeoFKkcbav05i9LclR3-Oc/NNsched/Codes/2_Single_Machine_NN/Models/singleMachineModel_6N.h5")
# filePath='G:/.shortcut-targets-by-id/17tL5VZqY-TmeoFKkcbav05i9LclR3-Oc/Matt/Codes/2_Single_Machine_NN/NNvsMILPdata.pickle'

weights=[]
biases=[]
for i, layer in enumerate(model.layers):
  a=0
  while a<=i:
    try:
      weights.append(layer.get_weights()[0])
      biases.append(layer.get_weights()[1])
      break
    except:
      a+=1
      continue

rng = default_rng()


# Performs stochastic optimization
def stochFn(A,D,X,Z,nS,nA,nI,pVec,qVec,rVec,stochParams):
    # A,D,X,Z pVec, qVec are as in the main program
    # nS is the number of solutions initially generated
    # nA is the number of additional solutions generated until solutions are reduced back to nS
    # nI is the number of iterations
    # Initialize best objective value (will be updated)
    # (since no solutions, set as large value)
    
    start = time.time() # Function start time
    
    # Initialize constants for selection
    maxAge, threshold = stochParams
    objBest = 1E50 # Initial objective function 
    
 
    # N,M  = number of jobs, machines
    N,M=X.shape
    # Array of latest possible due times (N by M)
    # (assuming there is no additional delay between machines)
    # This is computed once, because it doesn't change.
    #Initialize
    latestDue = np.zeros_like(X)
    # Compute latest due times
    for jm in range(M):
        # Compute current latest due times for jobs on this machine
        latestDue[:,jm]=D
        if jm < M:
            latestDue[:,jm] = latestDue[:,jm] -( np.sum(X[:,jm+1:],axis=1)
                 + np.sum(Z[:,jm+1:],axis=1))

    ### Step 1:  Generate nS initial stochastic solutions
    initSoln = [0]*(nS+nA) # New solutions will be placed in this list
    initObjVec = np.zeros(nS+nA) # Vector of obj. values for solutions in list
    # Generate new solutions one by one
    for js in range(nS):
        # Function for generation of the initial solution
        newSoln = initStochSolnFn(A,latestDue,X,Z,pVec,qVec,rVec,"random")
        initSoln[js]=cp.deepcopy(newSoln)
        # Save objective function value in vector
        initObjVec[js] = newSoln["obj"]
    
    # This identifies solutions with non-empty candidate lists
    availSolns = list(range(nS))
    # Record number of times solution has been chosen
    Age = np.zeros(nS + nA)
    nS_now = nS
    initObjVecBest = 1E50
    for jI in range(nI):
        # Create new solutions
        listOfSoln, availSolns, Age, objVec, Monitor = solnListExpandFn(
            initSoln,availSolns,nS_now,Age,initObjVec,M,A,D,X,Z,pVec,qVec,rVec,threshold)
        # Find best new solution and update overall best solution
        
        
        bestObjTmp = np.argmin(objVec)
        # Compare with overall best solution
        # If it's better, then replace
        if objVec[bestObjTmp]<objBest:
            objBest = objVec[bestObjTmp]
            bestSoln = cp.deepcopy(listOfSoln[bestObjTmp])
            print("Best objective achieved so far",objBest," with age ",Age[bestObjTmp])
        
        # If no more available solutions or if perfect solution, then terminate
        if (len(availSolns) == 0) or (objBest == 0):
            Func_Time = time.time() - start # Function runtime
            return listOfSoln, 0*Age, objVec, bestSoln, Func_Time, jI, A, Z, D, Monitor
        
        # Remove solutions with Age above maxAge
        nS_now=0
        while (nS_now == 0):
            youngSoln  = np.where(Age<=maxAge)[0]
            # Increase maxAge if no solutions
            nS_now = min(len(youngSoln),nS)
            maxAge = maxAge*(1 + 0.5*(nS_now==0))
            if nS_now==0:
                print("maxAge updated to", maxAge)
        # Rank solutions
        rankVec = objVec[youngSoln]
        # Sort rank vector from smallest to largest
        sortIx = np.argsort(rankVec)
        sortVec = rankVec[sortIx]
        # Remove duplicates
        sortIx = sortIx[np.where(sortVec[1:]-sortVec[:-1]>1E-4)]
        # nS = min(nS,len(sortIx))
        
        # choose nS smallest from original list
        bestVec = youngSoln[sortIx[:nS_now]]
        
        # Duplicate Age to reorder
        AgeNew = np.zeros_like(Age)
        # Select these solutions as the new set of initial
        for js in range(nS_now):
            thisSoln = listOfSoln[bestVec[js]]
            initSoln[js] = cp.deepcopy(thisSoln)
            initObjVec[js] = thisSoln["obj"]
            AgeNew[js]=Age[bestVec[js]]
        # reset the list of available solutions
        availSolns = list(range(nS_now))
        # Remove solutions that have already been checked from available solutions.
        for i in range(nS_now):
            if not initSoln[i]["CL"]:
                availSolns.remove(i)
                
        if initObjVec[0] != initObjVecBest: # If there is a new best solution...
            initObjVecBest = initObjVec[0] # ...replace the old best solution,
            timeObjVecBest = 0 # and set the amount of time it has been the best to zero.
        else: # Otherwise...
            timeObjVecBest += 1 # ...age the current best by 1.
        if timeObjVecBest == 20: # If there has been no change in 20 iterations...
            print(f"Best solution at iteration {jI}: {initObjVec[0]}.")
            print("Optimization terminated. No improvement in 20 iterations.")
            Func_Time = time.time() - start # Function runtime
            return initSoln, AgeNew, initObjVec, bestSoln, Func_Time, jI, A, Z, D, Monitor # ...terminate the program.
        print(f"Best solution at iteration {jI}: {initObjVec[0]}")
   
    Func_Time = time.time() - start # Function runtime
    
    
   
    return initSoln, AgeNew, initObjVec, bestSoln, Func_Time, jI, A, Z, D, Monitor
     
# Generate initial random solutions
def initStochSolnFn(A,latestDue,X,Z,pVec,qVec,rVec,iType):
    N,M=X.shape
    # Create a stochastic solution dictionary to store generated solutions
    stochSoln = {}
    
    # Initialize arrays needed for computation
    tgtDueTmp = np.zeros_like(X)
    Jtmp = np.int_(tgtDueTmp)
    Ytmp = np.zeros_like(X)
    
    # Set up availability time for iteration
    Atmp = 1.0*A
    
    # Loop through machines and schedule machines one by one.
    for jm in range(M):
        randExp = 1

        # Generate a random set of due times
        randVec = rng.uniform(size=N)
        ## Two options:
        # Option 1: generate purely random initial schedules
        if iType=="random":
            Jtmp[:,jm] = np.argsort(randVec)
        # Option 2: generate using MILP and random due times
        else:
            randVec = randVec**randExp
            tgtDueTmp[:,jm]= (randVec*latestDue[:,jm]
                         + (1-randVec)*(Atmp + X[:,jm]))
            tgtDueTmp[:,jm]= randVec*latestDue[:,jm]
            tgtDueTmp[:,jm] = randVec*np.max(latestDue[:,jm])
            
            # Generate optimal MILP schedule for machine jm based on these due times
            Jtmp[:,jm] = MILPsched1M(Atmp+Z[:,jm],tgtDueTmp[:,jm],X[:,jm],pVec,qVec,rVec)

        # compute finish times for jobs on machine jm
        # Index of first job scheduled on machine jm
        j0=Jtmp[0,jm]
        # Finish time for first job scheduled on machine jm
        Ytmp[j0,jm]=Atmp[j0]+Z[j0,jm]+X[j0,jm]
        
        # Compute finish times for other jobs on machine jm
        for jn in range(1,N):
            # Next job in list
            j1 = Jtmp[jn,jm]
            # update finish time
            Ytmp[j1,jm] = max(Ytmp[j0,jm],Atmp[j1]+Z[j1,jm] )+X[j1,jm]
            # Update previous job index
            j0=j1
        # Starting times for next machine
        Atmp = Ytmp[:,jm]
        # Continue loop to schedule all machines

    # Store computed schedule and finish times
    stochSoln = {}
    # J = job schedule
    stochSoln["J"] = 1*Jtmp
    # Finish times
    stochSoln["Y"] = 1.*Ytmp
    # Objective function
    stochSoln["obj"] = Obj_pq(latestDue[:,-1],Ytmp,pVec,qVec)
    stochSoln["CL"] = list(range(M))
    
    
   
    return stochSoln

# function to stochastically modify existing solution on machine m
### Outline:
# Construct new due date at level m
# Re-solve
# Re-generate Y for later levels
def newStochSolnFnNNWeight(A,D,X,Z,Soln,m,pVec,qVec,rVec):
    N,M=X.shape
    Ytmp = 1.*Soln["Y"]
    Jtmp = 1*Soln["J"]
  
    
    ## compute availability times for machine m   
    # If first machine, use initial availability
    if m==0:
        Atmp = 1.*A
    
    # Otherwise use  finish time from previous level
    else:
        Atmp = 1.*Ytmp[:,m-1]
    
    # Adjust  due time according to lateness of the job in the current schedule. 
    Dtmp = Ytmp[:,m] - (Ytmp[:,-1] - D)
    
    # Use MILP to compute schedule on current machine
 ##   Jtmp[:,m] = MILPsched1M(Atmp.flatten()+Z[:,m],Dtmp,X[:,m],pVec,qVec,rVec)
    if useWeights:
        instance=np.zeros([3,N])
        instance[0]=np.ndarray.flatten(Atmp)
        instance[1]=X[:,m]
        instance[2]=Dtmp
        samp=instance.flatten()
        for i in range(len(weights)):
            samp=(samp@weights[i])+biases[i]
        Jtmp[:,m] = np.argsort( samp )
    else:
        Jtmp[:,m], lateness, finish=insertHeurFn(A,D,X[:,m],P,Q)
            
    ## Compute finish times
    # Compute finish time for first job scheduled
    j0=Jtmp[0,m]
    Ytmp[j0,m]=Atmp[j0]+Z[j0,m]+X[j0,m]
    # Compute finish times for other jobs on this machine
    for jn in range(1,N):
        # Get next job scheduled
        j1 = Jtmp[jn,m]
        # Update finish time for this job
        Ytmp[j1,m] = max(Ytmp[j0,m],Atmp[j1]+Z[j1,m] ) + X[j1,m]
        # update previous job
        j0=j1
    
    for jm in range(m+1,M):
        j0=Jtmp[0,jm]
        Atmp = 1.*Ytmp[:,jm-1]
        Ytmp[j0,jm]=Atmp[j0]+Z[j0,jm]+X[j0,jm]
        for jn in range(1,N):
            j1 = Jtmp[jn,jm]
            Ytmp[j1,jm] = max(Ytmp[j0,jm],Atmp[j1]+Z[j1,jm] )+X[j1,jm]
            j0=j1

    # Create new stochastic solution    
    stochSoln = {}
    stochSoln["J"] = 1*Jtmp
    stochSoln["Y"] = 1.*Ytmp
    stochSoln["obj"] = Obj_pq(D,Ytmp,pVec,qVec)
    stochSoln["CL"] = list(range(M))
    
    return stochSoln

# Expands list of candidate solutions
def solnListExpandFn(initSoln,Options,nS,Age,objVec,M,A,D,X,Z,pVec,qVec,rVec,threshold):
    listOfSoln = cp.deepcopy(initSoln)

    # Total number of solutions that will be generated
    nL = len(listOfSoln)

    # Number of existing solutions--first index for new solutions
    jL = nS

    # Initialize lists for Monitor values
    Monitor = dict() # Dictionary for Monitor lists
    solTerm = list() # Has the chosen solution been terminated
    solUnch = list() # Is the 1-machine solution unchanged
    solDup = list() # Is the 1-machine solution a duplicate of a previously-generated solution
    solReject = list() # Is the 1-machine solution rejected
    solAccept = list() # Was a new solution added to the list of candidates
    objThis = list() # Objective function of the chosen solution
    objNew = list() # New objective function of solution
    objBest = [min(objVec[:jL])] # Best objective function over time
    cameFrom = list() # Where solution came from (solution that was modified)
    while jL < nL:
        # Choose an available solution at random
        if not Options:
            print("All solutions exhausted")
            break
        js = rng.choice(Options)
        CL = listOfSoln[js]["CL"]
        # Initialize Monitor values to be changed if patricular conditions are met.
        solTermTemp = False
        solUnchTemp = True
        solDupTemp = False
        solRejectTemp = False
        solAcceptTemp = False
        cameFrom.append(js)# This is solution that was modified for current

        # Increase age
        Age[js] = Age[js]+1

        # Randomly choose a machine (step 2)
        jm = rng.choice(CL)

        # Since it's been tried, remove it
        listOfSoln[js]["CL"].remove(jm)

        # If all machines tried, remove solution from options
        if not listOfSoln[js]["CL"]:
            Options.remove(js)
            solTermTemp = True
            print("No. created: ", jL, "No. active",len(Options) ,"ix. of removed: ",js,"Objective",objVec[js],"min",min(objVec[np.nonzero(objVec)]))

        # Randomly generate a new solution (Step 3)
        currentSoln = cp.deepcopy(listOfSoln[js])
        if len(Options)<0:
            newSoln = newStochSolnFn(A,D,X,Z,currentSoln,jm,pVec,qVec,rVec)
        else:
            newSoln = newStochSolnFnNNWeight(A,D,X,Z,currentSoln,jm,pVec,qVec,rVec)
        # Record values for Monitor
        objThisTemp = currentSoln["obj"]
        objNewTemp = newSoln["obj"]

        # If solution is unchanged, pass. Else, check to see if it is a duplicate.
        if np.array_equal(currentSoln["J"][:,jm], newSoln["J"][:,jm]) == False:
            # Solution is not unchanged -- set equal to False
            solUnchTemp = False
            # Check existing solutions for duplicates
            # Check existing objectives
            if newSoln["obj"] in objVec[:jL]:
                MatchSpot = int(np.argwhere(objVec[:jL] == newSoln["obj"])[0])
                #MatchSpot = int(np.argwhere(objVec[:jL] == newSoln["obj"]))
                if listOfSoln[MatchSpot]["J"].all() == newSoln["J"].all():
                    solDupTemp = True
                    solTerm.append(solTermTemp)
                    solUnch.append(solUnchTemp)
                    solDup.append(solDupTemp)
                    solReject.append(solRejectTemp)
                    solAccept.append(solAcceptTemp)
                    objThis.append(objThisTemp)
                    objNew.append(objNewTemp)
                    objBest.append(objBest[-1])
                    continue
                else:
                    print("------------------------???------------------------") # Has never printed (yet)

            # Otherwise, accept/reject according to simulated annealing (step 5)
            # https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7
            # https://en.wikipedia.org/wiki/Simulated_annealinSg
            # Generate threshold to determine probability (use min to avoid overflow)
            theta = 1 + threshold*np.sign(currentSoln["obj"]- newSoln["obj"])
            # If random number falls below threshold, then add new solution
            if np.random.rand()<theta:
                solAcceptTemp = True
                listOfSoln[jL] = cp.deepcopy(newSoln)
                # Add  new solution to the list of available solutions
                Options.append(jL)
                # Update list of objective functions
                objVec[jL] = newSoln["obj"]
                jL=jL+1
            # Solution is not good enough to save--do not add
            else:
                solRejectTemp = True
        solTerm.append(solTermTemp)
        solUnch.append(solUnchTemp)
        solDup.append(solDupTemp)
        solReject.append(solRejectTemp)
        solAccept.append(solAcceptTemp)

        objThis.append(objThisTemp)
        objNew.append(objNewTemp)
        objBest.append(min(objBest[-1],newSoln["obj"]))

        # Terminate if reached zero solution
        if objBest[-1]==0.0:
            break

    Monitor["solTerm"] = solTerm
    Monitor["solUnch"] = solUnch
    Monitor["solDup"] = solDup
    Monitor["solReject"] = solReject
    Monitor["solAccept"] = solAccept
    Monitor["objThis"] = objThis
    Monitor["objNew"] = objNew
    Monitor["objBest"] = objBest[1:]# Remove first entry which is just for priming
    Monitor["cameFrom"] = cameFrom

    return listOfSoln[:jL], Options, Age, objVec[:jL], Monitor

# scenarios=[]
# with open(filePath, 'rb') as f:
#     while True:
#         try:
#             scen = pickle.load(f)
#             scenarios.append(scen)
#         except EOFError:
#             # Reached the end of the file
#             break
        