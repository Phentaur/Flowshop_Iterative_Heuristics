import numpy as np
from Functions.InsertionHeuristicFunctions import *

# Function to evaluate the lateness and finish time for a given permutation of jobs

def eval_perm_ct_fn(atmp, xtmp, dtmp, ptmp, qtmp):
    ntmp = len(atmp)
    ytmp = np.zeros(ntmp)  # Initialize the list of finish times
    ytmp[0] = atmp[0] + xtmp[0]  # Finish time for the first task
    late = max(ytmp[0] - dtmp[0], 0)  # Calculate the lateness for the first task
    late = qtmp[0] * late + ptmp[0] * (late > 0)  # Calculate the lateness penalty

    # Calculate finish times and lateness penalties for each task
    for k in range(1, ntmp):
        ytmp[k] = max(ytmp[k - 1], atmp[k]) + xtmp[k]  # Finish time of task k
        late_add = max(0, ytmp[k] - dtmp[k])  # Calculate lateness of task k
        late += qtmp[k] * late_add + ptmp[k] * (late_add > 0)  # Add the lateness penalty

    return late, float(ytmp[-1])


#Earliest Due Date 
def EDD_order(A,X,D):
    EDD = np.argsort(D)  
    return  EDD
    
# Shortest Processing Time
def  SPT_order(A,X,D):

    SPT = np.argsort(X)  
    return  SPT
    
# Slack Time Remaining
def  STR_order(A,X,D):

    STR = np.argsort(D - X)  
    return  STR
    
# Critical Ratio Rule
def  CRR_order(A,X,D):

    CRR = np.argsort(D / X)  
    return  CRR
    
# Arrive
def  Arrive_order(A,X,D):

    CRR = np.argsort(A)  
    return  CRR
    
def First_arrive_order(A,X,D):
    return  np.arange(D.shape[0])
 


 
def Best_order(A,X,D,p,q):

    A = A.reshape(D.shape)  # Arrival times
    X = X.reshape(D.shape)
    D = D  # Due times
    N = len(D)  # Number of jobs
    if type(p) != "numpy.ndarray":
        p = p + 0 * A
        q = q + 0 * A
    lates= []
    finishs =[]
    flowO, lat, fin = insertHeurFn(A,D,X,p,q)
    orders = [EDD_order(A,X,D), SPT_order(A,X,D),flowO,CRR_order(A,X,D),Arrive_order(A,X,D),First_arrive_order(A,X,D)]
    for order in orders:
        lates.append(eval_perm_ct_fn(A[order], X[order], D[order], p, q)[0])
        finishs.append(eval_perm_ct_fn(A[order], X[order], D[order], p, q)[1])
    bon = np.argsort(lates)
        
    return orders[bon[0]]