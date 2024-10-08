# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:39:04 2023

@author: matth
"""

from Functions.FlowshopProblemClassDefs import *
from Functions.FlowshopProblemCreationRoutines_v1 import *
# from Functions.FlowshopMILPschedRoutines import *
from Functions.FlowshopBaseSchedRoutines import *
from scipy.optimize import linprog
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


pickleFilename = 'heuristic_v_MILP_8N.pickle'
dataFilename="single_machine_8N.pickle"
writeToPickle=True

# Number of instances generated
nInstances=12000

NKEEP = 60
#NKEEP=100 #Number of permutations to keep

N_jobs,M_stage,K_sen,mu_Ar,mu_Ai,mu_Dint=8,1,1,5,1E10,10

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

def evalPermCT_fn(Atmp, Xtmp, Dtmp, Ptmp, Qtmp):
    nTmp = len(Atmp)
    # Y will hold finish times
    Ytmp = np.zeros(nTmp)
    # Finish time of first job
    Ytmp[0] = Atmp[0] + Xtmp[0]
    # Late penalty of first job
    late = max(Ytmp[0] - Dtmp[0], 0)
    late = Qtmp[0]*late + Ptmp[0]*(late>0)
    
    # Calculate finish times and late penalties of jobs one by one
    for k in range(1, nTmp):
        # Finish time of k'th job
        Ytmp[k] = max(Ytmp[k - 1], Atmp[k]) + Xtmp[k]
        # Late penalty of k'th job
        late_add = max(0, Ytmp[k] - Dtmp[k])
        late = late + Qtmp[k]*late_add + Ptmp[k]*(late_add>0)
    return late,float(Ytmp[-1])

def nkeep_fn(tmp_lateness, tmp_finish,tmp_perms):
    indices = np.lexsort((tmp_finish, tmp_lateness))

    # Sort each array using the sorted indices
    tmp_lateness_sorted = tmp_lateness[indices]
    tmp_finish_sorted = tmp_finish[indices]
    tmp_perms_sorted = tmp_perms[indices]
    
    tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted = select_nkeep_items(tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted)

    # Output the sorted arrays
    return tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted

def select_nkeep_items(tmp_lateness, tmp_finish, tmp_perms):
    # Calculate the step to select items with as much diversity as possible
    step = len(tmp_lateness) // NKEEP
    
    # Select indices based on the step
    indices = [i * step for i in range(NKEEP)]
    
    indices[-1] = len(tmp_lateness) - 1
    
    # Select items based on the calculated indices
    selected_lateness = [tmp_lateness[i] for i in indices]
    selected_finish = [tmp_finish[i] for i in indices]
    selected_perms = [tmp_perms[i] for i in indices]

    return selected_lateness, selected_finish, selected_perms

def pruneList_fn(tmp_lateness, tmp_finish,tmp_perms):
    late_compare_gt = (tmp_lateness.reshape((1,-1))>tmp_lateness.reshape((-1,1))).astype(int)
    finish_compare_gt = (tmp_finish.reshape((1,-1))>tmp_finish.reshape((-1,1))).astype(int)
    
    index_remove_mx = (late_compare_gt * finish_compare_gt) 
    ix = (np.sum(index_remove_mx,axis=0)==0)
    return ix

def generate_permutations_in_stages_CT(partial_list, A, X, D, P, Q):
    job0 = partial_list[0]  
    remaining_jobs = partial_list[1:]

    result = np.array([[partial_list[0]]])
    length=1 #starting length of  partial_lists

    
    # Run for each job not yet included in the starting partial_list
    for next_job in remaining_jobs:
        length=length+1#
        new_perms = np.zeros((len(result)*length, length), dtype=int)
        new_lateness = np.zeros(len(result)*length)
        new_finish = np.zeros(len(result)*length)

        # For each saved partial_list from the last stage, run the insertion permutations
        for i, partial_list in enumerate(result):
            for j in range(length):
                nn =  j+length*i
                new_perm = np.insert(partial_list, j, next_job)
                new_perms[nn,:]=new_perm
                # Calculate lateness and finishing time for the new permutation
                # Define ordered subarrays for the calculation, in agreement with the current permutation
                new_lateness[nn], new_finish[nn] \
                    = evalPermCT_fn(A[new_perm],X[new_perm],
                                    D[new_perm],P[new_perm],Q[new_perm])
                
                
        # Prune list to keep only Pareto optimal orderings.
        ix = pruneList_fn(new_lateness, new_finish,new_perms)
        result = new_perms[ix,:]
        new_lateness = new_lateness[ix]
        new_finish = new_finish[ix]
        if len(result)>NKEEP:
            new_lateness, new_finish, result = nkeep_fn(new_lateness,new_finish,result)
    return result, new_lateness, new_finish

def generate_permutations_in_stages_CT_Ins(partial_list, A, X, D, P, Q, insert):
    job0 = partial_list[0]  
    remaining_jobs = partial_list[1:]
    # print("Starting with: ", job0, " still need: ", partial_list)

    result = np.array([[partial_list[0]]])
    length=1 #starting length of  partial_lists

    
    # Run for each job not yet included in the starting partial_list
    for next_job in remaining_jobs:
        length+=1
        new_perms = np.zeros((len(result)*length, length), dtype=int)
        new_lateness = np.zeros(len(result)*length)
        new_finish = np.zeros(len(result)*length)
        if length>insert:
            new_perms = np.zeros((len(result)*insert, length), dtype=int)
            new_lateness = np.zeros(len(result)*insert)
            new_finish = np.zeros(len(result)*insert)

        # For each saved partial_list from the last stage, run the insertion permutations
        for i, partial_list in enumerate(result):
            if length >= insert:
                start_pos = length - insert
                positions = range(start_pos, length)
            else:
                positions = range(length)
            for j, pos in enumerate(positions):
                nn = j + len(positions) * i
                new_perm = np.insert(partial_list, pos, next_job)
                new_perms[nn,:]=new_perm
                # Calculate lateness and finishing time for the new permutation
                # Define ordered subarrays for the calculation, in agreement with the current permutation
                new_lateness[nn], new_finish[nn] \
                    = evalPermCT_fn(A[new_perm],X[new_perm],
                                    D[new_perm],P[new_perm],Q[new_perm])
                # print("New order: ",new_perm," with Lateness: ",new_lateness[nn],' and Finishing: ',new_finish[nn])
                
                
        # Prune list to keep only Pareto optimal orderings.
        ix = pruneList_fn(new_lateness, new_finish,new_perms)
        result = new_perms[ix,:]
        new_lateness = new_lateness[ix]
        new_finish = new_finish[ix]
        if len(result)>NKEEP:
            new_lateness, new_finish, result = nkeep_fn(new_lateness,new_finish,result)
    return result, new_lateness, new_finish

# Sort by relative due time
def flow_heurDX_CT(A,D,X,P,Q):
  D= D.flatten() #due times
  A= A.reshape(D.shape) #arrival times
  X= X.reshape(D.shape)
  Due=D-X
  N=len(Due) #Number of jobs
  if type(P)!="numpy.ndarray":
    P=P+0*A
    Q=Q+0*A

    # Holds the partial_list
  partial_list=np.argsort(Due)
  
      # result_permutations= list of job partial_lists, lateness= list of lateness for each of those partial_lists, finish= finishing time of those partial_lists
  
  result_permutations, lateness, finish = generate_permutations_in_stages_CT(partial_list, A, X, D, P, Q)
  
  # Select perms. with minimum lateness
  ix0 = np.where(lateness==min(lateness))[0]
  if isinstance(ix0, (int, np.int64)):
    iSelect = ix0
  else:
      # Get finish times for these permutations
      finish0 = finish[ix0[0]]
      if isinstance(finish0, np.ndarray):
          # Select first permutation with minium finish time
          ix1 = np.where(finish0==min(finish0))[0][0]
          iSelect = ix0[ix1]
      else:
         iSelect = ix0[0]
  
  return result_permutations[iSelect], lateness[iSelect], finish[iSelect]

# Sort by relative due time
def flow_heurDX_CT_Ins(A,D,X,P,Q,insert):
  D= D.flatten()
  A= A.reshape(D.shape) #arrival times
  X= X.reshape(D.shape)
  Due=D-X
  N=len(Due) #Number of jobs
  if type(P)!="numpy.ndarray":
    P=P+0*A
    Q=Q+0*A

    # Holds the partial_list
  partial_list=np.argsort(Due)
  
      # partial_listPartial= list of job partial_lists, lateness= list of lateness for each of those partial_lists, finish= finishing time of those partial_lists
  
  result_permutations, lateness, finish = generate_permutations_in_stages_CT_Ins(partial_list, A, X, D, P, Q, insert)
  
  # Select perms. with minimum lateness
  ix0 = np.where(lateness==min(lateness))[0]
  if isinstance(ix0, (int, np.int64)):
    iSelect = ix0
  else:
      # Get finish times for these permutations
      finish0 = finish[ix0[0]]
      if isinstance(finish0, np.ndarray):
          # Select first permutation with minium finish time
          ix1 = np.where(finish0==min(finish0))[0][0]
          iSelect = ix0[ix1]
      else:
         iSelect = ix0[0]
  
  return result_permutations[iSelect], lateness[iSelect], finish[iSelect]

#### Testing nkeep_fn:
# late=np.array([10,3,5,3,9,3])
# fin=np.array([3,4,6,9,8,7])
# perm=np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
# tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted = nkeep_fn(late, fin, perm)
# print("Sorted tmp_lateness:", tmp_lateness_sorted)
# print("Sorted tmp_finish:", tmp_finish_sorted)
# print("Sorted tmp_perms:", tmp_perms_sorted)

# selected_lateness, selected_finish, selected_perms = select_nkeep_items(
#     tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted
# )

# # Output the selected arrays
# print("Selected tmp_lateness:", selected_lateness)
# print("Selected tmp_finish:", selected_finish)
# print("Selected tmp_perms:", selected_perms)


#### Testing flow_HeurDX_CT_Ins:
As = np.array([0,0.574746,5.16762,12.4096,14.2983,22.7059,22.9958,26.3943,32.1234,36.5723,38.449,38.9187,49.7048,50.9033,56.8808,62.2405,66.316,74.7409,78.4623,79.8485,86.4044,87.3429,87.5217,88.8794,
               92.4937,92.5973,96.942,97.0013,100.243,101.505,105.696,112.423,117.452,123.79,124.047,126.471,129.675,144.502,146.892,158.588,158.691,159.098,159.297,164.52,169.285,175.875,183.517,200.378,204.641,214.895
])
Ds = np.array([22.0396,17.303,15.3552,19.7576,55.6054,76.4801,49.4494,55.1,74.7472,42.9247,60.2903,57.1205,74.5317,64.8794,75.2274,96.7303,
76.7136,93.7989,96.5133,114.171,93.1639,99.4436,96.2646,115.733,98.838,102.822,168.673,110.571,121.864,144.801,120.538,135.995,146.875,
147.549,140.464,153.545,172.529,157.895,152.271,177.891,173.998,176.108,166.179,178.331,193.902,186.198,199.425,211.564,225.235,242.578
]).reshape([-1,1])
Xs = np.array([3.41126,6.33374,4.91497,2.61372,19.0877,22.0303,11.4663,7.75018,23.682,3.17542,16.821,2.96538,5.34392,3.37814,0.0755449,0.175378,
               8271,9.70133,5.39221,20.0552,4.5867,1.02421,2.60932,0.543228,4.23618,1.92448,13.0294,7.79192,9.79539,12.4071,3.98225,5.19226,11.2148,
               7.38731,4.42827,0.767703,9.15834,1.12786,0.553222,4.2033,5.83491,2.39155,0.305857,7.26218,3.28951,3.0985,8.19902,1.02803,8.28232,12.2979
])
order, lateness, finishing = flow_heurDX_CT(As,Ds,Xs,p,q)
#order, lateness, finishing = flow_heurDX_CT_Ins(As,Ds,Xs,p,q,3)