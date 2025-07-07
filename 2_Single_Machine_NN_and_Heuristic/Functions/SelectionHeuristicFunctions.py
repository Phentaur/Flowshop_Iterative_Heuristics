'''
This file includes 'selectHeurFn', which is a function that uses the selection heuristic to
estimate an optimum schedule for a single machine.  The file also uses accessory functions

To use the selection heuristic function, you may include in your code:
    
    from SelectionHeuristicFunctions import selectHeurFn

A function call to selectHeurFn has the following form:
    
    
    perms, penalties, finish =  selectHeurFn(A, D, X, P, Q)

Inputs to select:
    
    A: Arrival times (1-d array)
    D: Due times (1-d array, same size as A)
    X: Execution times (1-d array, same size as A and D)
    P,Q:  Weight coefficients for nonzero tardiness indicator and tardiness, respectively 
          (can be scalars or arrays)
          
Outputs are:

    perms: List of permutations that have minimum penalty that also have minimum finish time.
    penalties:  penalties of permutations in 'perms' list
    finish: finish times of permutations in 'perms' list

'''
import numpy as np
import itertools

printProgress = True

# Function to evaluate the penalty and finish time for a given permutation of jobs
def eval_perm_fn(atmp, xtmp, dtmp, ptmp, qtmp):
    ntmp = len(atmp)
    ytmp = np.zeros(ntmp)  # Initialize the list of finish times
    ytmp[0] = atmp[0] + xtmp[0]  # Finish time for the first task
    late = max(ytmp[0] - dtmp[0], 0)  # Calculate the tardiness for the first task
    late = qtmp[0] * late + ptmp[0] * (late > 0)  # Calculate the penalty for the first task

    # Calculate finish times and penalties for each task
    for k in range(1, ntmp):
        ytmp[k] = max(ytmp[k - 1], atmp[k]) + xtmp[k]  # Finish time of task k
        late_add = max(0, ytmp[k] - dtmp[k])  # Calculate tardiness of task k
        late += qtmp[k] * late_add + ptmp[k] * (late_add > 0)  # Add the total penalty

    return late, float(ytmp[-1])

# Function to generate  permutations of jobs

def generate_initial_permutations(partial_list1, J, initial = ()):
    masque = np.isin(partial_list1, list(initial))
    preliminary_order = partial_list1[~masque][:J]
    initial_permutations = []

    for job in preliminary_order:
        remaining_tasks = [task for task in preliminary_order if task != job]
        permutations = itertools.permutations(remaining_tasks, J - 1)

        initial_permutations.extend([initial + (job,) + perm for perm in permutations])

    return initial_permutations
    
#Find penalties and finish times for all jobs
def generate_permutations_in_stages(partial_list, A, X, D, P, Q):
    result = []
    new_penalties = []
    new_finish = []

    for perm in partial_list:
        penalties, finish_time = eval_perm_fn(A[list(perm)], X[list(perm)], D[list(perm)], P[list(perm)], Q[list(perm)])
        result.append(perm)
        new_penalties.append(penalties)
        new_finish.append(finish_time)
        
    
    return np.array(result), np.array(new_penalties), np.array(new_finish)
    
#pareto optimal solutions  
    
def pruneListFn(tmp_penalties, tmp_finish,tmp_perms):
    late_compare_gt = (tmp_penalties.reshape((1,-1))>tmp_penalties.reshape((-1,1))).astype(int)
    finish_compare_gt = (tmp_finish.reshape((1,-1))>tmp_finish.reshape((-1,1))).astype(int)
    late_compare_ge = (tmp_penalties.reshape((1,-1))>=tmp_penalties.reshape((-1,1))).astype(int)
    finish_compare_ge = (tmp_finish.reshape((1,-1))>=tmp_finish.reshape((-1,1))).astype(int)
    
    index_remove_mx = (late_compare_gt * finish_compare_ge + late_compare_ge * finish_compare_gt) 
    ix = (np.sum(index_remove_mx,axis=0)==0)
    return ix
   
def count_unique_vectors_2d(array):
    # Convertir le tableau 2D en une liste de tuples
    tuples_list = [tuple(row) for row in array]
    
    # Compter les occurrences de chaque tuple
    unique_values, counts = np.unique(tuples_list, return_counts=True, axis=0)
    
    # Créer un dictionnaire avec les vecteurs uniques et leur nombre d'occurrences
    unique_vector_counts = {tuple(val): count for val, count in zip(unique_values, counts)}
    
    # Trier le dictionnaire par valeur (nombre d'occurrences)
    sorted_vector_counts = dict(sorted(unique_vector_counts.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_vector_counts
    
#remove_duplicate
def remove_duplicate(unique_vector_counts):
    # Créer un dictionnaire des paires inversées et des comptes correspondants
    inverse_counts = {tuple(reversed(key)): count for key, count in unique_vector_counts.items()}
    
    # Fusionner les comptes des éléments dupliqués
    unique_vector_counts.update({key: unique_vector_counts.get(key, 0) + count for key, count in inverse_counts.items()})
    
    # Supprimer les paires inversées
    unique_vector_counts = {key: count for key, count in unique_vector_counts.items() if key <= tuple(reversed(key))}
    
    return dict(sorted(unique_vector_counts.items(), key=lambda item: item[1], reverse=True))

# Main function to compute the heuristic
def selectHeurFn(A, D, X, P, Q):
    # Algorithm parameters -- can be adjusted to trade off execution time and performance
    nStartJobs = 7 # Number of initial jobs to consider
    initial_jobs = 2 # Number of jobs to take at start
    max_candidates = 20 
    J = 6  # Number of jobs to evaluate when list of jobs is augmented

    
    
    D = D.reshape(-1)
    A = A.reshape(D.shape)  # Arrival times
    X = X.reshape(D.shape)
    D = D  # Due times
    difference = D
    partial_list1 = np.argsort(difference)
    N = len(difference)  # Number of jobs
    if type(P) != "numpy.ndarray":
        P = P + 0 * A
        Q = Q + 0 * A
    else:
        P = P.reshape(D.shape)
        Q = Q.reshape(Q.shape)
        
    
    
    # I Initialisation

    #1. Generate all nStartJobs initial permutations
    all_permutations = generate_initial_permutations(partial_list1, nStartJobs, initial = ())
    
    #2. Find penalties and finish times for all jobs
    result, penalties, finish = generate_permutations_in_stages(all_permutations, A, X, D, P, Q)

    
    # II Identify pareto optimal solutions     
    ix = pruneListFn(penalties, finish,result)
    pareto_list = result[ix]
    
    #III Iv classification and order the sequences of initial jobs
    pareto_list_init_jobs = pareto_list[:, 0:initial_jobs]
    unique_pareto = count_unique_vectors_2d(pareto_list_init_jobs)
    # V Remove duplicate sequences
    pareto_no_dupl = remove_duplicate(unique_pareto) 
    # VI Initial list
    if len(list(pareto_no_dupl.keys())) > max_candidates : 
        initial_list = list(pareto_no_dupl.keys())[0:max_candidates]
    else : 
        initial_list = list(pareto_no_dupl.keys())
    
    #B Iteration
   
   

    for it in range(N - nStartJobs):
        if printProgress:
            print("Iteration #",it," of ",N-nStartJobs)
    
         #1. generate J! schedules with the next J jobs in sequence
        partial_list = []
        for initial in initial_list:
            
            partial_list += generate_initial_permutations(partial_list1, J, initial)
            
        
            
        if len(partial_list[0]) < N : 
        
            #2. Find penalties and finish times for all jobs
           
            result, penalties, finish = generate_permutations_in_stages(partial_list, A, X, D, P, Q)
          
            #3 Identify pareto optimal solutions     
            ix = pruneListFn(penalties, finish,result)
            pareto_list = result[ix]
            
            #4 5 compute and order the sequences of initial jobs +1 

            pareto_list_jobs = pareto_list[:, 0:initial_jobs + 1]
            unique_pareto = count_unique_vectors_2d(pareto_list_jobs)
            
            # 6 Remove duplicate sequences
            pareto_no_dupl = remove_duplicate(unique_pareto) 
            
            # 7 Initial list
            if len(list(pareto_no_dupl.keys())) > max_candidates : 
                initial_list = list(pareto_no_dupl.keys())[0:max_candidates]
            else : 
                initial_list = list(pareto_no_dupl.keys())
            initial_jobs += 1
        
    result_permutations, penalties, finish = generate_permutations_in_stages(
        partial_list, A, X, D, P, Q)
    ix0 = np.where(penalties == min(penalties))[0]
    finish0 = finish[ix0]
    ix1 = np.where(finish0 == min(finish0))[0][0]
    iSelect = ix0[ix1]
    # 
    if iSelect > len(finish):
        print("Here")
    return result_permutations[iSelect], penalties[iSelect], finish[iSelect]

### For testing
if __name__ == "__main__":
    
        #Flat rate:
    p=50
        #Rate penalty:
    q=1
       
    #### Testing insertHeurLimInsFn:
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
    
    # Example test
    
    order, penalties, finishing = selectHeurFn(As,Ds,Xs,p,q)
