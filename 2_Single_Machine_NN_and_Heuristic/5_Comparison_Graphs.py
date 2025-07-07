# -*- coding: utf-8 -*-
import pickle
import numpy as np
import statistics as st
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import time

# Data to load
datafile1='Output/timeVsNKEEP_50N_Insertion.pkl'
datafile2='Output/timeVsNKEEP_50N_Selection.pkl'

# Number of jobs
N=50

# Load data @@@@@@@@ NEW FILES NEED THE NEXT BLOCK FOR LOADING
"Pairs plots are not updated to handle any set of data."
with open(datafile1,'rb') as file:
    numIns=pickle.load(file)
    KI=pickle.load(file)
    total_runtimesI=pickle.load(file)
    objHeur10I=pickle.load(file)
    objHeur15I=pickle.load(file)
    objHeur20I=pickle.load(file)
    objHeur30I=pickle.load(file)
    objHeur35I=pickle.load(file)
objHeurI=np.zeros([len(KI),len(numIns),100])
objHeurI[:,0,:]=objHeur10I
objHeurI[:,1,:]=objHeur15I
objHeurI[:,2,:]=objHeur20I
objHeurI[:,3,:]=objHeur30I
objHeurI[:,4,:]=objHeur35I

with open(datafile2,'rb') as file:
    numSel=pickle.load(file)
    KS=pickle.load(file) 
    total_runtimesS=pickle.load(file)
    objHeur3S=pickle.load(file)
    objHeur4S=pickle.load(file)
    objHeur5S=pickle.load(file)
    objHeur6S=pickle.load(file)
    objHeur7S=pickle.load(file) 
objHeurS=np.zeros([len(KS),len(numSel),100])
objHeurS[:,0,:]=objHeur3S
objHeurS[:,1,:]=objHeur4S
objHeurS[:,2,:]=objHeur5S
objHeurS[:,3,:]=objHeur6S
objHeurS[:,4,:]=objHeur7S
    
# with open(datafile1,'rb') as file:
#     numIns=pickle.load(file)
#     KI=pickle.load(file)
      # Kept permutations, heuristic parameter, instance
#     total_runtimesI=pickle.load(file)
#     objHeurI=pickle.load(file)

# with open(datafile2,'rb') as file:
#     numSel=pickle.load(file)
#     KS=pickle.load(file)
      # Kept permutations, heuristic parameter, instance
#     total_runtimesS=pickle.load(file)
#     objHeurS=pickle.load(file)

n_test=np.shape(objHeurI)[2]

accuracys_vectorsI = np.zeros([len(KI),len(numIns)])
difference_vectorsI = np.zeros([len(KI),len(numIns),n_test])
max_difference_vectorsI = np.zeros([len(KI),len(numIns)])
mean_difference_vectorsI = np.zeros([len(KI),len(numIns)])
std_difference_vectorsI = np.zeros([len(KI),len(numIns)])

accuracys_vectorsS = np.zeros([len(KS),len(numSel)])
difference_vectorsS = np.zeros([len(KI),len(numSel),n_test])
max_difference_vectorsS = np.zeros([len(KS),len(numSel)])
mean_difference_vectorsS = np.zeros([len(KS),len(numSel)])
std_difference_vectorsS = np.zeros([len(KS),len(numSel)])

objHeurMin=np.zeros(n_test)

# Calculate minimum objective
for i in range(n_test):
    objHeurMin[i]=min(np.min(objHeurI[:,:,i]), np.min(objHeurS[:,:,i]))
    
# In the event the objective mins are all 0, this stops dividing by 0
epsilon = 1e-10

# Generate statistics using the objective minimum
for i in range(len(KI)):
  accuracys_vectorsI[i,:] = np.sum(objHeurMin >= objHeurI[i],axis=1) / n_test*100
  difference_vectorsI[i] = objHeurI[i] - objHeurMin
  max_difference_vectorsI[i] = np.max(difference_vectorsI[i], axis=1)
  mean_difference_vectorsI[i] = np.mean(difference_vectorsI[i], axis=1)
  std_difference_vectorsI[i] = np.std((objHeurI[i]-objHeurMin), axis=1, ddof=1)
  
for i in range(len(KS)):
  accuracys_vectorsS[i,:] = np.sum(objHeurMin>=objHeurS[i],axis=1)/ n_test*100
  difference_vectorsS[i] = objHeurS[i]-objHeurMin
  max_difference_vectorsS[i] = np.max(difference_vectorsS[i], axis=1)
  mean_difference_vectorsS[i] = np.mean(difference_vectorsS[i], axis=1)
  std_difference_vectorsS[i] = np.std((objHeurS[i]-objHeurMin), axis=1, ddof=1)

# Define colors and first set of methods
colors = ['red', 'orange', 'olive', 'lime', 'blue']#,,'purple']
methods = ['Insert in the last 10', 'Insert in the last 15', 'Insert in the last 20', 'Insert in the last 30', 'Insert in the last 35']#'Selection Heuristic',  , 'Insert the last 10', 'Insert the last 40', 'Insert the last 50']

# Calculate needed values for error bar plotting
def calculate_stats(data):
    medians = np.median(data, axis=1)
    percentiles_5 = np.percentile(data, 5, axis=1)
    percentiles_95 = np.percentile(data, 95, axis=1)
    error_bars = np.array([medians - percentiles_5, percentiles_95 - medians])
    means = np.mean(data, axis=1)
    return medians, error_bars, means

###### Plot with error bars: Runtime vs Kept Perms
plt.figure(figsize=(12, 8))

# Calculate statistics for each algorithm
alg_medians = np.zeros([n_test, len(KI)])
error_bars = np.zeros([n_test, 2, len(KI)])
alg_means = np.zeros([n_test, len(KI)])
for i in range(len(numIns)):
    alg_medians[i], error_bars[i], alg_means[i] = calculate_stats(total_runtimesI[i])
    # Plotting for each algorithm
    plt.errorbar(KI+(i*0.6), alg_medians[i], yerr=error_bars[i], color=colors[i], fmt='-s', marker='X', capsize=5, label=methods[i])
    plt.scatter(KI+(i*0.6), alg_means[i], color=colors[i], marker='x')

# Customizing the plot
plt.xlabel('Kept Permutations')
plt.ylabel('Time (Seconds)')
plt.title('Average Runtimes vs Number of Kept Permutations for '+str(n_test)+' instances of '+str(N)+' jobs with Error Bars (5th to 95th Percentile)')
plt.legend()
#plt.yscale('log')
plt.show()


###### Plot with error bars: Obj Difference vs Kept Perms
plt.figure(figsize=(12, 8))

alg_medians = np.zeros([len(total_runtimesI), len(KI)])
error_bars = np.zeros([len(total_runtimesI), 2, len(KI)])
alg_means = np.zeros([len(total_runtimesI), len(KI)])
for i in range(len(total_runtimesI)):
    alg_medians[i], error_bars[i], alg_means[i] = calculate_stats(difference_vectorsI[:,i])

    #plt.plot(Ks, alg_means[i], color=colors[i], label=methods[i])
    # Plotting for each algorithm
    plt.errorbar(KI+(i*0.6), alg_medians[i], yerr=error_bars[i], color=colors[i], fmt='-s', marker='X', capsize=5, label=methods[i])
    plt.scatter(KI+(i*0.6), alg_means[i], color=colors[i], marker='x')

# Customizing the plot
plt.xlabel('Kept Permutations')
plt.ylabel('Objective Difference from Best')
plt.title('Average Difference vs Number of Kept Permutations for '+str(n_test)+' instances of '+str(N)+' jobs with Error Bars (5th to 95th Percentile)')
plt.legend()
#plt.yscale('log')
plt.show()


###### Scatter Plot: Obj Difference vs Kept Permutations
plt.figure(figsize=(12, 8))

for i in range(len(total_runtimesI)):
    # Plotting for each algorithm
    plt.scatter((np.ones([100,1])*np.array(KI).reshape([1,-1])).T.flatten()+(i*0.6), difference_vectorsI[:,i].flatten(), color=colors[i], label=methods[i])

# Customizing the plot
plt.xlabel('Kept Permutations')
plt.ylabel('Objective Difference from Best')
plt.title('Difference vs Number of Kept Permutations for '+str(n_test)+' instances of '+str(N)+' jobs')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=5)
#plt.yscale('log')
plt.show()

############## Create the dataframe for comparison
arrays = [objHeur10I.T, objHeur15I.T, objHeur20I.T, objHeur30I.T, objHeur35I.T]
data_2d = np.hstack(arrays)
column_titles = []
for y in range(5):
    for x in range(6):
        column_titles.append(f'Insert in the last {numIns[y]} \n with {KI[x]} kept permutations')

dfI = pd.DataFrame(data_2d, columns=column_titles)




############################ Selection Graphs
methods = ['Select from 3', 'Select from 4', 'Select from 5', 'Select from 6', 'Select from 7']
###### Plot with error bars: Runtime vs Kept Perms
plt.figure(figsize=(12, 8))

# Calculate statistics for each algorithm
alg_medians = np.zeros([len(total_runtimesS), len(KS)])
error_bars = np.zeros([len(total_runtimesS), 2, len(KS)])
alg_means = np.zeros([len(total_runtimesS), len(KS)])
for i in range(len(total_runtimesS)):
    alg_medians[i], error_bars[i], alg_means[i] = calculate_stats(total_runtimesS[i])

    #plt.plot(Ks, alg_means[i], color=colors[i], label=methods[i])
    # Plotting for each algorithm
    plt.errorbar(KS+(i*0.6), alg_medians[i], yerr=error_bars[i], color=colors[i], fmt='-s', marker='X', capsize=5, label=methods[i])
    plt.scatter(KS+(i*0.6), alg_means[i], color=colors[i], marker='x')

# Customizing the plot
plt.xlabel('Kept Permutations')
plt.ylabel('Time (Seconds)')
plt.title('Average Runtimes vs Number of Kept Permutations for '+str(n_test)+' instances of '+str(N)+' jobs with Error Bars (5th to 95th Percentile)')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=5)
plt.yscale('log')
plt.show()


###### Plot with error bars: Obj Difference vs Kept Perms
plt.figure(figsize=(12, 8))

alg_medians = np.zeros([len(total_runtimesS), len(KS)])
error_bars = np.zeros([len(total_runtimesS), 2, len(KS)])
alg_means = np.zeros([len(total_runtimesS), len(KS)])
for i in range(len(total_runtimesS)):
    alg_medians[i], error_bars[i], alg_means[i] = calculate_stats(difference_vectorsS[:,i])

    #plt.plot(Ks, alg_means[i], color=colors[i], label=methods[i])
    # Plotting for each algorithm
    plt.errorbar(KS+(i*0.6), alg_medians[i], yerr=error_bars[i], color=colors[i], fmt='-s', marker='X', capsize=5, label=methods[i])
    plt.scatter(KS+(i*0.6), alg_means[i], color=colors[i], marker='x')

# Customizing the plot
plt.xlabel('Kept Permutations')
plt.ylabel('Objective Difference from Best')
plt.title('Average Difference vs Number of Kept Permutations for '+str(n_test)+' instances of '+str(N)+' jobs with Error Bars (5th to 95th Percentile)')
plt.legend()
#plt.yscale('log')
plt.show()


###### Scatter Plot: Obj Difference vs Kept Permutations
plt.figure(figsize=(12, 8))

for i in range(len(total_runtimesS)):
    # Plotting for each algorithm
    plt.scatter((np.ones([100,1])*np.array(KS).reshape([1,-1])).T.flatten()+(i*0.6), difference_vectorsS[:,i].flatten(), color=colors[i], label=methods[i])

# Customizing the plot
plt.xlabel('Kept Permutations')
plt.ylabel('Objective Difference from Best')
plt.title('Difference vs Number of Kept Permutations for '+str(n_test)+' instances of '+str(N)+' jobs')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=5)
#plt.yscale('log')
plt.show()

############## Create the dataframe for comparison
arrays = [objHeur3S.T, objHeur4S.T, objHeur5S.T, objHeur6S.T, objHeur7S.T]
data_2d = np.hstack(arrays)
column_titles = []
for y in range(5):
    for x in range(6):
        column_titles.append(f'Select from {numSel[y]} \n with {KS[x]} kept permutations')

dfS = pd.DataFrame(data_2d, columns=column_titles)




##################### Objective Pairs
#### Extract columns for comparison: Insert 15, 20, 30 and Select 5, 6, 7 with 50 kept permutations
# Extract columns from dfI
cols_dfI = [8, 14, 20]  # 9th, 15th, and 21st columns (0-based index)
dfI_selected = dfI.iloc[:, cols_dfI]

# Extract columns from dfS
cols_dfS = [14, 20, 26]  # 15th, 21st, and 27th columns (0-based index)
dfS_selected = dfS.iloc[:, cols_dfS]

# Combine the selected columns from both DataFrames
df_combined = pd.concat([dfI_selected, dfS_selected], axis=1)

# Create the pairplot
pairplot = sns.pairplot(df_combined)

# Add y=x line to all plots except the diagonal
n = len(pairplot.axes)
for i in range(n):
    for j in range(n):
        if i != j:  # Skip diagonal
            ax = pairplot.axes[i, j]
            
            # Get the current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Define the limits for the y=x line within the current axis limits
            min_lim = max(min(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
            max_lim = min(max(xlim[1], ylim[1]), max(xlim[0], ylim[0]))
            
            # Plot the y=x line
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')  # 'k--' is black dotted line
            
            # Re-adjust the axis limits to ensure the line fits without affecting the plot's zoom level
            ax.set_xlim(xlim)  # Restore original x limits
            ax.set_ylim(ylim)  # Restore original y limits
            
# Adjust the layout to ensure everything fits
plt.tight_layout()     

# Show the plot    
plt.show()

#### Extract columns for comparison: Insert 15 and Select 5 with 50, 60, 70 kept permutations
# Extract columns from dfI
cols_dfI = [8, 9, 10]  # 9th, 15th, and 21st columns (0-based index)
dfI_selected = dfI.iloc[:, cols_dfI]

# Extract columns from dfS
cols_dfS = [14, 15, 16]  # 15th, 21st, and 27th columns (0-based index)
dfS_selected = dfS.iloc[:, cols_dfS]

# Combine the selected columns from both DataFrames
df_combined = pd.concat([dfI_selected, dfS_selected], axis=1)

# Create the pairplot
pairplot = sns.pairplot(df_combined)

# Add y=x line to all plots except the diagonal
n = len(pairplot.axes)
for i in range(n):
    for j in range(n):
        if i != j:  # Skip diagonal
            ax = pairplot.axes[i, j]
            
            # Get the current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Define the limits for the y=x line within the current axis limits
            min_lim = max(min(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
            max_lim = min(max(xlim[1], ylim[1]), max(xlim[0], ylim[0]))
            
            # Plot the y=x line
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')  # 'k--' is black dotted line
            
            # Re-adjust the axis limits to ensure the line fits without affecting the plot's zoom level
            ax.set_xlim(xlim)  # Restore original x limits
            ax.set_ylim(ylim)  # Restore original y limits

# Adjust the layout to ensure everything fits
plt.tight_layout()

# Show the plot
plt.show()

#### Extract columns for comparison: Insert 15 and Select 7 with 50, 60, 70 kept permutations
# Extract columns from dfI
cols_dfI = [8, 9, 10]  # 9th, 15th, and 21st columns (0-based index)
dfI_selected = dfI.iloc[:, cols_dfI]

# Extract columns from dfS
cols_dfS = [26, 27, 28]  # 15th, 21st, and 27th columns (0-based index)
dfS_selected = dfS.iloc[:, cols_dfS]

# Combine the selected columns from both DataFrames
df_combined = pd.concat([dfI_selected, dfS_selected], axis=1)

# Create the pairplot
pairplot = sns.pairplot(df_combined)

# Add y=x line to all plots except the diagonal
n = len(pairplot.axes)
for i in range(n):
    for j in range(n):
        if i != j:  # Skip diagonal
            ax = pairplot.axes[i, j]
            
            # Get the current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Define the limits for the y=x line within the current axis limits
            min_lim = max(min(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
            max_lim = min(max(xlim[1], ylim[1]), max(xlim[0], ylim[0]))
            
            # Plot the y=x line
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')  # 'k--' is black dotted line
            
            # Re-adjust the axis limits to ensure the line fits without affecting the plot's zoom level
            ax.set_xlim(xlim)  # Restore original x limits
            ax.set_ylim(ylim)  # Restore original y limits
            
# Adjust the layout to ensure everything fits
plt.tight_layout()     

# Show the plot    
plt.show()




##################### Objective-Best Pairs
############## Create the dataframe for comparison
arrays = [(objHeur10I-objHeurMin).T, (objHeur15I-objHeurMin).T, (objHeur20I-objHeurMin).T, (objHeur30I-objHeurMin).T, (objHeur35I-objHeurMin).T]
data_2d = np.hstack(arrays)
column_titles = []
for y in range(5):
    for x in range(6):
        column_titles.append(f'Insert in the last {numIns[y]} \n with {KI[x]} kept permutations \n minus best objective')

dfI = pd.DataFrame(data_2d, columns=column_titles)

arrays = [(objHeur3S-objHeurMin).T, (objHeur4S-objHeurMin).T, (objHeur5S-objHeurMin).T, (objHeur6S-objHeurMin).T, (objHeur7S-objHeurMin).T]
data_2d = np.hstack(arrays)
column_titles = []
for y in range(5):
    for x in range(6):
        column_titles.append(f'Select from {numSel[y]} \n with {KS[x]} kept permutations \n minus best objective')

dfS = pd.DataFrame(data_2d, columns=column_titles)

#### Extract columns for comparison: Insert 15, 20, 30 and Select 5, 6, 7 with 50 kept permutations
# Extract columns from dfI
cols_dfI = [8, 14, 20]  # 9th, 15th, and 21st columns (0-based index)
dfI_selected = dfI.iloc[:, cols_dfI]

# Extract columns from dfS
cols_dfS = [14, 20, 26]  # 15th, 21st, and 27th columns (0-based index)
dfS_selected = dfS.iloc[:, cols_dfS]

# Combine the selected columns from both DataFrames
df_combined = pd.concat([dfI_selected, dfS_selected], axis=1)

# Create the pairplot
pairplot = sns.pairplot(df_combined, plot_kws={'color': 'red'})
# Customizing diagonal histograms
for ax in pairplot.diag_axes:
    for patch in ax.patches:
        patch.set_facecolor('red')  # Change histogram bars' fill color to red
        patch.set_edgecolor('black')  # Set the outline color to black

# Add y=x line to all plots except the diagonal
n = len(pairplot.axes)
for i in range(n):
    for j in range(n):
        if i != j:  # Skip diagonal
            ax = pairplot.axes[i, j]
            
            # Get the current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Define the limits for the y=x line within the current axis limits
            min_lim = max(min(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
            max_lim = min(max(xlim[1], ylim[1]), max(xlim[0], ylim[0]))
            
            # Plot the y=x line
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')  # 'k--' is black dotted line
            
            # Re-adjust the axis limits to ensure the line fits without affecting the plot's zoom level
            ax.set_xlim(xlim)  # Restore original x limits
            ax.set_ylim(ylim)  # Restore original y limits
            
# Adjust the layout to ensure everything fits
plt.tight_layout()     

# Show the plot    
plt.show()

#### Extract columns for comparison: Insert 15 and Select 5 with 50, 60, 70 kept permutations
# Extract columns from dfI
cols_dfI = [8, 9, 10]  # 9th, 15th, and 21st columns (0-based index)
dfI_selected = dfI.iloc[:, cols_dfI]

# Extract columns from dfS
cols_dfS = [14, 15, 16]  # 15th, 21st, and 27th columns (0-based index)
dfS_selected = dfS.iloc[:, cols_dfS]

# Combine the selected columns from both DataFrames
df_combined = pd.concat([dfI_selected, dfS_selected], axis=1)

# Create the pairplot
pairplot = sns.pairplot(df_combined, plot_kws={'color': 'red'})
# Customizing diagonal histograms
for ax in pairplot.diag_axes:
    for patch in ax.patches:
        patch.set_facecolor('red')  # Change histogram bars' fill color to red
        patch.set_edgecolor('black')  # Set the outline color to black
        
# Add y=x line to all plots except the diagonal
n = len(pairplot.axes)
for i in range(n):
    for j in range(n):
        if i != j:  # Skip diagonal
            ax = pairplot.axes[i, j]
            
            # Get the current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Define the limits for the y=x line within the current axis limits
            min_lim = max(min(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
            max_lim = min(max(xlim[1], ylim[1]), max(xlim[0], ylim[0]))
            
            # Plot the y=x line
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')  # 'k--' is black dotted line
            
            # Re-adjust the axis limits to ensure the line fits without affecting the plot's zoom level
            ax.set_xlim(xlim)  # Restore original x limits
            ax.set_ylim(ylim)  # Restore original y limits

# Adjust the layout to ensure everything fits
plt.tight_layout()

# Show the plot
plt.show()

#### Extract columns for comparison: Insert 15 and Select 7 with 50, 60, 70 kept permutations
# Extract columns from dfI
cols_dfI = [8, 9, 10]  # 9th, 15th, and 21st columns (0-based index)
dfI_selected = dfI.iloc[:, cols_dfI]

# Extract columns from dfS
cols_dfS = [26, 27, 28]  # 15th, 21st, and 27th columns (0-based index)
dfS_selected = dfS.iloc[:, cols_dfS]

# Combine the selected columns from both DataFrames
df_combined = pd.concat([dfI_selected, dfS_selected], axis=1)

# Create the pairplot
pairplot = sns.pairplot(df_combined, plot_kws={'color': 'red'})
# Customizing diagonal histograms
for ax in pairplot.diag_axes:
    for patch in ax.patches:
        patch.set_facecolor('red')  # Change histogram bars' fill color to red
        patch.set_edgecolor('black')  # Set the outline color to black
        
# Add y=x line to all plots except the diagonal
n = len(pairplot.axes)
for i in range(n):
    for j in range(n):
        if i != j:  # Skip diagonal
            ax = pairplot.axes[i, j]
            
            # Get the current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Define the limits for the y=x line within the current axis limits
            min_lim = max(min(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
            max_lim = min(max(xlim[1], ylim[1]), max(xlim[0], ylim[0]))
            
            # Plot the y=x line
            ax.plot([min_lim, max_lim], [min_lim, max_lim], 'k--')  # 'k--' is black dotted line
            
            # Re-adjust the axis limits to ensure the line fits without affecting the plot's zoom level
            ax.set_xlim(xlim)  # Restore original x limits
            ax.set_ylim(ylim)  # Restore original y limits
            
# Adjust the layout to ensure everything fits
plt.tight_layout()     

# Show the plot    
plt.show()



####### Proportion Graph
# Calculate the proportions
def calculate_proportion(heuristics, min_values):
    proportions = []
    for row in heuristics:
        proportion = np.mean(row == min_values)
        proportions.append(proportion)
    return proportions

proportions_10 = calculate_proportion(objHeur10I, objHeurMin)
proportions_15 = calculate_proportion(objHeur15I, objHeurMin)
proportions_20 = calculate_proportion(objHeur20I, objHeurMin)
proportions_30 = calculate_proportion(objHeur30I, objHeurMin)
proportions_35 = calculate_proportion(objHeur35I, objHeurMin)

# Organize data for plotting
proportions_all = [proportions_10, proportions_15, proportions_20, proportions_30, proportions_35]

methods = ['Insert in the last 10', 'Insert in the last 15', 'Insert in the last 20', 'Insert in the last 30', 'Insert in the last 35']

# Plot the data
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.15
index = np.arange(len(KI))

# Adjusted the bar positions for each permutation set
for i, proportions in enumerate(proportions_all):
    ax.bar(index + i * bar_width, proportions, bar_width, label=methods[i])

# Set the y-axis range
ax.set_ylim(0, 0.5)

# Labeling the plot
ax.set_xlabel('Number of Kept Permutations')
ax.set_ylabel('Proportion Matching objHeurMin')
ax.set_title('Proportion of Matches with Minimum Objective Value')
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(KI)
ax.legend()

# Show the plot
plt.show()



proportions_3 = calculate_proportion(objHeur3S, objHeurMin)
proportions_4 = calculate_proportion(objHeur4S, objHeurMin)
proportions_5 = calculate_proportion(objHeur5S, objHeurMin)
proportions_6 = calculate_proportion(objHeur6S, objHeurMin)
proportions_7 = calculate_proportion(objHeur7S, objHeurMin)

# Organize data for plotting
proportions_all = [proportions_3, proportions_4, proportions_5, proportions_6, proportions_7]

methods = ['Select from 3', 'Select from 4', 'Select from 5', 'Select from 6', 'Select from 7']

# Plot the data
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.15
index = np.arange(len(KI))

# Adjusted the bar positions for each permutation set
for i, proportions in enumerate(proportions_all):
    ax.bar(index + i * bar_width, proportions, bar_width, label=methods[i])

# Set the y-axis range
ax.set_ylim(0, 0.5)

# Labeling the plot
ax.set_xlabel('Number of Kept Permutations')
ax.set_ylabel('Proportion Matching objHeurMin')
ax.set_title('Proportion of Matches with Minimum Objective Value')
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(KI)
ax.legend()

# Show the plot
plt.show()
