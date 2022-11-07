# -*- coding: utf-8 -*-
"""
In this file we generate all the tables, plots and other numerical results
of the experiments with the productivity data set by
1. Imran, A. A., Amin, M. N., Islam Rifat, M. R., & Mehreen, S. (2019). Deep Neural Network Approach for Predicting the Productivity of Garment Employees. 2019 6th International Conference on Control, Decision and Information Technologies (CoDIT). [Web Link]
2. Rahim, M. S., Imran, A. A., & Ahmed, T. (2021). Mining the Productivity Data of Garment Industry. International Journal of Business Intelligence and Data Mining, 1(1), 1.

These results were generated with the file productivity_experiments.py and presented in the Results section. 
We vary epsilon and gamma and run the centralized regression.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import copy

file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd+"/../experiments")
from simulation_functions import R2, lin_reg
os.chdir(file_wd+"/../../data")
prod = pd.read_csv( r"garments_worker_productivity.csv")
prod_processed = copy.deepcopy(prod)
prod_processed = prod_processed.drop(columns=['wip'])
prod_processed =  pd.get_dummies(prod_processed, drop_first=True, columns = ['department', 'day'])
prod_processed.quarter = prod_processed.quarter.factorize()[0]
prod_processed = prod_processed.drop(columns=['date'])
cols1 = ['quarter', 'team', 'targeted_productivity', 'smv', 'over_time',
       'incentive','day_Saturday', 'day_Sunday', 'day_Thursday',
       'day_Tuesday', 'day_Wednesday']
              
cols2 =['idle_time', 'idle_men', 'no_of_style_change',
       'no_of_workers', 'department_finishing ',
       'department_sweing']
X_splitted = []
X_splitted.append(np.array(prod_processed[cols1]))
X_splitted.append(np.array(prod_processed[cols2]))
y = np.array(prod_processed['actual_productivity'])
X = np.array(prod_processed[cols1+cols2])


'''explore the data set'''
beta_centralized= lin_reg(X, y)

# - get true R2
# lets get the true R2 (no differential privacy)
y_hat_nodp= np.sum(beta_centralized*X,1)
R2_no_dp= np.var(y_hat_nodp)/np.var(y)
#0.41432646008631663
R2_dp_bound= 1- 1.5**(2*5*2)*(1-R2_no_dp)
# -1946.514880227549
'''load results'''
# - experimental results for varying epsilon and gamma
os.chdir(file_wd+"/../../results")
with open("betas_epsilon_prod", "rb") as input_file:
    betas_epsilon=pickle.load(input_file)

epsilons= np.array([0.1,0.5,1,1.5,2.5,5,10])
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])

''' generate plots '''
# - epsilon: beta R**2 
# extract R2 values.
reps=betas_epsilon.shape[0]
R2_epsilon= np.zeros((epsilons.shape[0], reps))
for i in range(epsilons.shape[0]):
    for j in range(reps):
        betas_epsilon_i= betas_epsilon[j][i]
        R2_epsilon[i][j]= R2(y,np.dot(betas_epsilon_i, X.T)) # 0.0723196328281206

##############################################################################################
epsilons_nonzero= epsilons[1:]
pos= np.arange(1,len(epsilons))
flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                  linestyle='none', markeredgecolor='none')

# we take a selection of the epsilons tested
epsilons_nonzero_sel = epsilons[[0,1,3,5,7,8]]
R2_epsilon_sel = R2_epsilon[[0,1,3,5,7,8]]
epsilons_sel = epsilons_nonzero[[0,2,4,6,7]]
R2_epsilon_sel = R2_epsilon[[1,3,5,7,8]]

# Figure 11: R2 for garment industry data
plt.scatter(np.repeat(0, reps), R2_epsilon[0], alpha=.04, label='no DP')
plt.boxplot( R2_epsilon[0], widths=0.1, positions=[0], medianprops=dict(color='black'), flierprops=flierprops)
for e, eps in enumerate(epsilons_sel):
    plt.scatter(np.repeat(pos[e]+.2, reps), R2_epsilon_sel[e], alpha=.055, label= "$\epsilon$ = "+str(eps*2)+' DP')
    plt.boxplot( R2_epsilon_sel[e], widths=0.1, positions=[pos[e]+.2], medianprops=dict(color='black'), flierprops=flierprops)
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(.165, -.37, 0.6,0.4), loc='lower center', ncol=3)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r'$\epsilon$')
plt.ylabel('$R^2$')
plt.axhline(y=R2_epsilon[0][0], alpha=.8, linewidth=.5, color="black", ls='-.')
plt.ylim([-.5,.3])

##############################################################################################
# Figure 10: beta coefficients for garment industry data
col_names = cols1+cols2
ticks= col_names
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 8)
plt.xlim(xmin=-.5, xmax=X.shape[1])
# plt.ylim(ymin=np.percentile(betas_epsilon[:,4,:],.02), ymax=np.percentile(betas_epsilon[:,4,:], 99.99))
plt.ylim(ymin=-1.5, ymax=1.5)
x = [len(cols1)-.3,X.shape[1]]
y1 = [31,31]
plt.fill_between(x, y1, -31,
                 facecolor="grey", # The fill color
                 alpha=0.2)          # Transparency of the fill
for i in range(X.shape[1]):
    median20= np.median(betas_epsilon.T[i][-2])
    median2= np.median(betas_epsilon.T[i][3])
    plt.errorbar(y=np.median(betas_epsilon.T[i][-2]), x= i+.2, yerr=np.array([[median20-np.percentile(betas_epsilon.T[i][-2], 2.5),np.percentile(betas_epsilon.T[i][-2], 97.5)-median20]]).T, solid_capstyle='projecting', capsize=2, color="green", label="$\epsilon$=10")
    plt.errorbar(y=np.median(betas_epsilon.T[i][3]), x= i+.4, yerr=np.array([[median2-np.percentile(betas_epsilon.T[i][3], 2.5),np.percentile(betas_epsilon.T[i][3], 97.5)-median2]]).T, solid_capstyle='projecting', capsize=2, color="orange", label="$\epsilon$=1")
    plt.scatter(y=np.median(betas_epsilon.T[i][-2]), x= i+.2, color="green", label="$\epsilon$=10")
    plt.scatter(y=np.median(betas_epsilon.T[i][3]), x= i+.4, color="orange", label="$\epsilon$=1")
    plt.scatter(y=beta_centralized[i], x=i, color="black")
plt.xticks(range(0,len(ticks)), ticks, rotation='vertical')

