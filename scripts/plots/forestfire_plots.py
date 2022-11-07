# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:45:23 2022
In this file we generate all the tables, plots and other numerical results
of the experiments with the forest fires data set by Cortez et al. (2007).
These results were generated with the file forest_experiments.py and presented in the Results section. 
Cortez, P., & Morais, A. D. J. R. (2007). A data mining approach to predict forest fires using meteorological data. 
We vary epsilon and gamma and run the centralized regression.
@author: kroessks
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
# load data
file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd+"/..")
import generate_data_forestfires as forest
os.chdir(file_wd+"/../experiments")
from simulation_functions import R2, lin_reg
os.chdir(file_wd+"/../../data")
X, y, X_splitted = forest.get_data()
# load the data
df = pd.read_csv("forestfires.csv")
X_A = (
    pd.read_csv( "forest_fire_Alice.csv")
    .drop(columns=["Unnamed: 0"])
)
X_B = (
    pd.read_csv( "forest_fire_Bob.csv")
    .drop(columns=["Unnamed: 0"])
)
col_names= np.concatenate([X_A.columns, X_B.columns])
np.savetxt("X_forest.csv", X, delimiter=",")
np.savetxt("y_forest.csv", y, delimiter=",")

''' settings for boxplots '''
flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                  linestyle='none', markeredgecolor='none')

'''explore the data set'''
beta_centralized = lin_reg(X, y)
# - the correlations : these are relatively low
cors_forest= np.corrcoef(X.T)
np.fill_diagonal(cors_forest,0)
plt.hist(cors_forest.reshape(-1))
np.mean(abs(cors_forest)) # 0.08141255542743739
np.max(abs(cors_forest)) # 0.6821916119833173
np.median(abs(cors_forest)) # 0.08141255542743739

# - get true R2
# lets get the true R2 (no differential privacy)
y_hat_nodp= np.sum(beta_centralized*X,1)
R2_no_dp= np.var(y_hat_nodp)/np.var(y)
# 0.07708899860500595 so this is quite low compared to the experiments.
# so that means that for these two parties, for five iterations, the bound is equal to:
R2_dp_bound= 1- 1.5**(2*5*2)*(1-R2_no_dp)
# -3067.916018653254

'''load results'''
# - experimental results for varying epsilon and gamma
os.chdir(file_wd+"/../../results")
with open("betas_epsilon_forest", "rb") as input_file:
    betas_epsilon=pickle.load(input_file)
# with open("betas_gamma_forest", "rb") as input_file:
#     betas_gamma=pickle.load(input_file)
epsilons= np.array([0.1,0.5,1,1.5,2.5,5,10])
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])

##############################################################################################
''' generate plots '''
# Generate Figure 8. 
# (reproduce Figure 5 by van Kesteren et al. (2019)
ticks= col_names
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 8)
plt.xlim(xmin=-.5, xmax=27)
# plt.ylim(ymin=np.percentile(betas_epsilon[:,4,:],.02), ymax=np.percentile(betas_epsilon[:,4,:], 99.99))
plt.ylim(ymin=-10, ymax=10)
x = [22.7,27]
y1 = [10,10]
plt.fill_between(x, y1, -10,
                 facecolor="grey", # The fill color
                 alpha=0.2)          # Transparency of the fill
for i in range(27):
    median20= np.median(betas_epsilon.T[i][-2])
    median2= np.median(betas_epsilon.T[i][4])
    plt.errorbar(y=np.median(betas_epsilon.T[i][-2]), x= i+.2, yerr=np.array([[median20-np.percentile(betas_epsilon.T[i][-2], 2.5),np.percentile(betas_epsilon.T[i][-2], 97.5)-median20]]).T, solid_capstyle='projecting', capsize=2, color="green", label="$\epsilon$=10")
    plt.errorbar(y=np.median(betas_epsilon.T[i][4]), x= i+.4, yerr=np.array([[median2-np.percentile(betas_epsilon.T[i][4], 2.5),np.percentile(betas_epsilon.T[i][4], 97.5)-median2]]).T, solid_capstyle='projecting', capsize=2, color="orange", label="$\epsilon$=1")
    plt.errorbar(y=beta_centralized[i], x=i, yerr=CIs_arr.T[0][i], solid_capstyle='projecting', capsize=2, color="black")
    plt.scatter(y=np.median(betas_epsilon.T[i][-2]), x= i+.2, color="green", label="$\epsilon$=10")
    plt.scatter(y=np.median(betas_epsilon.T[i][4]), x= i+.4, color="orange", label="$\epsilon$=1")
    plt.scatter(y=beta_centralized[i], x=i, color="black")
plt.xticks(range(0,len(ticks)), ticks, rotation='vertical')
np.mean(betas_epsilon,(0))

############################################################################
# Generate Figure 9.

reps=500
R2_epsilon= np.zeros((epsilons.shape[0], reps))
for i in range(epsilons.shape[0]):
    for j in range(reps):
        betas_epsilon_i= betas_epsilon[j][i]
        R2_epsilon[i][j]= R2(y,np.dot(betas_epsilon_i, X.T)) # 0.0723196328281206

epsilons_nonzero= epsilons[1:]
pos= np.arange(1,len(epsilons))

# plt.title('$R^2$ over epsilon for gamma = 1.5, \n and delta = 1 with synthetic data (500 repetitions, centralized $R^2$=0.3)')
plt.scatter(np.repeat(0, reps), R2_epsilon[0], alpha=.04, label='no DP')
plt.boxplot( R2_epsilon[0], widths=0.1, positions=[0], medianprops=dict(color='black'), flierprops=flierprops)
for e, eps in enumerate(epsilons_nonzero):
    plt.scatter(np.repeat(pos[e]+.2, reps), R2_epsilon[e+1], alpha=.055, label= "$\epsilon$ = "+str(int(eps*2))+' DP')
    plt.boxplot( R2_epsilon[e+1], widths=0.1, positions=[pos[e]+.2], medianprops=dict(color='black'), flierprops=flierprops)
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(.165, -.37, 0.6,0.4), loc='lower center', ncol=3)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r'$\epsilon$')
plt.ylabel('$R^2$')
plt.axhline(y=R2_epsilon[0][0], alpha=.8, linewidth=.5, color="black", ls='-.')
plt.ylim([-.5,.1])

np.median(R2_epsilon[1])# -4.070205089837887
np.median(R2_epsilon[2])# -0.938228174142436


