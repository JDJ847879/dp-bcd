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
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/scripts")
import generate_data_forestfires as forest
from simulation_functions import R2, lin_reg
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/data")
# load data
df = pd.read_csv("forestfires.csv")
X_A = (
    pd.read_csv( "data_csv/forest_fire_Alice.csv")
    .drop(columns=["Unnamed: 0"])
)
X_B = (
    pd.read_csv( "data_csv/forest_fire_Bob.csv")
    .drop(columns=["Unnamed: 0"])
)
col_names= np.concatenate([X_A.columns, X_B.columns])
X, y, X_splitted = forest.get_data()

''' settings for boxplots '''
flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                  linestyle='none', markeredgecolor='none')

'''explore the data set'''
beta_centralized= lin_reg(X, y)
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
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/data")
# load results generated with forest_experiments.py and "confidence intervals forest data.R":
# - confidence intervals
CIs= pd.read_csv( "ci_forest.csv" )
CIs= CIs.drop( CIs.columns[0], axis=1)
beta_centralized= lin_reg(X, y)
CIs_arr=  np.array(CIs)
CIs['betas']= beta_centralized
CIs=CIs[[CIs.columns[0], CIs.columns[2], CIs.columns[1]]]
centralized_arr= np.array(CIs)

# - experimental results for varying epsilon and gamma
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/Synthetic data results")
with open("betas_epsilon_forest", "rb") as input_file:
    betas_epsilon=pickle.load(input_file)
# with open("betas_gamma_forest", "rb") as input_file:
#     betas_gamma=pickle.load(input_file)
epsilons= np.array([0.1,0.5,1,1.5,2.5,5,10])
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])

def R2(y, y_hat):
    """
    Compute the R2 score of the labels y and predicted values y_hat.
    """
    if len(y) != len(y_hat):
        print("R2 dimension error", y, y_hat)
        return None
    y_avg = 0.0
    for i in range(0, len(y)):
        y_avg += y[i]
    y_avg *= 1 / len(y)
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(0, len(y)):
        ss_res += (y[i] - y_hat[i]) ** 2
        ss_tot += (y[i] - y_avg) ** 2
    r2 = 1.0 - ss_res / ss_tot
    return r2
##############################################################################################
''' generate plots '''
# - epsilon: beta parameters (reproduce Figure 5 by van Kesteren et al. (2019))
ticks= col_names
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 8)
plt.xlim(xmin=-.5, xmax=27)
# plt.ylim(ymin=np.percentile(betas_epsilon[:,4,:],.02), ymax=np.percentile(betas_epsilon[:,4,:], 99.99))
plt.ylim(ymin=-31, ymax=31)
x = [22.7,27]
y1 = [31,31]
plt.fill_between(x, y1, -31,
                 facecolor="grey", # The fill color
                 alpha=0.2)          # Transparency of the fill
for i in range(27):
    plt.errorbar(y=np.mean(betas_epsilon.T[i][-2]), x= i+.2, yerr=np.percentile(betas_epsilon.T[i][-2], .025), solid_capstyle='projecting', capsize=2, color="green", label="$\epsilon$=10")
    plt.errorbar(y=np.mean(betas_epsilon.T[i][4]), x= i+.4, yerr=np.percentile(betas_epsilon.T[i][4], .025), solid_capstyle='projecting', capsize=2, color="orange", label="$\epsilon$=1")
    plt.errorbar(y=beta_centralized[i], x=i, yerr=CIs_arr.T[0][i], solid_capstyle='projecting', capsize=2, color="black")
    plt.scatter(y=np.mean(betas_epsilon.T[i][-2]), x= i+.2, color="green", label="$\epsilon$=10")
    plt.scatter(y=np.mean(betas_epsilon.T[i][4]), x= i+.4, color="orange", label="$\epsilon$=1")
    plt.scatter(y=beta_centralized[i], x=i, color="black")
plt.xticks(range(0,len(ticks)), ticks, rotation='vertical')
# # - epsilon: beta parameters
np.mean(betas_epsilon,(0))
# we had 500 repetitions which is a sizable amount

# we change it so that the SE's are no longer depicted

# - epsilon: beta parameters (reproduce Figure 5 by van Kesteren et al. (2019), without their data)
ticks= col_names
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 8)
plt.xlim(xmin=-.5, xmax=27)
# plt.ylim(ymin=np.percentile(betas_epsilon[:,4,:],.02), ymax=np.percentile(betas_epsilon[:,4,:], 99.99))
plt.ylim(ymin=-31, ymax=31)
x = [22.7,27]
y1 = [31,31]
plt.fill_between(x, y1, -31,
                 facecolor="grey", # The fill color
                 alpha=0.2)          # Transparency of the fill
for i in range(27):
    plt.errorbar(y=np.mean(betas_epsilon.T[i][-2]), x= i+.2, yerr=np.percentile(betas_epsilon.T[i][-2], .025), solid_capstyle='projecting', capsize=2, color="green", label="$\epsilon$=10")
    plt.errorbar(y=np.mean(betas_epsilon.T[i][4]), x= i+.4, yerr=np.percentile(betas_epsilon.T[i][4], .025), solid_capstyle='projecting', capsize=2, color="orange", label="$\epsilon$=1")
    # plt.errorbar(y=beta_centralized[i], x=i, yerr=CIs_arr.T[0][i], solid_capstyle='projecting', capsize=2, color="black")
    plt.scatter(y=np.mean(betas_epsilon.T[i][-2]), x= i+.2, color="green", label="$\epsilon$=10")
    plt.scatter(y=np.mean(betas_epsilon.T[i][4]), x= i+.4, color="orange", label="$\epsilon$=1")
    plt.scatter(y=beta_centralized[i], x=i, color="black")
plt.xticks(range(0,len(ticks)), ticks, rotation='vertical')
# # - epsilon: beta parameters
np.mean(betas_epsilon,(0))
##############################################################################################
''' generate plots '''
# - epsilon: beta parameters (reproduce Figure 5 by van Kesteren et al. (2019))
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
# # - epsilon: beta parameters
np.mean(betas_epsilon,(0))
# we had 500 repetitions which is a sizable amount
mins1=np.zeros((27))
maxs1=np.zeros((27))
mins10=np.zeros((27))
maxs10=np.zeros((27))
vars10=np.zeros((27))
vars1=np.zeros((27))

for i in range(27):
    mins1[i]= np.min(betas_epsilon.T[i][4])
    maxs1[i]= np.max(betas_epsilon.T[i][4])

    mins10[i]= np.min(betas_epsilon.T[i][-2])
    maxs10[i]= np.max(betas_epsilon.T[i][-2])
    
    vars10[i]= np.var(betas_epsilon.T[i][-2])
    vars1[i]= np.var(betas_epsilon.T[i][4])

mins1  -mins10  
maxs1  -maxs10  

range1= maxs1-mins1
range10= maxs10-mins10
range1-range10

range10[9]
range1[9]
mins1[9]
maxs1[9]

mins10[9]
maxs10[9]

max(vars10-vars1)
# It should not be symmetric!! 
##############################################################################################


##############################################################################################
# - epsilon: beta R**2 
# extract R2 values.
reps=500
R2_epsilon= np.zeros((epsilons.shape[0], reps))
for i in range(epsilons.shape[0]):
    for j in range(reps):
        betas_epsilon_i= betas_epsilon[j][i]
        R2_epsilon[i][j]= R2(y,np.dot(betas_epsilon_i, X.T)) # 0.0723196328281206

R2_epsilon_bottom= np.zeros(len(epsilons))
R2_epsilon_top= np.zeros(len(epsilons))
for i in range(epsilons.shape[0]):
    R2_epsilon_bottom[i]=  np.percentile(R2_epsilon[i], 2.5)
    R2_epsilon_top[i]= np.percentile(R2_epsilon[i], 97.5)
    print(np.percentile(R2_epsilon[i], 97.5))
    
    
plt.xlim(xmin=np.min(epsilons), xmax=np.max(epsilons))
plt.ylim(ymin=-.5, ymax=.1)
plt.errorbar(np.arange(0, len(epsilons)), np.mean(R2_epsilon,1), yerr=(np.mean(R2_epsilon,1)-R2_epsilon_bottom, R2_epsilon_top-np.mean(R2_epsilon,1)),  solid_capstyle='projecting', capsize=2,linestyle = 'none', color='black')
plt.scatter(x=np.arange(0, len(epsilons)), y=np.mean(R2_epsilon,1))
ticks= epsilons
plt.xticks(range(0,len(ticks)), ticks, rotation='vertical')
plt.axhline(y=R2_no_dp, linewidth=.6, linestyle='dashed', color='black')
##############################################################################################
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


