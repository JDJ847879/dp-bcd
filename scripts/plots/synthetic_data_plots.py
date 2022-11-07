# -*- coding: utf-8 -*-
"""
In this file we create the plots for the synthetic data simulations
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd+"/..")
import generate_data_synthetic as synthetic
os.chdir(file_wd+"/../experiments")
from simulation_functions import extract_R2, save_object , get_df

''' load results ''' 
os.chdir(file_wd+"/../../results")
with open("base_results", "rb") as input_file:
    base_results=pickle.load(input_file)
with open("resultsR2_8", "rb") as input_file:
    resultsR2_8=pickle.load(input_file)
with open("resultsR2_5", "rb") as input_file:
    resultsR2_5=pickle.load(input_file)
with open("resultsR2_1", "rb") as input_file:
    resultsR2_1=pickle.load(input_file)
with open("resultsR2_8", "rb") as input_file:
    resultsR2_8=pickle.load(input_file) 
with open("result_epsilon", "rb") as input_file:
    result_epsilon=pickle.load(input_file)     
with open("result_gamma", "rb") as input_file:
    result_gamma=pickle.load(input_file)     
with open("result_sample_size", "rb") as input_file:
    result_sample_size=pickle.load(input_file)      
with open("resultscor_1", "rb") as input_file:
    resultscor_1=pickle.load(input_file)     
with open("resultscor_3", "rb") as input_file:
    resultscor_3=pickle.load(input_file)      
with open("resultscor_5", "rb") as input_file:
    resultscor_5=pickle.load(input_file)     

''' settings for boxplots '''
flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                  linestyle='none', markeredgecolor='none')

'''Effect of R**2'''  
# Table 3: R2 x AAPD
R2_8_error= abs(resultsR2_8[0]-resultsR2_8[1])/abs(resultsR2_8[1])
R2_5_error= abs(resultsR2_5[0]-resultsR2_5[1])/abs(resultsR2_5[1])
base_error= abs(base_results[0]-base_results[1])/abs(base_results[1])
R2_1_error= abs(resultsR2_1[0]-resultsR2_1[1])/abs(resultsR2_1[1])

np.mean(R2_1_error) #4.728646691993917
np.mean(base_error) # 3.0796217692767094
np.mean(R2_5_error) #  1.8759236114187632
np.mean(R2_8_error) #  1.0204003145939353

np.median(R2_1_error) #  0.8549391813629279
np.median(base_error) # 0.47076090022124556
np.median(R2_5_error) # 0.32711101534797415
np.median(R2_8_error) # 0.21481870834710512

# Figure 2: R2 x R2
R2_1= extract_R2(resultsR2_1)
R2_base= extract_R2(base_results)
R2_5= extract_R2(resultsR2_5)
R2_8= extract_R2(resultsR2_8)

R2_R2= [R2_1, R2_base,R2_5,R2_8]
inds= np.array([1,2,3, 4])   
cors= [.1,.3,.5,.8]
for i, ind in enumerate(inds):
    plt.scatter(np.repeat(ind, 500), R2_R2[i][0], alpha=0.1, label= "$R^2$ = "+str(cors[i])+' DP')
    plt.boxplot( R2_R2[i][0], widths=0.1, positions=[ind], medianprops=dict(color='black'), flierprops=flierprops)
for i, ind in enumerate(inds):
    plt.scatter(np.repeat(ind+.2, 500), R2_R2[i][1], alpha=0.1, label= "$R^2$= "+str(cors[i])+' no DP')
    plt.boxplot( R2_R2[i][1], widths=0.1, positions=[ind+.2], medianprops=dict(color='black'), flierprops=flierprops)

leg = plt.legend(bbox_to_anchor=(1.005, 1.025))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xticks([])

'''Effect of correlation '''
errorcor_1= abs(resultscor_1[0]-resultscor_1[1])/abs(resultscor_1[1])
errorcor_5= abs(resultscor_5[0]-resultscor_5[1])/abs(resultscor_5[1])
errorcor_3= abs(resultscor_3[0]-resultscor_3[1])/abs(resultscor_3[1])

# Table  4: correlation x AAPD
np.round(np.median(np.median(errorcor_1, axis=1)),2) # 0.27
np.round(np.mean(errorcor_1),2) #0.56
np.round(np.median(np.median(errorcor_3, axis=1)),2) # 0.69
np.round(np.mean(errorcor_3),2) # 1.06
np.round(np.median(np.median(errorcor_5, axis=1)),2) # 0.69
np.round(np.mean(errorcor_5),2) #3.11

# Figure 4: correlation x R2
R2_cor_1= extract_R2(resultscor_1)
R2_cor_5= extract_R2(resultscor_5)
R2_cor= [R2_cor_1,R2_base,R2_cor_5]
cors2= np.array([1,2,3])   
for c, cor in enumerate(cors2):
    plt.scatter(np.repeat(cor, 500), R2_cor[c][0], alpha=0.1, label= "correlation = "+str(cors[c])+' DP')
    plt.boxplot( R2_cor[c][0], widths=0.04, positions=[cor], medianprops=dict(color='black'))

for c, cor in enumerate(cors2):
    plt.scatter(np.repeat(cor+.2, 500), R2_cor[c][1], alpha=0.1, label= "correlation = "+str(cors[c])+' no DP')
    plt.boxplot( R2_cor[c][1], widths=0.04, positions=[cor+.2], medianprops=dict(color='black'))
leg = plt.legend(bbox_to_anchor=(1.2, -.01), ncol=3)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xticks([])
plt.ylabel('$R^2$')

'''Effect of epsilon '''
# Figure 1: epsilon x R2
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5, 10, 100])
R2_eps=[]
for i in range(len(epsilons)):
    R2_eps.append(extract_R2(result_epsilon[i], result_epsilon[0])[0])
reps=500    
np.array(R2_eps)
epsilons_nonzero= epsilons[1:]
pos= np.arange(1,len(epsilons))

plt.scatter(np.repeat(0, reps), R2_eps[0], alpha=.02, label='no DP')
plt.boxplot( R2_eps[0], widths=0.1, positions=[0], medianprops=dict(color='black'), flierprops=flierprops)
for e, eps in enumerate(epsilons_nonzero):
    plt.scatter(np.repeat(pos[e]+.2, reps), R2_eps[e+1], alpha=.02, label= "$\epsilon$ = "+str(int(eps*2))+' DP')
    plt.boxplot( R2_eps[e+1], widths=0.1, positions=[pos[e]+.2], medianprops=dict(color='black'), flierprops=flierprops)
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(.165, -.37, 0.6,0.4), loc='lower center', ncol=3)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xlabel(r'$\varepsilon$')
plt.ylabel('$R^2$')
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')
plt.ylim([-.5,.4])

# Table 1: epsilon x AAPD
mean_R2_eps= np.zeros(len(R2_eps))
perc1_R2_eps= np.zeros(len(R2_eps))
perc2_R2_eps= np.zeros(len(R2_eps))
for i in range(len(R2_eps)):
    mean_R2_eps[i]= np.mean(np.absolute(R2_eps[i]))
    perc1_R2_eps[i]= np.percentile(np.absolute(R2_eps[i]), 5)
    perc2_R2_eps[i]= np.percentile(np.absolute(R2_eps[i]), 95)
# mean    
# array([0.30574449, 0.45339398, 0.21060409, 0.21551859, 0.2394048 ,
#        0.25103541, 0.2683771 , 0.28257663])
# perc1_R2_eps
# array([0.26975418, 0.02257164, 0.02942485, 0.03778564, 0.08116835,
#        0.11226726, 0.16748619, 0.21648078])
# perc2_R2_eps
# array([0.34441424, 1.68837715, 0.33784755, 0.31505128, 0.32185289,
#        0.32443777, 0.32813981, 0.33091174])

np.round(np.mean(error_epsilon,(2,1)),2)
#  array([0.  , 1.5 , 0.86, 0.67, 0.53, 0.47, 0.38, 0.3 , 0.21, 0.15])
np.round(np.median(error_epsilon,(2,1)),2)
# array([ 0.  , 10.09,  5.68,  4.37,  3.45,  3.08,  2.51,  1.94,  1.37,
#         0.97])

''' Effect of gamma '''
# Table 2: gamma x AAPD
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])
error_gamma=np.zeros((gammas.shape[0],500,10))

for g, gamma in enumerate(gammas):
    error_gamma[g]= abs(result_gamma[g][0]-base_results[1])/abs(base_results[1])
np.round(np.mean(error_gamma,(2,1)),2)
#  array([2.95, 3.21, 3.86, 4.64, 5.17, 6.51, 7.87])
np.round(np.median(error_gamma,(2,1)),2)
# array([0.45, 0.49, 0.59, 0.71, 0.79, 0.98, 1.18])

# Figure 1: gamma x R2
R2_gam=[]
for i in range(len(gammas)):
    R2_gam.append(extract_R2(result_gamma[i], result_gamma[0])[0])
reps=500
np.array(R2_gam)
gammas_nonzero= gammas[1:]
pos= np.arange(1,len(gammas))
plt.scatter(np.repeat(0, reps), R2_gam[0], alpha=.02, label='no DP')
plt.boxplot( R2_gam[0], widths=0.1, positions=[0], medianprops=dict(color='black'), flierprops=flierprops)
for g, gam in enumerate(gammas_nonzero):
    print(g)
    plt.scatter(np.repeat(pos[g]+.2, reps), R2_gam[g+1], alpha=.02, label= "$\gamma$ = "+str(gam)+' DP')
    plt.boxplot( R2_gam[g+1], widths=0.1, positions=[pos[g]+.2], medianprops=dict(color='black'), flierprops=flierprops)
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(.45, -.31), loc='lower center', ncol=3)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')
plt.ylabel('$R^2$')
plt.ylim([-.5,.4])

''' Effect of sample size '''
sample_sizes= np.array([100, 250, 1000, 5000, 10000])

# Figure 6: sample size x R2
R2_n_size=[]
for i in range(len(sample_sizes)):
    R2_n_size.append(extract_R2(result_sample_size[i])[0])
reps=100    
sample_sizes_nonzero= sample_sizes[1:]
pos= np.arange(len(sample_sizes))
alpha=.1
plt.ylim([-.5, .5])
for n, n_size in enumerate(sample_sizes):
    if n==(len(sample_sizes)-1):
        alpha=.02
    plt.scatter(np.repeat(pos[n]+.2, reps), R2_n_size[n], alpha=alpha, label= "$n$ = "+str(n_size)+' DP')
    plt.boxplot( R2_n_size[n], widths=0.04, positions=[pos[n]+.2], medianprops=dict(color='black'), flierprops=flierprops)
plt.legend() 
plt.xticks([]) 
leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')

# Table 5: sample size x AAPD
np.round(np.mean(error_sample_size,(2,1)),2)
 # array([7.68, 3.82, 2.3 , 0.85, 0.59])
np.round(np.median(error_sample_size,(2,1)),2)
#array([1.46, 0.95, 0.47, 0.22, 0.15])
