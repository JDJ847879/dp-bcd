# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:51:50 2022

@author: kroessks
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("~/bcd-glm/scripts")
import generate_data_synthetic as synthetic
import numpy as np
os.chdir("~/bcd-glm/scripts")
from bcd import fit_glm, initialize_players
os.chdir("~/bcd-glm/scripts")
from simulation_functions import extract_R2, save_object , get_df
import pickle

''' load results ''' 
os.chdir("~/Synthetic data results")
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

''' plots R**2'''    
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

R2s= np.array([0.1,0.3,0.5,0.8])    
results_R2= np.vstack([np.mean(R2_1_error,1),np.mean(base_error,1), np.mean(R2_5_error,1), np.mean(R2_8_error,1) ])
# plt.title('Percentual beta error over R2relation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
for c, R2 in enumerate(R2s):
    plt.scatter(np.repeat(R2, 500), results_R2[c], alpha=0.4, label= "R2relation = "+str(R2))
plt.boxplot(results_R2.T, widths=0.04, positions=R2s, medianprops=dict(color='black'), flierprops=flierprops)
plt.semilogy()         
plt.xlabel("$R^2$")
#########################################################################
R2_1= extract_R2(resultsR2_1)
R2_base= extract_R2(base_results)
R2_5= extract_R2(resultsR2_5)
R2_8= extract_R2(resultsR2_8)

R2_R2= [R2_1, R2_base,R2_5,R2_8]
inds= np.array([1,2,3, 4])   
cors= [.1,.3,.5,.8]
# plt.title('$R^2$ over correlation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
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

np.round(np.mean(R2_1_error),2) # 4.73

# for the median we first
np.round(np.median(R2_1_error.reshape(-1))) # 0.9
np.round(np.mean(base_error)) # 3
np.round(np.median(base_error.reshape(-1))) # 0
np.round(np.mean(R2_5_error),2) # 1.88
np.round(np.median(R2_5_error.reshape(-1))) # 0.9
np.round(np.mean(R2_8_error)) # 1
np.round(np.median(R2_8_error.reshape(-1))) # 0.9


'''correlation '''
errorcor_1= abs(resultscor_1[0]-resultscor_1[1])/abs(resultscor_1[1])
errorcor_5= abs(resultscor_5[0]-resultscor_5[1])/abs(resultscor_5[1])
errorcor_3= abs(resultscor_3[0]-resultscor_3[1])/abs(resultscor_3[1])

cors= np.array([0.1,0.3,0.5])    
results_cor= np.vstack([np.mean(errorcor_1,1),np.mean(errorcor_3,1), np.mean(errorcor_5,1) ])
# plt.title('Percentual beta error over correlation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
for c, cor in enumerate(cors):
    plt.scatter(np.repeat(cor, 500), results_cor[c], alpha=0.4, label= "correlation = "+str(cor))
plt.boxplot(results_cor.T, widths=0.04, positions=cors, medianprops=dict(color='black'))
plt.semilogy()

np.round(np.median(np.median(errorcor_1, axis=1)),2) # 0.27
np.round(np.mean(errorcor_1),2) #0.56
np.round(np.median(np.median(errorcor_3, axis=1)),2) # 0.69
np.round(np.mean(errorcor_3),2) # 1.06
np.round(np.median(np.median(errorcor_5, axis=1)),2) # 0.69
np.round(np.mean(errorcor_5),2) #3.11



########################################################################
'''plot betas '''
means1= np.mean(resultscor_1[0],0)
means3= np.mean(resultscor_3[0],0)
means5= np.mean(resultscor_5[0],0)
fig, axes= plt.subplots(3,3)
for i in range(9):
    df= get_df(i)
    df.plot.density(ax=axes[int(i/3),i%3], legend=False, style=[':', '--', '-'], linewidth=1)
    #axes[int(i/3),i%3].set_xlabel('variable'+str(i))
    min_val= np.percentile(np.array(df).reshape(1500),0.01)
    max_val= np.percentile(np.array(df).reshape(1500),99.99)
    axes[int(i/3),i%3].set_xlim(min_val,max_val)
    axes[int(i/3),i%3].axvline(x=resultscor_1[4][i], linewidth=.6)
    # axes[int(i/3),i%3].axvline(x=means1[i], linewidth=.6, color="blue")
    # axes[int(i/3),i%3].axvline(x=means3[i], linewidth=.6, color="orange")
    # axes[int(i/3),i%3].axvline(x=means5[i], linewidth=.6, color="green")
    axes[int(i/3),i%3].plot(means1[i],-0.005, marker='o', color="blue", alpha=1, markersize=2.5)
    axes[int(i/3),i%3].plot(means3[i],-0.005 ,marker='s', color="orange", alpha=1, markersize=2.5)
    axes[int(i/3),i%3].plot(means5[i],-0.005,  marker='x', color="green", alpha=1, markersize=2.5)
    
lines, labels=fig.axes[0].get_legend_handles_labels()
axes[0,1].set_ylabel("")
axes[0,2].set_ylabel("")
axes[1,1].set_ylabel("")
axes[1,2].set_ylabel("")
axes[2,1].set_ylabel("")
axes[2,2].set_ylabel("")
# fig.legend(lines, labels, loc="lower center", ncol=2)
plt.show()
    
########################################################################
'''R**2 and correlation'''
# So I think it is interesting that the correlation does not impact R2 but it does have a considerate
# effect on the beta parameters
R2_cor_1= extract_R2(resultscor_1)
R2_cor_5= extract_R2(resultscor_5)
R2_cor= [R2_cor_1,R2_base,R2_cor_5]
cors2= np.array([1,2,3])   
# plt.title('$R^2$ over correlation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
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

'''epsilon '''
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5, 10, 100])
reps=500
for e, epsilon in enumerate(epsilons):
    error_epsilon[e]= abs(result_epsilon[e][0]-result_epsilon[0][0])/abs(result_epsilon[0][0])
    
for e, epsilon in enumerate(epsilons):
    plt.scatter(np.repeat(epsilon, reps), np.mean(error_epsilon[e],1), alpha=0.2, label= "epsilon = "+str(int(eps*2)))
plt.boxplot(np.mean(error_epsilon,2).T, widths=0.04, positions=epsilons)
plt.xscale('symlog')
plt.xlabel(r'$\varepsilon$')
# plt.title('Percentual beta error over epsilon for gamma = 1.5, \n and delta = 1 with synthetic data (500 repetitions)')
plt.ylabel('Average percentual absolute error \n beta compared to no DP')
plt.legend()

##############################################################################################
# R2
R2_eps=[]
for i in range(len(epsilons)):
    R2_eps.append(extract_R2(result_epsilon[i], result_epsilon[0])[0])
reps=500    
np.array(R2_eps)
epsilons_nonzero= epsilons[1:]
pos= np.arange(1,len(epsilons))

# plt.title('$R^2$ over epsilon for gamma = 1.5, \n and delta = 1 with synthetic data (500 repetitions, centralized $R^2$=0.3)')
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

####################

''' Table 1'''
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
# we use gamma is 0 as the baseline    
##############################################################################################

''' gamma '''
#result_gamma=[]
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])
error_gamma=np.zeros((gammas.shape[0],500,10))

for g, gamma in enumerate(gammas):
    error_gamma[g]= abs(result_gamma[g][0]-base_results[1])/abs(base_results[1])

np.round(np.mean(error_gamma,(2,1)),2)
#  array([2.95, 3.21, 3.86, 4.64, 5.17, 6.51, 7.87])
np.round(np.median(error_gamma,(2,1)),2)
# array([0.45, 0.49, 0.59, 0.71, 0.79, 0.98, 1.18])
# we use gamma is 0 as the baseline    
    
for g, gamma in enumerate(gammas):
    plt.scatter(np.repeat(gamma, 500), np.mean(error_gamma[g],1), alpha=0.2, label= "gamma = "+str(gamma))
    plt.boxplot(np.mean(error_gamma[g],1), widths=0.04, positions=[gamma], medianprops=dict(color='black'), flierprops=flierprops)
plt.xscale('symlog')
plt.xlabel('gamma')
# plt.title('Percentual beta error over gamma for epsilon = 1, \n and delta = 1 with synthetic data (500 repetitions)')
plt.ylabel('Average percentual absolute error \n beta compared to no DP')
plt.legend()

##############################################################################################
# R2
R2_gam=[]
for i in range(len(gammas)):
    R2_gam.append(extract_R2(result_gamma[i], result_gamma[0])[0])


reps=500
np.array(R2_gam)
gammas_nonzero= gammas[1:]
pos= np.arange(1,len(gammas))
# plt.title('$R^2$ over gamma for epsilon = 1, \n with synthetic data (500 repetitions, centralized $R^2$=0.3)')
plt.scatter(np.repeat(0, reps), R2_gam[0], alpha=.02, label='no DP')
plt.boxplot( R2_gam[0], widths=0.1, positions=[0], medianprops=dict(color='black'), flierprops=flierprops)
for g, gam in enumerate(gammas_nonzero):
    print(g)
    plt.scatter(np.repeat(pos[g]+.2, reps), R2_gam[g+1], alpha=.02, label= "$\gamma$ = "+str(gam)+' DP')
    plt.boxplot( R2_gam[g+1], widths=0.1, positions=[pos[g]+.2], medianprops=dict(color='black'), flierprops=flierprops)
plt.legend()
plt.xticks([])
# leg = plt.legend(loc='lower center')
leg = plt.legend(bbox_to_anchor=(.45, -.31), loc='lower center', ncol=3)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')
plt.ylabel('$R^2$')
# plt.xlabel(r'$\gamma$')
plt.ylim([-.5,.4])
##############################################################################################
##############################################################################################
##############################################################################################
''' sample size '''
error_sample_size=np.zeros((sample_sizes.shape[0],100, 10))
sample_sizes= np.array([100, 250, 1000, 5000, 10000])
for n, sample_size in enumerate(sample_sizes):
    error_sample_size[n]= abs(result_sample_size[n][0]-result_sample_size[n][1])/abs(result_sample_size[n][1])

for n, n_size in enumerate(sample_sizes):
    print(n)
    plt.scatter(np.repeat(n, 100), np.mean(error_sample_size[n],1), alpha=0.2, label = "sample size = "+str(n_size))
    plt.boxplot(np.mean(error_sample_size[n],1), widths=0.15, positions=[n], medianprops=dict(color='black'), flierprops=flierprops)
plt.xticks(sample_sizes) 
plt.xlim([-1,5])
plt.xlabel('n')
# plt.title('Percentual beta error over sample size for gamma = 1.5, epsilon = 1, \n and delta = 1 with synthetic data (100 repetitions)')
plt.ylabel('AAPD')
leg=plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)
# here we need to subtract the true beta still
##############################################################################################
# R2
R2_n_size=[]
for i in range(len(sample_sizes)):
    R2_n_size.append(extract_R2(result_sample_size[i])[0])
reps=100    
sample_sizes_nonzero= sample_sizes[1:]
pos= np.arange(len(sample_sizes))
# plt.title('$R^2$ over sample_size for epsilon = 1, \n with synthetic data (500 repetitions, centralized $R^2$=0.3)')
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
# leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')

np.round(np.mean(error_sample_size,(2,1)),2)
 # array([7.68, 3.82, 2.3 , 0.85, 0.59])
np.round(np.median(error_sample_size,(2,1)),2)
#array([1.46, 0.95, 0.47, 0.22, 0.15])
