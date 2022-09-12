# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:51:50 2022

@author: kroessks
"""
import os
import pandas as pd
import generate_data_synthetic as synthetic
import numpy as np
import matplotlib.pyplot as plt
from bcd import fit_glm, initialize_players
import matplotlib.pyplot as plt
from simulation_functions import simulation, extract_R2, run_fit_glm, save_object 
import pickle

''' R squared '''
# now we can start looping this over values of R squared or otherwise. 
# the default settings are the "base" to which we can compare
base_results= simulation(reps=500)
base_error= abs(base_results[0]-base_results[1])/abs(base_results[1])
np.mean(abs(base_results[0]-base_results[1])/abs(base_results[1]))
# 5.578660255074383

resultsR2_8= simulation(reps=500, R_squared=.8)
R2_8_error= abs(resultsR2_8[0]-resultsR2_8[1])/abs(resultsR2_8[1])
np.mean(abs(resultsR2_8[0]-resultsR2_8[1])/abs(resultsR2_8[1]))

resultsR2_5= simulation(reps=500, R_squared=.5)
R2_5_error= abs(resultsR2_5[0]-resultsR2_5[1])/abs(resultsR2_5[1])

R2_R2_5= extract_R2(resultsR2_5)
R2_R2_8= extract_R2(resultsR2_8)
# R2_base= extract_R2(base_results)
# save_object(base_results, "base_results")
save_object(resultsR2_5, "resultsR2_5")
save_object(resultsR2_8, "resultsR2_8")

# with open("base_results", "rb") as input_file:
#     base_results=pickle.load(input_file)
# with open("resultsR2_5", "rb") as input_file:
#     resultsR2_5=pickle.load(input_file)
# with open("R2_R2_8", "rb") as input_file:
#     R2_R2_8=pickle.load(input_file)
R2s= np.array([0.1,0.3,0.8])    
results_R2= np.vstack([np.mean(base_error,1),np.mean(R2_5_error,1), np.mean(R2_8_error,1) ])
# plt.title('Percentual beta error over R2relation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
for c, R2 in enumerate(R2s):
    plt.scatter(np.repeat(R2, 500), results_R2[c], alpha=0.4, label= "R2relation = "+str(R2))
plt.boxplot(results_R2.T, widths=0.04, positions=R2s, medianprops=dict(color='black'))
plt.semilogy()         
plt.xlabel("$R^2$")
#It indeed makes a huge difference
# plt.hist(base_error.reshape(-1),alpha=.5, density=True,log=True, label="R_squared = 0.3")
# plt.hist(R2_8_error.reshape(-1),alpha=.5, density=True,log=True, label="R_squared = 0.8")
# plt.title('Percentual beta error over gamma for epsilon = 1, gamma=1.5, and delta = 1/n with synthetic data (500 repetitions)')
# plt.ylabel('log(density)')
# plt.xlabel('Average percentual absolute error beta compared to no DP')
# plt.legend()

# plt.hist(R2_R2_8[1],alpha=.5, density=True, label="R_squared = 0.8, no differential privacy")
# plt.hist(R2_base[1],alpha=.5, density=True, label="R_squared = 0.3, no differential privacy")
# plt.hist(R2_R2_8[0],alpha=.5, density=True, label="R_squared = 0.8, differential privacy")
# plt.hist(R2_base[0],alpha=.5, density=True, label="R_squared = 0.3, differential privacy")
# plt.title('R2 for epsilon = 1, gamma=1.5, and delta = 1/n with synthetic data (500 repetitions)')
# plt.ylabel('log(density)')
# plt.xlabel('R2 compared to no DP')
# plt.legend()
R2_5= extract_R2(resultsR2_5)
R2_8= extract_R2(resultsR2_8)
R2_base= extract_R2(base_results)
R2_R2= [R2_base,R2_5,R2_8]
cors2= np.array([1,2,3])   
cors= [.3,.5,.8]
plt.title('$R^2$ over correlation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
for c, cor in enumerate(cors2):
    plt.scatter(np.repeat(cor, 500), R2_R2[c][0], alpha=0.1, label= "$R^2$ = "+str(cors[c])+' DP')
    plt.scatter(np.repeat(cor+.2, 500), R2_R2[c][1], alpha=0.1, label= "$R^2$= "+str(cors[c])+' no DP')
    plt.boxplot( R2_R2[c][0], widths=0.04, positions=[cor], medianprops=dict(color='black'))
    plt.boxplot( R2_R2[c][1], widths=0.04, positions=[cor+.2], medianprops=dict(color='black'))
leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xticks([])



np.round(np.mean(base_error),2)
# 1.47
np.round(np.median(base_error),2)
# 0.4
np.round(np.mean(R2_8_error,(1)),2)
np.round(np.median(R2_8_error,(1)),2)

'''correlation '''
resultscor_1= simulation(reps=500, cor=.1, beta_every_rep=False)
errorcor_1= abs(resultscor_1[0]-resultscor_1[1])/abs(resultscor_1[1])
save_object(resultscor_1, "resultscor_1")
resultscor_5= simulation(reps=500, cor=.5, beta_every_rep=False)
errorcor_5= abs(resultscor_5[0]-resultscor_5[1])/abs(resultscor_5[1])
save_object(resultscor_5, "resultscor_5")
resultscor_3= simulation(reps=500, beta_every_rep=False)
errorcor_3= abs(resultscor_5[0]-resultscor_5[1])/abs(resultscor_5[1])
save_object(resultscor_3, "resultscor_3")

# with open("resultscor_1", "rb") as input_file:
#     resultscor_1=pickle.load(input_file)
# with open("resultscor_5", "rb") as input_file:
#     resultscor_5=pickle.load(input_file)
# with open("resultscor_3", "rb") as input_file:
#     resultscor_3=pickle.load(input_file)
    
np.save("resultscor_1.npy", resultscor_1)
np.save("resultscor_5.npy", resultscor_5)

plt.hist(errorcor_1.reshape(-1),alpha=.8, density=True,log=True, label="correlation = 0.1")
plt.hist(base_error.reshape(-1),alpha=.8, density=True,log=True, label="correlation = 0.3")
plt.hist(errorcor_5.reshape(-1),alpha=.3, density=True,log=True, label="correlation = 0.8")
plt.title('Percentual beta error over correlation for epsilon = 1.5, gamma =1.2, and delta = 1 with synthetic data (500 repetitions)')
plt.ylabel('log(density)')
plt.xlabel('Average percentual absolute error beta compared to no DP')
plt.legend()

cors= np.array([0.1,0.3,0.8])    
results_cor= np.vstack([np.mean(errorcor_1,1),np.mean(errorcor_3,1), np.mean(errorcor_5,1) ])
plt.title('Percentual beta error over correlation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
for c, cor in enumerate(cors):
    plt.scatter(np.repeat(cor, 500), results_cor[c], alpha=0.4, label= "correlation = "+str(cor))
plt.boxplot(results_cor.T, widths=0.04, positions=cors, medianprops=dict(color='black'))
plt.semilogy()
########################################################################
'''plot betas '''

def get_df(var):
    estimates=np.vstack([resultscor_1[0].T[var], resultscor_3[0].T[var],resultscor_5[0].T[var]]).T
    df_res= pd.DataFrame(estimates)
    df_res.columns=['correlation=.1', 'correlation=.3', 'correlation=.5']
    return df_res

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
plt.title('$R^2$ over correlation for epsilon = 1, \n gamma =1.2, and delta = 1/n with synthetic data (500 repetitions)')
for c, cor in enumerate(cors2):
    plt.scatter(np.repeat(cor, 500), R2_cor[c][0], alpha=0.1, label= "correlation = "+str(cors[c])+' DP')
    plt.scatter(np.repeat(cor+.2, 500), R2_cor[c][1], alpha=0.1, label= "correlation = "+str(cors[c])+' no DP')
    plt.boxplot( R2_cor[c][0], widths=0.04, positions=[cor], medianprops=dict(color='black'))
    plt.boxplot( R2_cor[c][1], widths=0.04, positions=[cor+.2], medianprops=dict(color='black'))
leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xticks([])


########################################################################

# np.save("errorcor_1", errorcor_1)
# np.save("errorcor_8", errorcor_8)
np.round(np.mean(errorcor_1),2)
# 0.71
np.round(np.median(errorcor_1),2)
# 0.24
np.round(np.mean(errorcor_3),2)
# 1.47
np.round(np.median(errorcor_3),2)
# 0.4
np.round(np.mean(errorcor_8),2)
# 16.51
np.round(np.median(errorcor_8),2)
# 1.02

'''epsilon '''
result_epsilon=[]
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5, 10, 100])
reps=500
error_epsilon=np.zeros((epsilons.shape[0],reps, 10))

for e, epsilon in enumerate(epsilons):
    result_epsilon.append(simulation(reps=reps,epsilon=epsilon, no_dp=False))
# we use epsilon is 0 as the baseline
for e, epsilon in enumerate(epsilons):
    error_epsilon[e]= abs(result_epsilon[e][0]-result_epsilon[0][0])/abs(result_epsilon[0][0])
save_object(result_epsilon, "result_epsilon")
    
for e, epsilon in enumerate(epsilons):
    plt.scatter(np.repeat(epsilon, reps), np.mean(error_epsilon[e],1), alpha=0.2, label= "epsilon = "+str(epsilon))
plt.boxplot(np.mean(error_epsilon,2).T, widths=0.04, positions=epsilons)
plt.xscale('symlog')
plt.xlabel(r'$\epsilon$')
plt.title('Percentual beta error over epsilon for gamma = 1.5, \n and delta = 1 with synthetic data (500 repetitions)')
plt.ylabel('Average percentual absolute error \n beta compared to no DP')
plt.legend()

np.round(np.mean(error_epsilon,(2,1)),2)
# array([0.  , 2.54, 1.95, 1.73, 1.55, 1.47, 1.35, 1.2 ])
np.round(np.median(error_epsilon,(2,1)),2)
# array([0.  , 0.69, 0.53, 0.47, 0.42, 0.4 , 0.36, 0.33])

##############################################################################################
# R2
R2_eps=[]
for i in range(len(epsilons)):
    R2_eps.append(extract_R2(result_epsilon[i], result_epsilon[0])[0])
    
np.array(R2_eps)
epsilons_nonzero= epsilons[1:]
pos= np.arange(1,len(epsilons))

plt.title('$R^2$ over epsilon for gamma = 1.5, \n and delta = 1 with synthetic data (500 repetitions, centralized $R^2$=0.3)')
plt.scatter(np.repeat(0, reps), R2_eps[0], alpha=.02, label='no DP')
plt.boxplot( R2_eps[0], widths=0.04, positions=[0], medianprops=dict(color='black'))
for e, eps in enumerate(epsilons_nonzero):
    plt.scatter(np.repeat(pos[e]+.2, reps), R2_eps[e+1], alpha=.02, label= "epsilon = "+str(eps)+' DP')
    plt.boxplot( R2_eps[e+1], widths=0.04, positions=[pos[e]+.2], medianprops=dict(color='black'))
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.semilogy()
plt.xlabel(r'$\epsilon$')



plt.title('$R^2$ over epsilon for gamma = 1.5, \n and delta = 1 with synthetic data (500 repetitions, centralized $R^2$=0.3)')
plt.scatter(np.repeat(0, reps), R2_eps[0], alpha=.008, label='no DP')
# plt.boxplot( R2_eps[0], widths=0.04, positions=[0], medianprops=dict(color='black'))
plt.boxplot( R2_eps[0], widths=0.35, positions=[0], medianprops=dict(color='yellow'),  flierprops=dict(alpha=.2, markersize=2))
for e, eps in enumerate(epsilons_nonzero):
    plt.scatter(np.repeat(pos[e]+.2, reps), R2_eps[e+1], alpha=.008, label= "epsilon = "+str(eps)+' DP')
    plt.boxplot( R2_eps[e+1], widths=0.35, positions=[pos[e]+.2], medianprops=dict(color='yellow'),  flierprops=dict(alpha=.2, markersize=2))
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')

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

# we could also do this for Gamma = 1.2!
''' epsilon: gamma = 1.2 '''
result_epsilon_1_2=[]
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5, 5, 10])
reps=500

for e, epsilon in enumerate(epsilons):
    result_epsilon_1_2.append(simulation(reps=reps,epsilon=epsilon, no_dp=False, gamma=1.2))
save_object(result_epsilon_1_2, "result_epsilon_1_2")
    

    
# let's also evaluate whether R2 is above the minimum, by plotting its limit
# considering that we have a gamma of 1.5, 2 parties, 5 iterations and a true R2 of 0.3
# that would indicate that the bound for R2 is (on average) -996
# which is very small.
# we can still evaluate whether it is not even lower.
# it might also be nice to do an analysis using the required iterations by
# van Kesteren, assuming 2 (the minimum no.) parties and then show what can be
# expected realisticly (though it can be above the bound).


##############################################################################################
##############################################################################################
##############################################################################################

''' gamma '''
#result_gamma=[]
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])
error_gamma=np.zeros((gammas.shape[0],500,10))

np.round(np.mean(error_gamma,(2,1)),2)
#  array([2.95, 3.21, 3.86, 4.64, 5.17, 6.51, 7.87])
np.round(np.median(error_gamma,(2,1)),2)
# array([0.45, 0.49, 0.59, 0.71, 0.79, 0.98, 1.18])
for g, gamma in enumerate(gammas):
    result_gamma.append(simulation(reps=500,gamma=gamma, no_dp=False))
    # error_gamma[g]= abs(result_gamma[g][0]-result_gamma[g][1])/abs(result_gamma[g][1])
# we use gamma is 0 as the baseline
for g, gamma in enumerate(gammas):
    error_gamma[g]= abs(result_gamma[g][0]-base_results[1])/abs(base_results[1])
    
    
for g, gamma in enumerate(gammas):
    plt.scatter(np.repeat(gamma, 500), np.mean(error_gamma[g],1), alpha=0.2, label= "gamma = "+str(gamma))
plt.boxplot(np.mean(error_gamma,2).T, widths=0.04, positions=gammas)
plt.xscale('symlog')
plt.xlabel('gamma')
# plt.title('Percentual beta error over gamma for epsilon = 1, \n and delta = 1 with synthetic data (500 repetitions)')
plt.ylabel('Average percentual absolute error \n beta compared to no DP')
plt.legend()

# also look at the median and mean
np.round(np.mean(error_gamma,(2,1)),2)
# array([0.  , 0.13, 0.46, 0.85, 1.11, 1.76, 2.42])
np.round(np.median(error_gamma,(2,1)),2)
# array([0.  , 0.03, 0.09, 0.17, 0.22, 0.35, 0.48])

##############################################################################################
# R2
R2_gam=[]
for i in range(len(gammas)):
    R2_gam.append(extract_R2(result_gamma[i], result_gamma[0])[0])
    
np.array(R2_gam)
gammas_nonzero= gammas[1:]
pos= np.arange(1,len(gammas))
plt.title('$R^2$ over gamma for epsilon = 1, \n with synthetic data (500 repetitions, centralized $R^2$=0.3)')
plt.scatter(np.repeat(0, reps), R2_gam[0], alpha=.02, label='no DP')
plt.boxplot( R2_gam[0], widths=0.04, positions=[0], medianprops=dict(color='yellow'),  flierprops=dict(alpha=.2, markersize=2))
for g, gam in enumerate(gammas_nonzero):
    print(g)
    plt.scatter(np.repeat(pos[g]+.2, reps), R2_gam[g+1], alpha=.02, label= "gamma = "+str(gam)+' DP')
    plt.boxplot( R2_gam[g+1], widths=0.04, positions=[pos[g]+.2], medianprops=dict(color='yellow'),  flierprops=dict(alpha=.2, markersize=2))
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.axhline(y=.3, alpha=.8, linewidth=.5, color="black", ls='-.')
plt.xlabel('$R^2$')

##############################################################################################
##############################################################################################
##############################################################################################

# ''' delta '''
''' sample size '''

result_sample_size=[]
sample_sizes= np.array([100, 250, 1000, 5000, 10000])
error_sample_size=np.zeros((sample_sizes.shape[0],100, 10))

for n, sample_size in enumerate(sample_sizes):
    print(n)
    result_sample_size.append(simulation(reps=100,n=sample_size))
    error_sample_size[n]= abs(result_sample_size[n][0]-result_sample_size[n][1])/abs(result_sample_size[n][1])

for i in range(error_sample_size.shape[1]):
    plt.scatter(np.repeat(sample_sizes[i], 100), error_sample_size.T[i], alpha=0.2)
plt.boxplot(error_sample_size, widths=0.04, positions=sample_sizes)

plt.xlabel('sample_size')
plt.title('Percentual beta error over sample size for gamma = 1.5, epsilon = 1, and delta = 1 with synthetic data (100 repetitions)')
plt.ylabel('Percentual absolute error beta')
plt.xscale('log')

# np.save("result_sample_size.npy", result_sample_size)

np.mean(error_sample_size==0, (1,2))
# Out[47]: array([0., 0., 0., 0., 1., 1.])

# So we only have the first four
error_sample_size_subset= error_sample_size[0:4]
for i in range(4):
    plt.scatter(np.repeat(sample_sizes[i], 100), np.mean(error_sample_size[i],1), alpha=0.2, label = "sample size = "+str(sample_sizes[i]))
plt.boxplot(np.mean(error_sample_size_subset,2).T, widths=0.04, positions=sample_sizes[0:4])
plt.xscale('log')
plt.xlim([10**1.75, 10**4.2])
plt.xlabel('sample size')
# plt.title('Percentual beta error over sample size for gamma = 1.5, epsilon = 1, \n and delta = 1 with synthetic data (100 repetitions)')
plt.ylabel('Percentual absolute error beta')
plt.legend()

##############################################################################################
# R2
R2_n_size=[]
for i in range(len(sample_sizes)):
    R2_n_size.append(extract_R2(result_sample_size[i], result_sample_size[0])[0])
    
sample_sizes_nonzero= sample_sizes[1:]
pos= np.arange(1,len(sample_sizes))
# plt.title('$R^2$ over sample_size for epsilon = 1, \n with synthetic data (500 repetitions, centralized $R^2$=0.3)')
plt.scatter(np.repeat(0, reps), R2_n_size[0], alpha=.02, label='no DP')
plt.boxplot( R2_n_size[0], widths=0.04, positions=[0], medianprops=dict(color='black'))
for g, n_size in enumerate(sample_sizes_nonzero):
    plt.scatter(np.repeat(pos[n]+.2, reps), R2_n_size[n+1], alpha=.02, label= "sample_size = "+str(n_size)+' DP')
    plt.boxplot( R2_n_size[n+1], widths=0.04, positions=[pos[n]+.2], medianprops=dict(color='black'))
plt.legend()
plt.xticks([])
leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)

np.round(np.mean(error_sample_size,(2,1)),2)
 # array([7.68, 3.82, 2.3 , 0.59, 0.  , 0.  ])
np.round(np.median(error_sample_size,(2,1)),2)
# array([1.46, 0.95, 0.47, 0.15, 0.  , 0.  ])
