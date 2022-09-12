# -*- coding: utf-8 -*-
"""
In this file we run the experiments with synthetic data. 

"""
import os
import numpy as np
os.chdir("~/bcd-glm/scripts")
import generate_data_synthetic as synthetic
from simulation_functions import simulation, save_object 
 
os.chdir("~/bcd-glm/results/Synthetic data results")
# we set the number of repetitions to 500
reps=500
# R squared 
base_results= simulation(reps=reps)
resultsR2_8= simulation(reps=reps, R_squared=.8)
resultsR2_5= simulation(reps=reps, R_squared=.5)
resultsR2_1= simulation(reps=reps, R_squared=.1)

save_object(base_results, "base_results")
save_object(resultsR2_5, "resultsR2_5")
save_object(resultsR2_8, "resultsR2_8")
save_object(resultsR2_1, "resultsR2_1")

#correlation 
resultscor_1= simulation(reps=reps, cor=.1, beta_every_rep=False)
save_object(resultscor_1, "resultscor_1")
resultscor_5= simulation(reps=reps, cor=.5, beta_every_rep=False)
save_object(resultscor_5, "resultscor_5")
resultscor_3= simulation(reps=reps, beta_every_rep=False)
save_object(resultscor_3, "resultscor_3")

#epsilon 
result_epsilon=[]
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5,5,10])
error_epsilon=np.zeros((epsilons.shape[0],reps, 10))
for e, epsilon in enumerate(epsilons):
    result_epsilon.append(simulation(reps=reps,epsilon=epsilon, no_dp=False))
save_object(result_epsilon, "result_epsilon")

# gamma 
result_gamma=[]
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5,3])
for g, gamma in enumerate(gammas):
    result_gamma.append(simulation(reps=500,gamma=gamma, no_dp=False))
save_object(result_gamma, "result_gamma")

# sample size 
result_sample_size=[]
sample_sizes= np.array([100, 250, 1000, 5000, 10000])
for n, sample_size in enumerate(sample_sizes):
    print(n)
    result_sample_size.append(simulation(reps=100,n=sample_size))
save_object(result_sample_size, "result_sample_size")