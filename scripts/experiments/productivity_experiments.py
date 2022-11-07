# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:55:31 2022

@author: kroessks
"""
import pandas as pd
import numpy as np
import os

file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd+"/../../data")
prod = pd.read_csv( r"garments_worker_productivity.csv")
os.chdir(file_wd+"/..")
from bcd import fit_glm, initialize_players
os.chdir(file_wd)
from simulation_functions import save_object, lin_reg
# preprocess data
import copy
from datetime import datetime 
prod_processed = copy.deepcopy(prod)
prod_processed = prod_processed.drop(columns=['wip'])
prod_processed =  pd.get_dummies(prod_processed, drop_first=True, columns = ['department', 'day'])
prod_processed.quarter = prod_processed.quarter.factorize()[0]
prod_processed = prod_processed.drop(columns=['date'])

# [1197 rows x 19 columns]
# formulate a research question:
# predict actual_productivity with the rest

# now we split the days between the parties:
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
# so we have omitted two variables: wip and date


betas_init= None
beta_centralized= lin_reg(X, y)

###################################################################
""" test epsilon """
# so lets start to evaluate the impact of epsilon
import datetime
reps=1000
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5,5,10])
betas_epsilon=np.zeros((reps, epsilons.shape[0], X.shape[1]))
for j in range(reps):
    print('STARTING ON REPETITION ' + str(j)+' eps')
    print('\n\n\n')
    file1 = open("log_process.txt", "a")
    now = datetime.datetime.now()
    file1.write(str(j)+"___"+str(now)+"___epsilon\n")
    file1.close()
    for i, e in enumerate(epsilons):
        betas_init= None
        players_dp = initialize_players(
            model="linear",
            X_splitted=X_splitted,
            y=y,
            beta_init_splitted=betas_init,
            DP=True, 
            epsilon=e,
            gamma=1.2
        )
        betas_dp, it, all_betas_dp = fit_glm(
            players=players_dp,
            outer_iterations=5,
            inner_iterations=1,
            tol=10**-15,
        )
        betas_epsilon[j][i]= betas_dp
    os.chdir(file_wd+"/../../results")
    save_object(betas_epsilon, "betas_epsilon_prod")

""" test gamma """
# save_object(betas_gamma, 'betas_gamma_forest')
# so lets start to evaluate the impact of gamma
reps=1000
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5, 3])
betas_gamma=np.zeros((reps, gammas.shape[0], X.shape[1]))
for j in range(reps):
    print('STARTING ON REPETITION ' + str(j)+' gam')
    print('\n\n\n')
    for i, g in enumerate(gammas):
        betas_init= None
        players_dp = initialize_players(
            model="linear",
            X_splitted=X_splitted,
            y=y,
            beta_init_splitted=betas_init,
            DP=True, 
            epsilon=1,
            gamma=g
        )
        betas_dp, it, all_betas_dp = fit_glm(
            players=players_dp,
            outer_iterations=5,
            inner_iterations=1,
            tol=10**-15,
        )
        betas_gamma[j][i]= betas_dp
    os.chdir(file_wd+"/../../results")
    save_object(betas_gamma, "betas_gamma_prod")





