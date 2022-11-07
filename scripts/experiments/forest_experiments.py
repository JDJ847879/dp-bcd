# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:45:23 2022
In this file we run the experiments we performed with the forest fire data set by Cortez et al. (2007).
Cortez, P., & Morais, A. D. J. R. (2007). A data mining approach to predict forest fires using meteorological data. 
We vary epsilon and gamma and run the centralized regression.
@author: kroessks
"""
import os
import numpy as np
import pandas as pd
file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd+"/..")
from bcd import fit_glm, initialize_players
os.chdir(file_wd+"/..")
import generate_data_forestfires as forest
os.chdir(file_wd)
from simulation_functions import save_object
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

""" test epsilon """
# so lets start to evaluate the impact of epsilon
reps=1000
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5,5,10])
betas_epsilon=np.zeros((reps, epsilons.shape[0], 27))
for j in range(reps):
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
save_object(betas_epsilon, "betas_epsilon_forest")

""" test gamma """
# evaluate the impact of gamma
reps=1000
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5, 3])
betas_gamma=np.zeros((reps, gammas.shape[0], 27))
for j in range(reps):
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
save_object(betas_gamma, "betas_gamma_forest")

