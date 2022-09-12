# -*- coding: utf-8 -*-
"""
All functions required to run and save simulations with synthetic data. 
"""

import os
import pandas as pd
os.chdir("~/bcd-glm/scripts")
import generate_data_synthetic as synthetic
import numpy as np
import matplotlib.pyplot as plt
os.chdir("~/bcd-glm/scripts")
from bcd import fit_glm, initialize_players
import matplotlib.pyplot as plt
import pickle

def run_fit_glm(X_splitted, y, outer_iterations=5, epsilon=1,
    gamma=1.5,
    DP=True
):
    betas_init= None
    players = initialize_players(
        model="linear",
        X_splitted=X_splitted,
        y=y,
        beta_init_splitted=betas_init,
        DP=DP,    
        epsilon=epsilon,
        gamma=gamma
    )
    
    betas, it, all_betas = fit_glm(
        players=players,
        outer_iterations=outer_iterations,
        inner_iterations=1,
        tol=10**-15,
    )
    return betas, it, all_betas

def simulation(reps=100, 
               no_features=10, 
               m=2, 
               R_squared=0.3,  
               epsilon=1,
               gamma=1.2,
               cor=0.3,
               n=1000,
               no_dp=True,
               beta_every_rep= True,
               outer_iterations=5):
    
    all_betas_nodp= np.zeros((reps, outer_iterations, no_features))
    all_betas_dp= np.zeros((reps,outer_iterations, no_features))
    betas_dp = np.zeros((reps, no_features))
    betas_nodp = np.zeros((reps, no_features))
    model = "linear"
    generation_data = "choleski"
    outcome = "gaussian" 
    y_s= np.zeros((reps, n))
    X_s = np.zeros((reps, n, no_features))
    # define possible splitting of the data among parties
    features_splitting = []
    d = no_features // m
    r = no_features % m
    features_per_party=([d] * (m - 1) + [d + r])
    desired_corr_mat = synthetic.get_correlation_matrix(
        n_features=no_features, c=cor
    )
    if not beta_every_rep:
        np.random.seed(int(2022))
        real_betas = np.random.normal(loc=2, scale=1.5, size=no_features)
    for rep in range(reps):
        np.random.seed(int(2022+rep))
        if beta_every_rep:
            real_betas = np.random.normal(loc=2, scale=1.5, size=no_features)
        X_s[rep], y_s[rep] = synthetic.generate_synthetic_data(
            desired_corr_mat, n_features=no_features, R_squared=R_squared,
            betas=real_betas, outcome=outcome, random_seed=int(2022+rep), n_sample=n)
        X_splitted = synthetic.split_data_parties(X_s[rep], features_per_party)
        betas_dp[rep], it_dp, all_betas_dp[rep]= run_fit_glm(X_splitted,y_s[rep], outer_iterations=outer_iterations, epsilon=epsilon, gamma=gamma)
        if no_dp:
            betas_nodp[rep], it_nodp, all_betas_nodp[rep]= run_fit_glm(X_splitted,y_s[rep], outer_iterations=outer_iterations, DP=False)
        else:
            betas_nodp=0
    return betas_dp, betas_nodp, y_s, X_s, real_betas, all_betas_dp, all_betas_nodp


def extract_R2_iterations(res):
    n=res[-1].shape[1]
    y_hat_dp=np.zeros((res[2].shape[0], n))
    y_hat_nodp=np.zeros((res[2].shape[0], n))
    
    for i in range(res[2].shape[0]):
        y_hat_dp[i]= np.sum(res[0][i]*res[3][i],1)
        y_hat_nodp[i]= np.sum(res[1][i]*res[3][i],1)
    
    res_dp= y_hat_dp-res[2]
    res_nodp= y_hat_nodp-res[2]
    
    y_var= np.var(res[2],1)
    R2_dp=1-np.var(res_dp,1)/y_var
    R2_no_dp= 1-np.var(res_nodp,1)/y_var
    return R2_dp, R2_no_dp

def extract_R2(res, res_nodp=None):
    n=res[2].shape[1]
    y_hat_dp=np.zeros((res[2].shape[0], n))
    y_hat_nodp=np.zeros((res[2].shape[0], n))
    
    for i in range(res[2].shape[0]):
        y_hat_dp[i]= np.sum(res[0][i]*res[3][i],1)
        if res_nodp==None:
            y_hat_nodp[i]= np.sum(res[1][i]*res[3][i],1)
        else:
            y_hat_nodp[i]= np.sum(res_nodp[0][i]*res_nodp[3][i],1)
    
    res_dp= y_hat_dp-res[2]
    res_nodp= y_hat_nodp-res[2]
    
    y_var= np.var(res[2],1)
    R2_dp=1-np.var(res_dp,1)/y_var
    R2_no_dp= 1-np.var(res_nodp,1)/y_var
    return R2_dp, R2_no_dp

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
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
        
def lin_reg(X, y, b=None):
    """
    Compute linear regression and the sum of squared errors. If b is provided, distort the labels by b.
    """
    Z = X.T
    dummy1 = np.dot(Z, X)
    dummy2 = np.dot(Z, y)
    if b is not None:
        if(len(y)==len(b)):
            for x in b:
                if not isinstance(x, (float,int)):
                    break
            dummy2 = np.dot(Z, y-b)
    beta = np.dot(np.linalg.inv(dummy1), dummy2)
    return beta

def get_df(var):
    estimates=np.vstack([resultscor_1[0].T[var], resultscor_3[0].T[var],resultscor_5[0].T[var]]).T
    df_res= pd.DataFrame(estimates)
    df_res.columns=['correlation=.1', 'correlation=.3', 'correlation=.5']
    return df_res