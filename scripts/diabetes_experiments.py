# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:45:23 2022

@author: kroessks
"""
import os
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/scripts")
import generate_data_synthetic as synthetic
import numpy as np
import matplotlib.pyplot as plt
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/scripts")
from bcd import fit_glm, initialize_players

os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/scripts")
import matplotlib.pyplot as plt
import generate_data_diabetes as diabetes
import pandas as pd
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/data")
X_A = (
    pd.read_csv( "data_csv/diab_administration_15000.csv")
    .drop(columns=["Unnamed: 0"])
)
X_B = (
    pd.read_csv( "data_csv/diab_medication_15000.csv")
    .drop(columns=["Unnamed: 0"])
)

col_names= np.concatenate([X_A.columns, X_B.columns])
m=len(col_names)
# so this data set takes a lot longer! 


def lin_reg(X, y, b=None):
    """
    Compute linear regression and the sum of squared errors. If b is provided, distoart the labels by b.
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

X, y, X_splitted = diabetes.get_data()
betas_init= None
beta_centralized= lin_reg(X, y)
# lets explore the data
cors_diabetes= np.corrcoef(X.T).reshape(-1)
plt.hist(cors_diabetes)
# np.mean(abs(cors_diabetes))

#############################################
players_nodp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=0,
    delta=1,
    gamma=2
)

betas_nodp_diab, it, all_betas_nodp_diab = fit_glm(
    players=players_nodp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
    
)

#######################################################
""" test epsilon """
# so lets start to evaluate the impact of epsilon
reps=10
epsilons= np.array([0,0.1,0.3,0.5,0.8,1,1.5,2.5])
betas_epsilon_diab=np.zeros((reps, epsilons.shape[0], m))

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
            delta=1,
            gamma=1.2
        )
        betas_dp, it, all_betas_dp = fit_glm(
            players=players_dp,
            outer_iterations=5,
            inner_iterations=1,
            tol=10**-15,
        )
        betas_epsilon_diab[j][i]= betas_dp
error_epsilon= np.mean(abs(betas_epsilon_diab-betas_nodp_diab),(2))
mean_error_epsilon= np.mean(error_epsilon,0)

# lets look at the lowest epsilon and plot a histogram
for i in range(m):
    hist= plt.hist(betas_epsilon_diab.T[i][1], label= "DP", alpha=.5, bins=15, density=True)
    plt.hist(all_betas_nodp_diab.T[i], label= "no DP", alpha=.5, bins=hist[1], density=True)
    plt.axvline(beta_centralized[i], color="black")
    plt.legend()
    plt.show()

np.max(betas_epsilon_diab.T.T[1])
np.min(betas_epsilon_diab.T.T[1])


ticks= col_names
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(20, 8)
plt.xlim(xmin=-.5, xmax=m)
plt.ylim(ymin=np.min(betas_epsilon_diab.T.T[1]), ymax=np.max(betas_epsilon_diab.T.T[1]))
x =  [m-X_B.columns,m]
y1 = [np.max(betas_epsilon_diab.T.T[1]),np.max(betas_epsilon_diab.T.T[1])]
plt.fill_between(x, y1, -np.min(betas_epsilon_diab.T.T[1]),
                 facecolor="grey", # The fill color
                 alpha=0.2)          # Transparency of the fill
for i in range(m):
    plt.scatter(np.repeat(i, betas_epsilon_diab.shape[0]), betas_epsilon_diab.T[i][1], alpha=0.1, label= "DP", color="red")
    plt.scatter(np.repeat(i+.2, all_betas_nodp_diab.T[i].shape[0]),all_betas_nodp_diab.T[i], alpha=0.1, label= " no DP", color="blue")
    plt.boxplot(  betas_epsilon_diab.T[i][1], widths=0.04, positions=[i], medianprops=dict(color='black'))
    plt.boxplot( all_betas_nodp_diab.T[i], widths=0.04, positions=[i+.2], medianprops=dict(color='black'))
plt.xticks(range(0,len(ticks)), ticks, rotation='vertical')
plt.title('Coefficients and standard errors for diabetes fire analyses over 500 repetitions')



# Shade the area between y1 and line y=0

# Show the plot
plt.show()

np.repeat([1,0], m, axis=0)    

leg = plt.legend(bbox_to_anchor=(1.05, 1))
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.xticks([])
# you expect higher epsilon is lower privacy and higher utility
for i in range(error_epsilon.shape[1]):
    plt.scatter(np.repeat(epsilons[i], reps), error_epsilon.T[i], alpha=0.2)
plt.boxplot(error_epsilon, widths=0.04, positions=epsilons)

plt.xlabel('Epsilon')
plt.title('Average beta error over epsilon for gamma = 1.2 and delta = 1 with diabetes data (100 repetitions)')
plt.ylabel('Average absolute error beta')

perc_error_epsilon= np.mean(abs(abs(betas_epsilon_diab-betas_nodp_diab)/betas_nodp_diab),(2))
mean_perc_error_epsilon= np.mean(perc_error_epsilon,0)

# you expect higher epsilon is lower privacy and higher utility
for i in range(perc_error_epsilon.shape[1]):
    plt.scatter(np.repeat(epsilons[i], reps), perc_error_epsilon.T[i], alpha=0.2)
plt.boxplot(perc_error_epsilon, widths=0.04, positions=epsilons)

plt.xlabel('Epsilon')
plt.title('Percentual beta error over epsilon for gamma = 1.2 and delta = 1 with diabetes data (100 repetitions)')
plt.ylabel('Percentual absolute error beta')
 
""" test gamma """

# so lets start to evaluate the impact of gamma
reps=2
gammas= np.array([1.15,1.25,1.5,1.8,2,2.5, 3])
betas_gamma_diab=np.zeros((reps, gammas.shape[0], m))

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
            delta=1,
            gamma=g
        )
        
        
        betas_dp, it, all_betas_dp = fit_glm(
            players=players_dp,
            outer_iterations=5,
            inner_iterations=1,
            tol=10**-15,
        )
        betas_gamma_diab[j][i]= betas_dp

# is epsilon 0 gelijk aan no dp?
# because gamma is nonzero
error_gamma= np.mean(abs(betas_gamma_diab-betas_nodp_diab),(2))
mean_error_gamma= np.mean(error_gamma,0)

betas_gamma_diab.shape



# you expect higher gamma is lower privacy and higher utility
    
for i in range(error_gamma.shape[1]):
    plt.scatter(np.repeat(gammas[i], reps), error_gamma.T[i], alpha=0.5)
plt.boxplot(error_gamma, widths=0.04, positions=gammas)

plt.xlabel('gamma')
plt.title('Average beta error over gamma for epsilon = 1 and delta = 1 with diabetes data (100 repetitions)')
plt.ylabel('Average absolute error beta')
plt.xscale('log')

perc_error_gamma= np.mean(abs(abs(betas_gamma_diab-betas_nodp_diab)/betas_nodp_diab),(2))
mean_perc_error_gamma= np.mean(perc_error_gamma,0)

# you expect higher gamma is lower privacy and higher utility
    
for i in range(perc_error_gamma.shape[1]):
    plt.scatter(np.repeat(gammas[i], reps), perc_error_gamma.T[i], alpha=0.2)
plt.boxplot(perc_error_gamma, widths=0.04, positions=gammas)

plt.xlabel('gamma')
plt.title('Percentual beta error over gamma for epsilon = 1 and delta = 1 with diabetes data (100 repetitions)')
plt.ylabel('Percentual absolute error beta')
plt.xscale('log')






# why is the error length smaller for high epsilon?? It does match with the pictures we see
betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1000,
    delta=1,
    gamma=2
)


betas_dp4, it, all_betas_dp4 = fit_glm(
    players=players_dp4,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
    
    epsilon=1000,
    delta=1,
    gamma=2
)

plt.plot(all_betas_dp4)







betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1000,
    delta=1,
    gamma=200
)


betas_dp4, it, all_betas_dp4 = fit_glm(
    players=players_dp4,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)

plt.plot(all_betas_dp4)




betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1,
    delta=1,
    gamma=500
)


betas_dp4, it, all_betas_dp4 = fit_glm(
    players=players_dp4,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)

plt.plot(all_betas_dp4)

betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=500,
    delta=1,
    gamma=2
)


betas_dp4, it, all_betas_dp4 = fit_glm(
    players=players_dp4,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)




# compute R2 









betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1,
    delta=1,
    gamma=100
)


betas_dp4, it, all_betas_dp4 = fit_glm(
    players=players_dp4,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)

plt.plot(all_betas_dp4)





betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=100,
    delta=1,
    gamma=100
)


betas_dp4, it, all_betas_dp4 = fit_glm(
    players=players_dp4,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)

plt.plot(all_betas_dp4)




# we can do a quick R2 test?



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

R2(y,np.dot(betas_dp4,X.T)) # -587.9583543443293












betas_init= None
players_dp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=2,
    delta=1,
    gamma=1.2
)


betas_dp, it, all_betas_dp = fit_glm(
    players=players_dp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)

R2(y,np.dot(betas_nodp_diab,X.T)) #0.0723196328281206
# it was already very small
R2(y,np.dot(betas_dp,X.T)) #-0.m43620920536449
1-1.2**5

R2(y,np.dot(betas_dp,X.T)) #-0.m43620920536449



betas_init= None
players_dp4 = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1,
    delta=1,
    gamma=1.5
)




players_dp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=2,
    delta=1,
    gamma=1.2
)

betas_dp, it, all_betas_dp = fit_glm(
    players=players_dp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,

)




players = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
)
betas, it = fit_glm(
    players=players,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
)

# irls = IRLSProcedure(irls_iterator=DefaultIRLSIterator(glm=glm))
# array([ 0.11665507, -0.01309001,  1.08311m2,  0.70287547,  1.30846334,
#         0.5004933 ,  0.21822131,  0.551755m,  0.70225826,  0.31395616,
#         2.44204902,  1.31041287, -0.64087926,  0.4095089 ,  0.40649118,
#         0.31104463,  0.24253646,  0.28122522,  0.14693208,  0.13797547,
#        -0.02825601,  0.120m655,  0.0126564 ,  0.05834396,  0.2571665 ,
#        -0.42605224, -0.06545001])

# # X.shape is 517 by m
# len(players)
# players[0].X.shape # 517, 23
# players[1].X.shape # 517, 4



# so it runs now...
# lets see the results compared to when we dont have 



betas_init= None
players_dp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
)


betas_dp, it = fit_glm(
    players=players_dp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
    
    epsilon=1,
    delta=1,
    gamma=10
)





betas_init= None
players_dp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    epsilon=1,
    delta=1,
    gamma=10
)


betas_dp, it = fit_glm(
    players=players_dp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
    
    epsilon=1,
    delta=1,
    gamma=10
)

betas_init= None
players_dp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1,
    delta=1,
    gamma=2
)


betas_dp, it = fit_glm(
    players=players_dp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
    
    epsilon=1,
    delta=1,
    gamma=2
)

# array([ 0.14899978, -0.06771107,  1.76469207,  0.55635253,  1.40070481,
#         0.43424026,  0.10473263,  0.6414831 ,  0.51866083, -2.97001592,
#         2.06850574,  1.71819592, -1.62526592,  0.46804001,  0.40338075,
#         0.30268462,  0.24989247,  0.4288552 ,  0.10077382, -0.01266035,
#        -0.30932125,  0.13667504,  0.04619985, -0.15005818,  0.45391454,
#        -0.48838219, -0.01336229])


betas_init= None
players_dp = initialize_players(
    model="linear",
    X_splitted=X_splitted,
    y=y,
    beta_init_splitted=betas_init,
    DP=True, 
    
    epsilon=1000,
    delta=1,
    gamma=2
)


betas_dp, it = fit_glm(
    players=players_dp,
    outer_iterations=5,
    inner_iterations=1,
    tol=10**-15,
    
    epsilon=1000,
    delta=1,
    gamma=2
)


# Out[251]: 
# array([ 0.10223943,  0.01268085,  1.3337143 ,  0.62072866,  1.32643808,
#         0.39776502,  0.48963409,  0.41403007,  0.48100169, -0.68877179,
#         2.66465994,  1.28196558, -0.94552939,  0.64665887,  0.45030101,
#         0.49673172,  0.20461115,  0.32410637,  0.35002931,  0.04948635,
#        -0.06200745,  0.09384261, -0.0205682 , -0.06141646,  0.26800832,
#        -0.34106224, -0.07801042])


