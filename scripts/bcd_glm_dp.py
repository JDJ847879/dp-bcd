# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:36:44 2022

@author: kroessks
"""
# studying the bcd_glm code and repository. 
import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import numpy.typing as npt
from bcd_glm.glm import BaseGLMFamily, BernoulliFamily, GaussianFamily
from bcd_glm.irls import IRLSProcedure
from bcd_glm.irls_iteration import DefaultIRLSIterator
from bcd_glm.player import Player

logger = logging.getLogger(__name__)

glm_dict: Dict[str, BaseGLMFamily] = {
    "linear": GaussianFamily(),
    "logistic": BernoulliFamily(),
}

def minimum_gamma(X: npt.NDArray[Any],
                  y: npt.NDArray[Any]
) -> float:
    '''
    Compute minimum required gamma to achieve epsilon-differential privacy
    by iteratively setting a row to zeros and computing the difference.
    :param X: Numpy array with training data
    :param y: 1-dimensional numpy array with labels.
    :return: float with minimum 
    '''
    n = len(X)
    m = len(X[0])
    
    if( m != np.linalg.matrix_rank(X) ):
        print("Matrix X not of full rank.\n", X)
        return None
    
    beta_nodp = lin_reg(X, y)
    sse = compute_sse(X, y, beta_nodp)
    gamma_minimum = 1.0
    
    for i in range(n):
        X_copy = X.copy()
        for j in range(m):
            X_copy[i][j] = 0
        if np.linalg.matrix_rank(X_copy) == m:
            beta_nodp_copy = lin_reg(X_copy, y)
            sse_copy = compute_sse(X_copy, y, beta_nodp_copy)
            if( (sse_copy/sse) > gamma_minimum ):
                gamma_minimum = (sse_copy/sse)
            elif( (sse/sse_copy) > gamma_minimum ):
                gamma_minimum = (sse/sse_copy)
    
    return np.sqrt(gamma_minimum)



def random_direction(l):
    """
    Return a random vector of length 1 in l dimensions.
    """
    y = [1.0]
    while( len(y) < l):
        # get the first element of y and remove it from y
        x = y.pop(0)       
        theta = 2.0 * np.pi * np.random.uniform()
        y.append( x * np.sin( theta ) )
        y.append( x * np.cos( theta ) )
    
    # then randomly reorder the values that have been generated (could take a sample with replacement)
    z = []
    while( len(y) > 0 ):
        m = np.random.randint(len(y))
        z.append( y.pop(m) )
    return np.asarray(z)

def fit_glm_dp(
    players: List[Player],
    outer_iterations: int,
    inner_iterations: int,
    epsilon: float,
    gamma: float,
    delta: float,
    tol: float = 10**-15,
) -> Tuple[npt.NDArray[Any], int]:
    """
    Perform M-party BCD with a number of iterations with differential privacy
    via objective perturbation.

    In this implementation it is manifested that all parties need to have
    the labels.

    :param players: Initialized players that wish to fit the glm of choice on
        their joint (vertically-partitioned) data.
    :param outer_iterations: Number of times that every player is requested to
        update the local model, e.g. number of runs of the block coordinate
        descent algorithm.
    :param inner_iterations: Number of iterations of the IRLS solver performed
        by every player every time she is requested to update the local model.
    :param epsilon: Differential privacy parameter epsilon (privacy budget).
    :param gamma: The cost of adding differential privacy, directly related to 
        utility.
    :param tol: Tolerance level for terminating the training.
    :return: Trained linear regression model.
    """

    # Initialize containers that hold data that is shared amongs players every
    # iteration
    significant_updates = [False] * len(players)
    linear_predictor_local = [player.linear_predictor() for player in players]

    for i in range(outer_iterations):
        # iterate the regression; every player once in every iteration (as long as
        # the update is significant)
        logger.debug(f"--- Global iteration: {i+1} ---")

        # Update model per player
        for p, player in enumerate(players):
            # Train model, taking care to correct for the part of eta that is
            # predicted by others
            linear_predictor_offset = np.sum(
                [_ for pi, _ in enumerate(linear_predictor_local) if pi != p],
                axis=0,
            )
            
            # Add differential privacy by adding noise to linear_predictor_offset
            xi = gamma * np.sqrt(compute_sse(X_splitted[p], error_local, beta_nodp))
            epsilon_iter = compute_iteration_eps(epsilon, delta, iterations)
            sigma = xi / np.sqrt(epsilon_iter)
            err_l = error_length( sigma, m )
            b = random_direction(m) * err_l
            linear_predictor_offset_noisy=linear_predictor_offset-b
            
            # Determine whether gamma is sufficiently high:
        
            gamma_min = minimum_gamma(X_splitted[p], error_local)
            if( gamma < gamma_min ):
                print("Protocol aborted by party", p, ".\nThe learning goal cannot be met.", gamma, gamma_min)
                return None
        
            player.train_glm_locally(
                iterations=inner_iterations,
                tol=tol,
                linear_predictor_offset=linear_predictor_offset_noisy,
            )

            # Log whether the model received a significant update
            significant_updates[p] = player.metadata[-1].significant_update

            # Compute new local eta
            linear_predictor_local[p] = player.linear_predictor()

        # Terminal condition: no local model received a significant update
        if not any(significant_updates):
            logger.info(
                f"No significant local updates in iteration {i+1}, stopping training."
            )
            break

    beta_global = np.concatenate([p.beta for p in players])
    return beta_global, i + 1, linear_predictor_offset