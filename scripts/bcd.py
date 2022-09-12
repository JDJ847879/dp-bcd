import logging
from typing import Any, Dict, List, Optional, Tuple
import os
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/source")
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


def fit_glm(
    players: List[Player],
    outer_iterations: int,
    inner_iterations: int,
    tol: float = 10**-15,
) -> Tuple[npt.NDArray[Any], int, npt.NDArray[Any]]:
    """
    Perform M-party BCD with a number of iterations.

    In this implementation it is manifested that all parties need to have
    the labels.

    :param players: Initialized players that wish to fit the glm of choice on
        their joint (vertically-partitioned) data.
    :param outer_iterations: Number of times that every player is requested to
        update the local model, e.g. number of runs of the block coordinate
        descent algorithm.
    :param inner_iterations: Number of iterations of the IRLS solver performed
        by every player every time she is requested to update the local model.initialize_players
    :param tol: Tolerance level for terminating the training.
    :return: Trained linear regression model.
    """

    # Initialize containers that hold data that is shared amongs players every
    # iteration
    significant_updates = [False] * len(players)
    linear_predictor_local = [player.linear_predictor() for player in players]
    len_beta= len(np.concatenate([p.beta for p in players]))

    all_betas= np.zeros((outer_iterations, len_beta))
    for i in range(outer_iterations):
        logger.debug(f"--- Global iteration: {i+1} ---")

        # Update model per player
        for p, player in enumerate(players):
            # Train model, taking care to correct for the part of eta that is
            # predicted by others
            # this means taking the sum over every column, excluding the pth row. 
            print("player "+ str(p) + " iteration "+str(i))
            linear_predictor_offset = np.sum(
                [_ for pi, _ in enumerate(linear_predictor_local) if pi != p],
                axis=0,
            )
            # so here we attach an estimated beta to the player
            player.train_glm_locally(
                iterations=inner_iterations,
                tol=tol,
                linear_predictor_offset=linear_predictor_offset,
                outer_iterations=outer_iterations,          
            )

            # Log whether the model received a significant update
            significant_updates[p] = player.metadata[-1].significant_update

            # Compute new local eta
            linear_predictor_local[p] = player.linear_predictor()
        beta_global_i = np.concatenate([p.beta for p in players])
        all_betas[i]=beta_global_i
        print(all_betas[i])
        # Terminal condition: no local model received a significant update
        if not any(significant_updates):
            logger.info(
                f"No significant local updates in iteration {i+1}, stopping training."
            )
            break

    beta_global = np.concatenate([p.beta for p in players])
    return beta_global, i + 1, all_betas


def initialize_players(
    model: str,
    X_splitted: List[npt.NDArray[Any]],
    y: npt.NDArray[Any],
    beta_init_splitted: Optional[List[npt.NDArray[Any]]] = None,
    DP: bool = False, 
    gamma: Optional[float] = None, 
    epsilon: Optional[float] = 1,
) -> List[Player]:
    """
    Initialize players for fitting a GLM through theBlock Coordinate Descent
    algorithm.

    In this implementation it is manifested that all parties need to have
    the labels.

    :model: Name of the Generalized Linear Model that is to be solved with the
        Block Coordinate Descent implementation of the Iterative Reweighted
        Least Squares solver.
    :param X_splitted: List of data arrays. Each array represents the local
        data at one of the participating parties.
    :param y: Dependent variables or labels that correspond to the input data
        X.
    :param beta_init_splitted: List local models to initialize IRLS solver. If None,
        the local models are initialized to zero-valued vectors.
    :param DP: whether noise is added to achieve differential privacy.
    :param delta: 
    :return: Initialized player instances.
    """
    # print("gamma")
    # print(gamma)
    delta=1/X_splitted[0].shape[0]
    glm = glm_dict[model]
    irls = IRLSProcedure(irls_iterator=DefaultIRLSIterator(glm=glm, DP=DP, epsilon=epsilon, gamma=gamma, delta=delta))
    if beta_init_splitted is None:
        beta_init_splitted = [None] * len(X_splitted)  # type: ignore

    return [
        Player(
            irls=irls,
            X=X_player,
            y=y.copy(),
            beta_init=beta_init_player,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            DP=DP,
        )
        for (X_player, beta_init_player) in zip(X_splitted, beta_init_splitted)
    ]

