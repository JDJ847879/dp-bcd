from typing import Any, List, Tuple
import numpy as np
import numpy.typing as npt
import os
file_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_wd+"/../source")
from bcd_glm.glm import BernoulliFamily
from sklearn.datasets import make_classification, make_regression
from typing import Any, Optional

# small modification on original code by Bart Kamphorst.

def get_correlation_matrix(
    n_features: int, c: float = 0
) -> npt.NDArray[Any]:
    matrix = np.identity(n_features)
    for i in range(n_features):
        for j in range(i):
            matrix[i][j] = matrix[j][i] = c
    return matrix


def generate_synthetic_data(
    corr_mat: npt.NDArray[Any],
    n_features: int,
    betas: npt.NDArray[Any],
    n_sample: int = 1000,
    outcome: str = "gaussian",
    random_seed: int = 0,
    R_squared: Optional[float] = .3,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Generate synthetic data for a given correlation matrix. For Gaussian data an R squared can be specified and the residuals will be generated to match this value.

    Methodology is discribed [in this blogpost](https://mlisi.xyz/post/simulating-correlated-variables-with-the-cholesky-factorization/ ).
    """

    np.random.seed(random_seed)
    X_raw = np.random.normal(loc=0, scale=1, size=(n_sample, n_features))

    chol = np.linalg.cholesky(corr_mat)
    X = np.matmul(X_raw, chol.T)
    y_pred = np.matmul(X, betas)
    
    if outcome == "gaussian":
        # compute variance of y 
        y_var= np.var(y_pred)/R_squared 
        # use residual variance required to obtain specified R squared in expectation.
        eps = np.random.normal(loc=0, scale=np.sqrt(y_var-np.var(y_pred)), size=n_sample)
        # for gaussian labels (y), y=eta
        y = y_pred + eps
        
        
    elif outcome == "binomial":
        eps = np.random.normal(loc=0, scale=1, size=n_sample)
        eta= y_pred + eps
        link_inv_eta = BernoulliFamily().inverse_link_function(eta)
        y = np.array(
            [
                np.random.binomial(n=1, p=p, size=1)[0]
                for p in np.array(link_inv_eta).flatten()
            ]
        )
    else:
        raise ValueError("outcome has to be either 'gaussian' or 'binomial'.")
    return X, y

def generate_synthetic_data(
    corr_mat: npt.NDArray[Any],
    n_features: int,
    betas: npt.NDArray[Any],
    n_sample: int = 1000,
    outcome: str = "gaussian",
    random_seed: int = 0,
    R_squared: Optional[float] = .3,
    round_x: bool = False,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Generate synthetic data for a given correlation matrix. For Gaussian data an R squared can be specified and the residuals will be generated to match this value.

    Methodology is discribed [in this blogpost](https://mlisi.xyz/post/simulating-correlated-variables-with-the-cholesky-factorization/ ).
    """

    np.random.seed(random_seed)
    X_raw = np.random.normal(loc=0, scale=1, size=(n_sample, n_features))

    chol = np.linalg.cholesky(corr_mat)
    X = np.matmul(X_raw, chol.T)
    if round_x:
        X=np.round(X,1)
    y_pred = np.matmul(X, betas)
    
    if outcome == "gaussian":
        # compute variance of y 
        y_var= np.var(y_pred)/R_squared 
        # use residual variance required to obtain specified R squared in expectation.
        eps = np.random.normal(loc=0, scale=np.sqrt(y_var-np.var(y_pred)), size=n_sample)
        # for gaussian labels (y), y=eta
        y = y_pred + eps
        
        
    elif outcome == "binomial":
        eps = np.random.normal(loc=0, scale=1, size=n_sample)
        eta= y_pred + eps
        link_inv_eta = BernoulliFamily().inverse_link_function(eta)
        y = np.array(
            [
                np.random.binomial(n=1, p=p, size=1)[0]
                for p in np.array(link_inv_eta).flatten()
            ]
        )
    else:
        raise ValueError("outcome has to be either 'gaussian' or 'binomial'.")
    return X, y


def get_data_sklearn(
    tot_features: int, model: str, correlated: bool
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    if model == "linear":
        return get_data_linear(tot_features=tot_features, correlated=correlated)
    elif model == "logistic":
        X, y = get_data_logistic(
            tot_features=tot_features,
            min_n_informative=int(tot_features / 2),
            correlated=correlated,
        )
        return X, y, np.ndarray([])
    else:
        raise ValueError("model has to be either 'linear' or 'logistic'.")


def get_data_linear(
    tot_features: int,
    n_samples: int = 1000,
    correlated: bool = False,
    noise: float = 4.0,
    random_state: int = 0,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    # we use the default for n_informative that is 10; but would be nicer to understand better the impact of this choice
    if not correlated:
        effective_rank = None  # The input set is well conditioned, centered and gaussian with unit variance.
    else:
        # The approximate number of singular vectors required to
        # explain most of the input data by linear combinations.
        # Using this kind of singular spectrum in the input allows
        # the generator to reproduce the correlations often observed in practice.
        effective_rank = tot_features // 3
    X, y, betas = make_regression(
        n_samples=n_samples,
        n_features=tot_features,
        # n_informative=tot_features,
        effective_rank=effective_rank,
        random_state=random_state,
        noise=noise,
        coef=True,
    )
    return X, y, betas


def get_data_logistic(
    tot_features: int,
    min_n_informative: int,
    n_samples: int = 1000,
    correlated: bool = False,
    noise: float = 4.0,
    random_state: int = 0,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # X horizontally stacks features in the following order: the primary n_informative features,
    # followed by n_redundant linear combinations of the informative features, followed by n_repeated duplicates,
    # drawn randomly with replacement from the informative and redundant features.
    # The remaining features are filled with random noise. Thus, without shuffling,
    # all useful features are contained in the columns X[:, :n_informative + n_redundant + n_repeated].

    # m parties --> I want at least m informative features.
    # For the correlated case I also want some redundant features.

    # m in (1,10)
    # tot features = 20
    # informative_features = 10 in the correlated case

    if correlated:
        n_informative = min_n_informative
        n_redundant = tot_features - min_n_informative
    else:
        n_informative = tot_features
        n_redundant = 0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=tot_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        shuffle=True,
        random_state=random_state,
    )
    return X, y


def split_data_parties(
    X: npt.NDArray[Any], feature_per_party: List[int]
) -> List[npt.NDArray[Any]]:
    x_split = []
    feature_from = 0
    feature_to = 0
    for n in feature_per_party:
        feature_to += n
        x_split.append(X[:, feature_from:feature_to].copy())
        feature_from += n
    return x_split


def test_generate_synthetic_data(n: int = 10, c:float=0) -> None:
    corr = get_correlation_matrix(n, c)
    X, _ = generate_synthetic_data(corr, n, np.asarray(list(range(n))))
    print(np.amax(abs(np.corrcoef(X.T) - corr)))


if __name__ == "__main__":
    for n in [2, 3, 5, 8, 10, 100, 1000, 10000]:
        test_generate_synthetic_data(n)


