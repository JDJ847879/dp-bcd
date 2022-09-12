from typing import Any, Optional, Sequence

import numpy.typing as npt
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge


def GLM_beta(
    X: npt.NDArray[Any], y: npt.NDArray[Any], model: str, nr_iterations: int = 100
) -> Sequence[float]:
    """Computes the beta according to the GLM net library."""
    # TODO: Unused function?
    # TODO: No regularization is passed or implemented.
    # TODO: Add type hint.
    if model == "linear":
        lr_model = sm.GLM(y, X, family=sm.families.Gaussian(), method="IRLS")
    elif model == "logistic":
        lr_model = sm.GLM(y, X, family=sm.families.Binomial(), method="IRLS")
    res = lr_model.fit(maxiter=nr_iterations)
    beta_glm = res.params
    return beta_glm


def sklearn_beta(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    model: str,
    nr_iterations: int = 100,
    fit_intercept: bool = False,
    tol: Optional[float] = 0.0001,
) -> npt.NDArray[Any]:
    """Computes the beta according to the sklearn library."""
    if model == "linear":
        lr = Ridge(
            fit_intercept=fit_intercept,
            max_iter=nr_iterations,
            alpha=0,
            tol=tol,
        )
        lr.fit(X, y)
        return lr.coef_
    elif model == "logistic":
        lr = LogisticRegression(
            penalty="none",
            fit_intercept=fit_intercept,
            max_iter=nr_iterations,
            tol=tol,
        )
        lr.fit(X, y)
    return lr.coef_[0]


def get_sklearn_model(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    model: str,
    nr_iterations: int = 100,
    fit_intercept: bool = False,
    tol: Optional[float] = 0.0001,
) -> BaseEstimator:
    if model == "linear":
        lr = Ridge(
            fit_intercept=fit_intercept,
            max_iter=nr_iterations,
            alpha=0,
            tol=tol,
        )
        lr.fit(X, y)
        return lr
    elif model == "logistic":
        lr = LogisticRegression(
            penalty="none",
            fit_intercept=fit_intercept,
            max_iter=nr_iterations,
            tol=tol,
        )
        lr.fit(X, y)
        return lr
    else:
        raise RuntimeError("Model has to be either linear or logistic")
