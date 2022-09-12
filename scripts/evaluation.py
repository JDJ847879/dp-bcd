from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/source")

from bcd_glm.glm.logistic import sigmoid
from sklearn.metrics import accuracy_score, r2_score


def accuracy(X: npt.NDArray[Any], y: npt.NDArray[Any], beta: npt.NDArray[Any]) -> float:
    """Calculates the accuracy of the model described by beta."""
    z = sigmoid(np.dot(X, beta))
    y_pred: npt.NDArray[Any] = z >= 0.5
    return accuracy_score(y, y_pred)


def r2(X: npt.NDArray[Any], y: npt.NDArray[Any], beta: npt.NDArray[Any]) -> float:
    """Calculates the R2 of the model described by beta."""
    score = r2_score(y, np.dot(X, beta))
    return score


def plot_r2s(X: npt.NDArray[Any], y: npt.NDArray[Any], betas: npt.NDArray[Any]) -> None:
    """Plot the R2 score for every model described in betas."""
    r2s = []
    for beta in betas:
        ev = r2(X, y, beta[0])
        r2s.append(ev)

    _, (ax1, ax2) = plt.subplots(1, 2)
    for beta in betas:
        ax1.scatter(range(len(beta[0])), beta[0], label=beta[1])
    ax1.legend()

    ax2.scatter(range(len(r2s)), r2s)
    ax2.plot(r2s)
    ax2.set_xticks(range(len(betas)))
    ax2.set_xticklabels(betas[:, 1], rotation=45, fontsize=12)
    ax2.set_ylim([0, 1])
    ax1.set_title("The resulting betas for different models")
    ax2.set_title("The final training R2 score for different models")
    plt.show()


def plot_accuracies(
    X: npt.NDArray[Any], y: npt.NDArray[Any], betas: npt.NDArray[Any]
) -> None:
    """Plot the accuracy for every model described in betas."""
    accs = []
    for beta in betas:
        ev = accuracy(X, y, beta[0])
        accs.append(ev)

    _, (ax1, ax2) = plt.subplots(1, 2)
    for beta in betas:
        ax1.scatter(range(len(beta[0])), beta[0], label=beta[1])
    ax1.legend()

    ax2.scatter(range(len(accs)), accs)
    ax2.plot(accs)
    ax2.set_xticks(range(len(betas)))
    ax2.set_yticks(accs)
    ax2.set_xticklabels(betas[:, 1], rotation=45, fontsize=12)
    ax1.set_title("The resulting betas for different models")
    ax2.set_title("The final training accuracy for different models")
    # ax2.set_ylim([0.5, 1])
    plt.show()
