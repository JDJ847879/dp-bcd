import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import evaluation
import os
os.chdir("C:/Users/kroessks/Documents/Projects/sERP/bcd-glm/scripts")

import generate_data_diabetes as diabetes
import generate_data_forestfires as forest
import generate_data_synthetic as synthetic
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from experiment_parser import parse_arguments
from run import calculate_betas_and_performance

BASEDIR = "C:/Users/kroessks/Documents/Projects/sERP/bcd-glm"
OUTCOMEDIR = BASEDIR / "outcome"
LOGDIR = OUTCOMEDIR

logger = logging.getLogger("bcd_glm")
logger.setLevel(level=logging.DEBUG)
hl = logging.FileHandler(LOGDIR / "experiment.log", mode="w")
hl.setLevel(level=logging.DEBUG)
logger.addHandler(hl)

class Experiment:
    """
    Class Experiment is a wrapper to call faster the calculate_betas function in the run.py script, and store the result to do further analysis, or create plots.
    It is initialized passing a sequence of strings corresponding to the arguments defined in parser in the run.py script
    Minimum example to run an experiment and print the betas obtained from the run.py script:
        ex = Experiment(["--model", "logistic", "--central-iter", "1000", "--tol", "0.000000000001"])
        X, y = get_data()  # get your training data and labels
        X_splitted = split_data(X_splitted)
        ex.set_data_and_run(X, y, X_splitted)
        print(ex.get_betas())

    The class has also functions to return a (nice) plot with the betas and to plot the scores (R2 or accuracy) for all the model trained

    """

    def __init__(self, cmdline_options: Sequence[str]) -> None:
        self.args = parse_arguments(cmdline_options)
        self.X: np.ndarray
        self.y: np.ndarray
        self.X_splitted: List[npt.NDArray[Any]]
        self.betas: Dict[str, npt.NDArray[Any]]
        self.ex_time: Dict[str, float]
        self.iterations: Dict[str, float]

    def set_data_and_run(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        X_splitted: List[npt.NDArray[Any]],
        betas_init: Optional[List[npt.NDArray[Any]]] = None,
    ) -> None:
        self.X, self.y, self.X_splitted = X, y, X_splitted
        self.betas, self.ex_time, self.iterations = calculate_betas_and_performance(
            X=self.X,
            y=self.y,
            X_splitted=self.X_splitted,
            betas_init=betas_init,
            args=self.args,
        )

    def get_betas(self) -> Dict[str, npt.NDArray[Any]]:
        return self.betas

    def get_ex_time(self) -> Dict[str, float]:
        return self.ex_time

    def get_iterations(self) -> Dict[str, float]:
        return self.iterations

    def get_r2_scores(
        self, betas: Optional[Dict[str, npt.NDArray[Any]]] = None
    ) -> Dict[str, float]:
        if self.args.model != "linear":
            raise ValueError("r2 scores should be calculated only for linear models")
        r2_scores = {}
        if betas is None:
            betas = self.betas
        for k, beta in betas.items():
            r2_scores[k] = evaluation.r2(self.X, self.y, beta)
        return r2_scores

    def get_accuracy_scores(
        self, betas: Optional[Dict[str, npt.NDArray[Any]]] = None
    ) -> Dict[str, float]:
        if self.args.model != "logistic":
            raise ValueError(
                "Accuracy scores should be calculated only for linear models"
            )
        accuracy_scores = {}
        if betas is None:
            betas = self.betas
        for k, beta in betas.items():
            accuracy_scores[k] = evaluation.accuracy(self.X, self.y, beta)
        return accuracy_scores

    def plot_betas_sns(
        self, betas: Optional[Dict[str, npt.NDArray[Any]]] = None
    ) -> plt.Figure:
        if betas is None:
            betas = self.betas
        b = len(betas[list(betas.keys())[0]])
        h = math.floor(b / 3)
        fig, ax = plt.subplots(figsize=(b, h), dpi=80)

        df = pd.DataFrame.from_dict(betas, orient="index")
        df["model"] = df.index
        df.reset_index(drop=True, inplace=True)
        df = pd.melt(df, id_vars=["model"], var_name="beta_index")

        sns.stripplot(
            df.beta_index,
            df.value,
            jitter=True,
            dodge=True,
            hue=df.model,
            size=15,
            ax=ax,
            linewidth=0.5,
        )
        ax.legend(fontsize=20)
        return fig

    def plot_scores(
        self, betas: Optional[Dict[str, npt.NDArray[Any]]] = None
    ) -> plt.Figure:
        if betas is None:
            betas = self.betas

        b = len(betas[list(betas.keys())[0]])
        h = math.floor(b / 3)
        fig, ax = plt.subplots(figsize=(b, h), dpi=80)

        scores = {}
        if self.args.model == "linear":
            scores = self.get_r2_scores(betas)
        elif self.args.model == "logistic":
            scores = self.get_accuracy_scores(betas)
        else:
            logger.warning("Model should be either linear or logistic")
        ax.plot(range(len(scores)), scores.values())
        ax.plot(scores.values())
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(scores.keys(), rotation=45, fontsize=12)
        ax.set_ylim([0, 1])
        return fig


# Utils
def create_args_list(model: str) -> List[str]:
    return ["--model", model]


def experiment_forest(results_folder: Path) -> None:
    """
    We run the forest fires experiment as in the van Kesteren paper.
    The function saves the plot of the different betas and the csv of the betas in the provided folder
    """

    ex = Experiment(create_args_list("linear"))
    X, y, X_splitted = forest.get_data()
    ex.set_data_and_run(X=X, y=y, X_splitted=X_splitted)

    # collect the betas obtained by our experiment and in the Van Kesteren paper
    betas = ex.get_betas().copy()
    betas["baseline"] = pd.read_csv(
        BASEDIR / "data/data_csv/forest_fire_resultfull.csv"
    )["coef"]

    # save the plot of betas
    FIG_PATH = results_folder / "ex_forest_linear_betas.png"
    fig = ex.plot_betas_sns(betas)
    fig.savefig(FIG_PATH, dpi=fig.dpi)
    logger.info(f"Model coefficients for all solvers plotted to {FIG_PATH}.")

    CSV_PATH = results_folder / "ex_forest_linear_betas.csv"
    df = pd.DataFrame()
    df["IRLS_central"] = betas["IRLS_central"]
    df["IRLS_bcd"] = betas["IRLS_bcd"]
    df["Baseline_paper"] = betas["baseline"]
    df.to_csv(CSV_PATH)
    logger.info(f"Model coefficients for all solvers saved to {CSV_PATH}.")


def experiment_diabetes(results_folder: Path) -> None:
    """
    We run the diabetes experiment as in the van Kesteren paper.
    The function saves the plot of the different betas and the csv of the betas in the provided folder
    """
    ex = Experiment(create_args_list("logistic"))
    X, y, X_splitted = diabetes.get_data()
    ex.set_data_and_run(X=X, y=y, X_splitted=X_splitted)

    # collect the betas obtained by our experiment and in the Van Kesteren paper
    betas = ex.get_betas().copy()
    betas["baseline"] = pd.read_csv(
        BASEDIR / "data/data_csv/diab_15000_resultfull.csv"
    )["coef"]

    # save the plot of betas
    FIG_PATH = results_folder / "ex_diabetes_logistic_betas.png"
    fig = ex.plot_betas_sns(betas)
    fig.savefig(FIG_PATH, dpi=fig.dpi)
    logger.info(f"Model coefficients for all solvers plotted to {FIG_PATH}.")

    CSV_PATH = results_folder / "ex_diabetes_logistic_betas.csv"
    df = pd.DataFrame()
    df["IRLS_central"] = betas["IRLS_central"]
    df["IRLS_bcd"] = betas["IRLS_bcd"]
    df["Baseline_paper"] = betas["baseline"]
    df.to_csv(CSV_PATH)
    logger.info(f"Model coefficients for all solvers saved to {CSV_PATH}.")


def experiment_synthetic(
    results_folder: Path,
    model: str = "linear",
    correlated: bool = False,
    generation_data: str = "choleski",
    max_parties: int = 10,
    tot_features: int = 20,
    repetitions: int = 20,
) -> None:
    """
    Solve the linear or logistic problem using synthetic data, assuming the
    data is distributed among a different number of parties.

    The function saves:
    - the graph of the correlation values of the generated data
    - a csv containing for each number of parties, the real betas of the
        model, the betas obtained with sklearn, with the IRLS_central and with
        IRLS_bcd, the r2 or accuracy scores and the number of iterations
        required from the IRLS_bcd
    - a csv file containing the time of execution of all the experiments
        (so (max_parties - 2)* repetitions rows)
    - the plot of #parties vs average execution time

    :param results_folder: Folder in which to store the results.
    :param model: "linear" or "logistic".
    :param correlated: Whether we want to generate data with high or low
        correlation.
    :param generation_data: If 'sklearn' the sklearn.make_regression function
        will be used to generate the data; with different effective_rank
        depending on the required correlation. If 'choleski' the data is
        generated so to have a correlation close to 0.1 or 0.5 outside the
        diagonal.
    :param max_parties: The experiment will be run for a number of parties
        from 2 to max_parties. The features will be split in equal number
        among the parties, with the last one possibly ending having more
        features.
    :param tot_features: Number of features in total considering all parties.
    :param repetitions: Number of times that the experiment is repeated (for
        every number of parties).
    """

    correlation = "high_corr" if correlated else "low_corr"

    ex = Experiment(create_args_list(model))
    X: np.ndarray
    y: np.ndarray
    real_betas: np.ndarray

    # Generation of data
    if generation_data == "sklearn":
        X, y, real_betas = synthetic.get_data_sklearn(
            tot_features, model, correlated=correlated
        )

    elif generation_data == "choleski":
        step = 2 / (tot_features - 1)
        real_betas = (
            np.arange(-1, 1 + step, step) / tot_features * 10
        )  # this definition of betas is derived from the R code of the van Kesteren paper
        desired_corr_mat = synthetic.get_correlation_matrix(
            n_features=tot_features, correlated=correlated
        )
        outcome = "gaussian" if (model == "linear") else "binomial"
        X, y = synthetic.generate_synthetic_data(
            desired_corr_mat, n_features=tot_features, betas=real_betas, outcome=outcome
        )

    # generate and save plot of the correlation matrix
    fig_corr_matrix = get_correlation_matrix_fig(X)
    correlation_matrix_fig = "ex_syn_" + model + "_" + correlation + "_matrix.png"
    fig_corr_matrix.savefig(
        results_folder / correlation_matrix_fig, dpi=fig_corr_matrix.dpi
    )

    # define possible splitting of the data among parties
    features_splitting = []
    for m in range(2, max_parties + 1):
        d = tot_features // m
        r = tot_features % m
        features_splitting.append([d] * (m - 1) + [d + r])

    models_results_rows = []
    model_time_performance_rows = []

    for features_per_party in features_splitting:

        X_splitted = synthetic.split_data_parties(X, features_per_party)
        ex.set_data_and_run(X=X, y=y, X_splitted=X_splitted)

        # betas
        betas = ex.get_betas().copy()
        if model == "linear":
            betas["real"] = real_betas

        # save the plot of betas for this specific splitting of the features
        betas_fig_name = (
            "ex_syn_"
            + model
            + "_"
            + str(len(features_per_party))
            + "_"
            + correlation
            + "_betas.png"
        )
        fig = ex.plot_betas_sns(betas)
        fig.savefig(results_folder / betas_fig_name, dpi=fig.dpi)

        # store the results of the models, i.e. all the betas, the r2scores, and the number of iterations for the bcd
        results: Dict[str, Union[int, float]] = {
            "tot_features": tot_features,
            "parties": len(features_per_party),
        }
        for i in range(tot_features):
            if model == "linear":
                results["beta_" + str(i) + "_real"] = betas["real"][i]
            results["beta_" + str(i) + "_sklearn"] = betas["sklearn"][i]
            results["beta_" + str(i) + "_IRLS_central"] = betas["IRLS_central"][i]
            results["beta_" + str(i) + "_IRLS_bcd"] = betas["IRLS_bcd"][i]
        scores = ex.get_r2_scores() if (model == "linear") else ex.get_accuracy_scores()
        iterations = ex.get_iterations()
        results["score_sklearn"] = scores["sklearn"]
        results["score_central"] = scores["IRLS_central"]
        results["score_bcd"] = scores["IRLS_bcd"]
        results["bcd_iterations"] = iterations["IRLS_bcd"]

        models_results_rows.append(results)

        # store the time of execution for the same experiment repeted #repetitions times
        execution_time = ex.get_ex_time()
        model_time_performance_rows.append(
            {
                "tot_features": tot_features,
                "parties": len(features_per_party),
                "time_central": execution_time["IRLS_central"],
                "time": execution_time["IRLS_bcd"],
            }
        )

        for i in range(0, repetitions - 1):
            ex.set_data_and_run(X, y, X_splitted)
            execution_time = ex.get_ex_time()
            model_time_performance_rows.append(
                {
                    "tot_features": tot_features,
                    "parties": len(features_per_party),
                    "time_central": execution_time["IRLS_central"],
                    "time": execution_time["IRLS_bcd"],
                }
            )

        print(f"Done with {features_per_party} ")

    # save to csv the models results
    df_models_results = pd.DataFrame(models_results_rows)
    CSV_PATH = results_folder / f"ex_syn_{model}_{correlation}_models_results.csv"
    df_models_results.to_csv(CSV_PATH)
    logger.info(f"Model coefficients for all solvers saved to {CSV_PATH}.")

    # save to csv and the plot of the performances in time
    df_time_performances = pd.DataFrame(model_time_performance_rows)
    MODELS_TIME_PERFORMANCE_CSV_PATH = results_folder / (
        "ex_syn_" + model + "_" + correlation + "_time_performance.csv"
    )
    MODELS_TIME_PERFORMANCE_FIG_PATH = results_folder / (
        "ex_syn_" + model + "_" + correlation + "_time_performance.png"
    )
    df_time_performances.to_csv(MODELS_TIME_PERFORMANCE_CSV_PATH)
    logger.info(f"Experiment timings for all solvers saved to {CSV_PATH}.")
    time_fig = get_time_plot(df_time_performances)
    time_fig.savefig(MODELS_TIME_PERFORMANCE_FIG_PATH, dpi=time_fig.dpi)
    logger.info(
        f"Experiment timings for all solvers plotted to {MODELS_TIME_PERFORMANCE_FIG_PATH}."
    )


def get_time_plot(df: pd.DataFrame) -> plt.Figure:
    average_time_central = []
    average_time_bcd = []
    for m in sorted(df["parties"].unique()):
        mean_central = df[df["parties"] == m]["time_central"].mean()
        mean_bcd = df[df["parties"] == m]["time"].mean()
        average_time_central.append(mean_central)
        average_time_bcd.append(mean_bcd)
    parties = range(2, len(average_time_bcd) + 2)
    fig = plt.figure(figsize=(19, 15))
    plt.plot(parties, average_time_central, label="IRLS_central")
    plt.plot(parties, average_time_bcd, label="IRLS_bcd")
    plt.xlabel("Number of parties", fontsize=16)
    plt.ylabel("Time of total training", fontsize=16)
    plt.legend(prop={"size": 16})
    return fig


def get_correlation_matrix_fig(X: npt.NDArray[Any]) -> plt.Figure:
    df = pd.DataFrame(X)
    fig = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=fig.number)
    plt.xticks(
        range(df.select_dtypes(["number"]).shape[1]),
        df.select_dtypes(["number"]).columns,
        fontsize=14,
        rotation=45,
    )
    plt.yticks(
        range(df.select_dtypes(["number"]).shape[1]),
        df.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    plt.clim(-1, 1)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)
    return fig


if __name__ == "__main__":
    experiment_forest(OUTCOMEDIR / "forests_results")

    # experiment_synthetic(OUTCOMEDIR/"synthetic_chol_linear", model="linear", correlated=False, generation_data="choleski")
    # experiment_synthetic(OUTCOMEDIR/"synthetic_chol_linear", model="linear", correlated=True, generation_data="choleski")
    # experiment_synthetic(OUTCOMEDIR/"synthetic_chol_logistic", model="logistic", correlated=False, generation_data="choleski")
    # experiment_synthetic(OUTCOMEDIR/"synthetic_chol_logistic", model="logistic", correlated=True, generation_data="choleski")
    # experiment_synthetic(
    #     OUTCOMEDIR / "synthetic_sklearn_linear",
    #     model="linear",
    #     correlated=False,
    #     generation_data="sklearn",
    # )
    # experiment_synthetic(
    #     OUTCOMEDIR / "synthetic_sklearn_linear",
    #     model="linear",
    #     correlated=True,
    #     generation_data="sklearn",
    # )
    # experiment_synthetic(
    #     OUTCOMEDIR / "synthetic_sklearn_logistic",
    #     model="logistic",
    #     correlated=False,
    #     generation_data="sklearn",
    # )
    # experiment_synthetic(
    #     OUTCOMEDIR / "synthetic_sklearn_logistic",
    #     model="logistic",
    #     correlated=True,
    #     generation_data="sklearn",
    # )
