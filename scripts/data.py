from pathlib import Path
from typing import Any, List, Optional, Tuple

import generate_data_diabetes as diabetes_big
import generate_data_forestfires as forest
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

DATADIR = Path(__file__).parent.parent / "data"

def dataset(
    dataset_name: str,
    centering: bool = False,
    add_intercept: bool = False,
    n: int = 100,
    m: int = 10,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Return dataset for testing.

    :param dataset_name: Name of the dataset.
    :param centering: Center the values of every column.
    :param add_intercept: Append a column of ones to the dataset.
    :param n: Number of samples in the dataset. Used for randomly generated datasets only.
    :param m: Number of attributes in the dataset. Used for randomly generated datasets only.
    :return: Tuple of dataset and target variables.
    """

    if dataset_name == "random_linear":
        X, y, _ = generate_data(n, m)
    elif dataset_name == "random_logistic":
        X, y, _ = generate_data(n, m)
        y = y > np.mean(y)
    elif dataset_name == "iris":
        iris = datasets.load_iris()
        X = iris.data[:100]
        y = iris.target[:100]
    elif dataset_name == "wine":
        wine = datasets.load_wine()
        X = wine.data[:130, :8]
        y = wine.target[:130]
    elif dataset_name == "diabetes":
        diabetes = pd.read_csv(DATADIR / "diabetes.csv")
        X = diabetes[diabetes.columns[:-1]].values
        y = diabetes[diabetes.columns[-1]].values
    elif dataset_name == "fish":
        data = pd.read_csv(DATADIR / "fish.csv")
        y = np.array(data["Weight"])
        X = (
            data.join(pd.get_dummies(data["Species"]))
            .drop(["Species", "Weight"], axis=1)
            .values
        )
    elif dataset_name == "forestfires":
        X, y, _ = forest.get_data()
    elif dataset_name == "diabetes_big":
        X, y, _ = diabetes_big.get_data()
    if centering:
        meanX = np.mean(X, axis=0)
        X -= meanX
    if add_intercept:
        X = np.append(np.ones([len(X), 1]), X, axis=1)
    return X, y


def import_data(
    name: str, directory: str, nr_samples: int, nr_features: int
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Import data from collection of datasets.

    :param name: Name of the dataset.
    :param directory: The overall directory of the folder. I had problems with importing.
    :return: matrix with data, vector of labels.
    """
    if name == "heart":
        heart_data = pd.read_csv(Path(directory) / "/data/heart.csv")
        X = heart_data[heart_data.columns[:-1]].values
        y = heart_data[heart_data.columns[-1]].values
    elif name == "forestfires":
        forest_data = pd.read_csv(Path(directory) / "/data/forestfires.csv")
        le = LabelEncoder()
        forest_data["month"] = le.fit_transform(forest_data["month"])
        forest_data["day"] = le.fit_transform(forest_data["day"])
        X = forest_data[forest_data.columns[:-1]].values
        y = forest_data[forest_data.columns[-1]].values
        y = np.log(y + 1)
    elif name == "random":
        X, y = datasets.make_regression(nr_samples, nr_features, random_state=0)
    return X, y


def generate_data(
    m: int, n: int, seed: Optional[int] = None
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Generate some arbitrary data.

    :param m: Number of samples.
    :param n: Number of features, including intercept.
    :param seed: Random seed used in data generation.
    :return: m-by-n matrix with dummy data with intercept column (first
        column), vector of dummy labels, vector of true weights.
    """

    np.random.seed(seed)
    beta_true = np.asarray([(-1.0 + 2.0 * np.random.uniform()) for _ in range(n)])
    # Data
    X = np.asarray(
        [[1.0] + [-1 + 2 * np.random.uniform() for _ in range(n - 1)] for _ in range(m)]
    )
    y_true = np.dot(X, beta_true)
    y_noisy = y_true + np.asarray(
        [0.5 * np.sqrt(n) * (-1.0 + 2.0 * np.random.uniform()) for _ in range(m)]
    )
    return X, y_noisy, beta_true


def normalize_data(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return data / np.max(np.abs(data), axis=0)


def split_data(
    X: npt.NDArray[Any], num_slices: int = 2, loc_slices: Optional[List[int]] = None
) -> List[npt.NDArray[Any]]:
    """Vertically split data in parts.

    :param X: Dataset to split.
    :param num_slices: Number of slices to be created.
    :param loc_slices: List with indices that indicate where to split the
        dataset. Every index denotes the first column of a new slice. Takes
        precedence over num_slices.
    :return: List of arrays that result from splitting.
    """

    if loc_slices is None:
        n = X.shape[1]
        loc_slices = [i * n // num_slices for i in range(1, num_slices)]
    return np.split(X, loc_slices, axis=1)
