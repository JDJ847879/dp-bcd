from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

DATADIR = Path(__file__).parent.parent / "data"

"""
The preprocessing of data is following what Van Kesteren et al did, in order to reproduce they result.
"""


def get_data_python_preprocessing() -> Tuple[
    npt.NDArray[Any], npt.NDArray[Any], List[npt.NDArray[Any]]
]:
    df = pd.read_csv(DATADIR / "forestfires.csv")
    columnsA = ["X", "Y", "temp", "RH", "wind", "rain"]
    columnsB = ["FFMC", "DMC", "DC", "ISI"]
    target = "area"

    X = df[columnsA + columnsB].copy().values
    X_A = df[columnsA].copy().values
    X_B = df[columnsB].copy().values
    y_raw = np.array(df[target])
    y_log = np.log(y_raw + 1).flatten()
    return X, y_log, [X_A, X_B]


def get_data() -> Tuple[npt.NDArray[Any], npt.NDArray[Any], List[npt.NDArray[Any]]]:
    # y_raw = pd.read_csv(datadir / "data_csv/forest_fire_y.csv").drop(columns=["Unnamed: 0"]).values
    X_A = (
        pd.read_csv(DATADIR / "forest_fire_Alice.csv")
        .drop(columns=["Unnamed: 0"])
        .values
    )
    X_B = (
        pd.read_csv(DATADIR / "forest_fire_Bob.csv")
        .drop(columns=["Unnamed: 0"])
        .values
    )
    logy = (
        pd.read_csv(DATADIR / "forest_fire_logy.csv")
        .drop(columns=["Unnamed: 0"])
        .values.flatten()
    )
    return np.append(X_A, X_B, axis=1), logy, [X_A, X_B]
