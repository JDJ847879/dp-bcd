from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

DATADIR = Path(__file__).parent.parent / "data"

"""
The preprocessing of data is following what Van Kesteren et al did, in order to reproduce they result.
The get_data returns the data used in Van Kesteren code.
"""


def get_data() -> Tuple[npt.NDArray[Any], npt.NDArray[Any], List[npt.NDArray[Any]]]:
    y = (
        pd.read_csv(DATADIR / "data_csv/diab_label_15000.csv")
        .drop(columns=["Unnamed: 0"])
        .values.flatten()
    )
    # X_A = pd.read_csv(datadir / "data_csv/diab_administration_15000.csv").drop(columns=["(Intercept)", "Unnamed: 0"]).values
    X_A = (
        pd.read_csv(DATADIR / "data_csv/diab_administration_15000.csv")
        .drop(columns=["Unnamed: 0"])
        .values
    )
    X_B = (
        pd.read_csv(DATADIR / "data_csv/diab_medication_15000.csv")
        .drop(columns=["Unnamed: 0"])
        .values
    )
    return np.append(X_A, X_B, axis=1), y, [X_A, X_B]


def get_data_python_preprocessing(
    sample_dim: int = 15000, random_seed: int = 45
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], List[npt.NDArray[Any]]]:
    # take df:
    df_admin = pd.read_csv(DATADIR / "diab_administration.csv").drop(
        columns=["Unnamed: 0"]
    )
    df_med = pd.read_csv(DATADIR / "diab_medication.csv").drop(columns=["Unnamed: 0"])

    # all values of df_med are 'Yes' or 'True': let's map them in 1 or 0
    for c in [c for c in df_med.columns if c != "readmitted"]:
        df_med[c] = df_med[c].map({"Yes": 1, "No": 0})

    # create one_hot_encoder for admin features
    categorical_features_admin = [
        "race",
        "gender",
        "age",
        "diabetesMed",
        "change",
        "glu_serum_positive",
        "A1C_positive",
    ]
    enc_admin = OneHotEncoder(categories="auto", drop="first", handle_unknown="error")

    # take a sample of datasets, and apply one_hot_encoder to df_admin
    df_A = (
        df_admin.sample(n=sample_dim, random_state=random_seed)
        .drop(columns=["readmitted"])
        .copy()
    )
    # index = df_A.index

    X_A_enc = enc_admin.fit_transform(df_A[categorical_features_admin])
    df_A_enc = pd.DataFrame(X_A_enc.toarray(), columns=enc_admin.get_feature_names())
    df_A = (
        df_A[[c for c in df_A.columns if not (c in categorical_features_admin)]]
        .reset_index()
        .join(df_A_enc)
        .set_index("index")
    )

    # take the same sample for df_med and y
    df_B = df_med.loc[df_med.index.isin(df_A.index)].drop(columns=["readmitted"]).copy()
    df_y = df_med.loc[df_med.index.isin(df_A.index)]["readmitted"].copy()

    # to values
    X = df_A.join(df_B).values
    X_A = df_A.values
    X_B = df_B.values
    y = df_y.values.flatten()

    return X, y, [X_A, X_B]
