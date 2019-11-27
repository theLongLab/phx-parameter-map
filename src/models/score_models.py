# train_sklearn_models.py

import csv
from pathlib import Path
import pickle
from typing import List, Union

from .base_tuner import BaseTuner
from .model_tuners import LassoRidgeTuner, ElasticNetTuner, SVRTuner, RFRTuner, XGBRTuner

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, train_test_split
from xgboost.compat import XGBModelBase


def _score_model(
    X_test: pd.DataFrame, y_test: pd.DataFrame, tuner: BaseTuner, output_dir: Path
) -> None:
    estimator: Union[BaseEstimator, XGBModelBase] = tuner.tune("maximize")
    model_name: str = estimator.__class__.__name__
    print("{}: {}".format(model_name, estimator.score(X_test, y_test)))
    pickle.dump(estimator, open(Path(estimator, "{}.pickle".format(model_name)), "wb"))


def main(trials: int = 100) -> None:
    """
    Train models.
    """
    project_home: Path = Path("../../").absolute()
    X: pd.DataFrame = pd.read_csv(
        Path(project_home, "data", "raw", "phx_params.txt"), sep="\t"
    ).drop("Project_Name")
    y: pd.DataFrame = pd.read_csv(
        Path(project_home, "data", "processed", "phx_processed_metrics.csv")
    ).drop("sim")
    output_dir: Path = Path(project_home, "models")

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    kfold_cv: KFold = KFold(n_splits=8, shuffle=True)
    tuners: List[BaseTuner] = [
        LassoRidgeTuner(
            trials=trials,
            model=Lasso,
            alpha_range={"alpha": (1e-10, 10)},
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        LassoRidgeTuner(
            trials=trials,
            model=Ridge,
            alpha_range={"alpha": (1e-10, 10)},
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        ElasticNetTuner(
            trials=trials,
            params={"alpha": (1e-10, 10), "l1_ratio": (1e-2, 1)},
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        SVRTuner(
            trials=trials,
            params={"kernel": ("linear", "rbf"), "C": (1e-10, 1e10), "epsilon": (1e-10, 10)},
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        RFRTuner(
            trials=trials,
            params={
                "n_estimators": (200, 2000),
                "max_depth": (0, 100),
                "max_features": ("auto", "sqrt"),
                "min_samples_leaf": (1, 100),
            },
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        XGBRTuner(
            trials=trials,
            params={
                "n_estimators": (200, 2000),
                "learning_rate": (1e-10, 1),
                "max_depth": (1, 100),
                "min_child_weight": (1, 1),
                "gamma": (1e-10, 1),
                "subsample": (1e-10, 1),
                "colsample_bytree": (1e-10, 1),
                "reg_alpha": (1e-10, 10),
                "reg_lambda": (1e-10, 10),
            },
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
    ]

    tuner: BaseTuner
    for tuner in tuners:
        _score_model(X_test, y_test, tuner, output_dir)
