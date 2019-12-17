# src/models/score_models.py

import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, NoReturn, Optional, Tuple, Union

from src.models.base_tuner import BaseTuner
from src.models.model_tuners import LassoRidgeTuner, ElasticNetTuner, RFRTuner, XGBRTuner

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split
from xgboost.compat import XGBModelBase


def _score_model(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    tuner: BaseTuner,
    output_dpath: Path,
    seed: Optional[int],
) -> None:
    estimator: Union[BaseEstimator, XGBModelBase] = tuner.tune(seed=seed)
    model_name: str = estimator.__class__.__name__
    print(
        "{}: [MSE: {}, R^2: {}]".format(
            model_name, mse(y_test, estimator.predict(X_test)), estimator.score(X_test, y_test)
        )
    )
    pickle.dump(estimator, open(Path(output_dpath, "{}.pickle".format(model_name)), "wb"))
    pickle.dump(tuner.study, open(Path(output_dpath, "{}_study.pickle".format(model_name)), "wb"))
    pickle.dump(
        tuner.study.best_trial, open(Path(output_dpath, "{}_trial.pickle".format(model_name)), "wb")
    )


def main(
    trials: int, split: float, folds: int, model: str, seed: Optional[int]
) -> Optional[NoReturn]:
    """
    Train models.
    """
    options: Dict[str, int] = {"lasso": 0, "ridge": 1, "elasticnet": 2, "rf": 3, "gbtree": 4}
    if model not in options.keys():
        raise ValueError(
            "Unsupported model selected. Currently support models are lasso, ridge, elasticnet, rf "
            + "and gbtree"
        )

    # Load data without simulation labels.
    project_home: Path = Path(__file__).parents[2]
    X: pd.DataFrame = pd.read_csv(
        Path(project_home, "data", "processed", "phx_processed_params.csv")
    ).drop(["Project_Name"], axis=1)
    y: pd.DataFrame = pd.read_csv(
        Path(project_home, "data", "processed", "phx_processed_metrics.csv")
    ).drop("sim", axis=1)
    output_dpath: Path = Path(project_home, "models")
    hyperparam_space: Dict[str, Any] = json.load(
        Path(output_dpath, "hyperparameter_search.json").open()
    )

    # Split dataset.
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    # Set k-fold cv and tuner objects.
    kfold_cv: KFold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    tuners: Tuple[BaseTuner, ...] = (
        LassoRidgeTuner(
            trials=trials,
            model=Lasso(),
            alpha_range=hyperparam_space["lasso"],
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        LassoRidgeTuner(
            trials=trials,
            model=Ridge(),
            alpha_range=hyperparam_space["ridge"],
            X=X_train,
            y=y_train,
            cv=kfold_cv,
        ),
        ElasticNetTuner(
            trials=trials, params=hyperparam_space["elasticnet"], X=X_train, y=y_train, cv=kfold_cv
        ),
        RFRTuner(trials=trials, params=hyperparam_space["rf"], X=X_train, y=y_train, cv=kfold_cv),
        XGBRTuner(
            trials=trials, params=hyperparam_space["gbtree"], X=X_train, y=y_train, cv=kfold_cv
        ),
    )

    # Score the selected model.
    _score_model(
        X_test=X_test,
        y_test=y_test,
        tuner=tuners[options[model]],
        output_dpath=output_dpath,
        seed=seed,
    )

    return None  # make mypy happy


if __name__ == "__main__":
    trials: int = int(sys.argv[1])  # in data/raw/
    split: float = float(sys.argv[2])
    folds: int = int(sys.argv[3])
    model: str = sys.argv[4]
    seed: Optional[int] = None

    try:
        seed = int(sys.argv[5])
    except IndexError:
        pass

    main(trials=trials, split=split, folds=folds, model=model, seed=seed)
