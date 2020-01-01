# src/models/model_tuners.py

from typing import Any, Dict, Mapping, Optional, Tuple, Union

from src.models.base_tuner import BaseTuner

from optuna import create_study, Study, Trial
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from torch.cuda import device_count
from xgboost import XGBModelBase, XGBRegressor


class LassoRidgeTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        model: Union[Lasso, Ridge],
        alpha_range: Mapping[str, Tuple[float, float]],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=model, params=alpha_range, X=X, y=y, cv=cv)

    def objective(self, trial: Trial) -> float:
        alpha: float = trial.suggest_loguniform(
            "alpha", self.params["alpha"][0], self.params["alpha"][1]
        )
        est: BaseEstimator = self.model.__class__(alpha=alpha)
        return -cross_val_score(
            estimator=est, X=self.X, y=self.y, cv=self.cv, scoring="neg_mean_squared_error"
        ).mean()


class ElasticNetTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, Tuple[float, float]],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=ElasticNet(), params=params, X=X, y=y, cv=cv)

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, float] = {
            "alpha": trial.suggest_loguniform(
                "alpha", self.params["alpha"][0], self.params["alpha"][1]
            ),
            "l1_ratio": trial.suggest_loguniform(
                "l1_ratio", self.params["l1_ratio"][0], self.params["l1_ratio"][1]
            ),
        }
        est: BaseEstimator = self.model.__class__(**suggest)
        return -cross_val_score(
            estimator=est, X=self.X, y=self.y, cv=self.cv, scoring="neg_mean_squared_error"
        ).mean()


class RFRTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, tuple],
        X: DataFrame,
        y: Series,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(
            trials=trials, model=RandomForestRegressor(), params=params, X=X, y=y.iloc[:, 0], cv=cv
        )

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, Union[bool, float, int, str]] = {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.params["n_estimators"][0], self.params["n_estimators"][1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.params["max_depth"][0], self.params["max_depth"][1]
            ),
            "max_features": trial.suggest_categorical("max_features", self.params["max_features"]),
            "min_samples_leaf": trial.suggest_loguniform(
                "min_samples_leaf",
                self.params["min_samples_leaf"][0],
                self.params["min_samples_leaf"][1],
            ),
            "min_samples_split": trial.suggest_loguniform(
                "min_samples_split",
                self.params["min_samples_split"][0],
                self.params["min_samples_split"][1],
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", (True, False)),
        }
        est: BaseEstimator = self.model.__class__(**suggest)
        return -cross_val_score(
            estimator=est, X=self.X, y=self.y, cv=self.cv, scoring="neg_mean_squared_error"
        ).mean()


class XGBRTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, Union[str, Tuple[Union[float, int, str], ...]]],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=XGBRegressor(), params=params, X=X, y=y, cv=cv)
        self.tree_method: str = "gpu_hist" if device_count() else "hist"

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, Union[float, int, str]] = {
            "objective": "reg:squarederror",  # xgboost v.90
            "tree_method": self.tree_method,
            "n_estimators": trial.suggest_int(
                "n_estimators", self.params["n_estimators"][0], self.params["n_estimators"][1]
            ),
            "reg_alpha": trial.suggest_loguniform(
                "reg_alpha", self.params["alpha"][0], self.params["alpha"][1]
            ),
            "reg_lambda": trial.suggest_loguniform(
                "reg_lambda", self.params["lambda"][0], self.params["lambda"][1]
            ),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", self.params["learning_rate"][0], self.params["learning_rate"][1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.params["max_depth"][0], self.params["max_depth"][1]
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                self.params["min_child_weight"][0],
                self.params["min_child_weight"][1],
            ),
            "gamma": trial.suggest_loguniform(
                "gamma", self.params["gamma"][0], self.params["gamma"][1]
            ),
            "subsample": trial.suggest_uniform(
                "subsample", self.params["subsample"][0], self.params["subsample"][1]
            ),
            "colsample_bytree": trial.suggest_uniform(
                "colsample_bytree", self.params["colsample"][0], self.params["colsample"][1]
            ),
            "colsample_bylevel": trial.suggest_uniform(
                "colsample_bylevel", self.params["colsample"][0], self.params["colsample"][1]
            ),
            "colsample_bynode": trial.suggest_uniform(
                "colsample_bynode", self.params["colsample"][0], self.params["colsample"][1]
            ),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }
        est: BaseEstimator = self.model.__class__(**suggest)
        return -cross_val_score(
            estimator=est, X=self.X, y=self.y, cv=self.cv, scoring="neg_mean_squared_error"
        ).mean()

    def tune(
        self, seed: Optional[int], direction: Optional[str] = None, n_jobs: Optional[int] = None
    ) -> XGBModelBase:
        self.study: Study = create_study(direction=direction)
        self.study.optimize(self.objective, n_trials=self.trials, n_jobs=n_jobs)
        return self.model.__class__(
            **self.study.best_params,
            objective="reg:squarederror",
            tree_method=self.tree_method,
            random_state=seed
        ).fit(self.X, self.y)
