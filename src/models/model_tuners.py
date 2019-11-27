# models.py


from typing import Any, Dict, Mapping, Tuple, Union

from .base_tuner import BaseTuner

from numpy import mean
from optuna import Trial
from pandas import DataFrame
from sklearn.base import BaseCrossValidator, BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import cross_val_score
from sklearn.svm import SVR
from xgboost import XGBRegressor


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
        alpha_lower: float = self.params["alpha"][0]
        alpha_upper: float = self.params["alpha"][1]
        alpha: float = trial.suggest_loguniform("alpha", alpha_lower, alpha_upper)
        self.model = self.model(alpha=alpha)
        return mean(cross_val_score(estimator=self.model, X=self.X, y=self.y, cv=self.cv))


class ElasticNetTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, Tuple[float, float]],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=ElasticNet, params=params, X=X, y=y, cv=cv)

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, float] = {
            "alpha": trial.suggest_loguniform(
                "alpha", self.params["alpha"][0], self.params["alpha"][1]
            ),
            "l1_ratio": trial.suggest_loguniform(
                "l1_ratio", self.params["l1_ratio"][0], self.params["l1_ratio"][1]
            ),
        }
        self.model = self.model(**suggest)
        return mean(cross_val_score(estimator=self.model, X=self.X, y=self.y, cv=self.cv))


class SVRTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, Tuple[Union[float, str], ...]],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=SVR, params=params, X=X, y=y, cv=cv)

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, Union[float, str]] = {
            "kernel": trial.suggest_categorical("kernel", self.params["kernel"]),
            "gamma": "auto",
            "C": trial.suggest_loguniform("C", self.params["C"][0], self.params["C"][1]),
            "epsilon": trial.suggest_loguniform(
                "epsilon", self.params["epsilon"][0], self.params["epsilon"[1]]
            ),
        }
        self.model = self.model(**suggest)
        return mean(cross_val_score(estimator=self.model, X=self.X, y=self.y, cv=self.cv))


class RFRTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, tuple],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=RandomForestRegressor, params=params, X=X, y=y, cv=cv)

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, Union[float, int, str]] = {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.params["n_estimators"][0], self.params["n_estimators"][1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.params["max_depth"][0], self.params["max_depth"][1]
            ),
            "max_features": trial.suggest_categorical("max_features", self.params["max_features"]),
            "min_samples_leaf": trial.suggest_int(
                "min_sample_leaf",
                self.params["min_sample_leaf"][0],
                self.params["min_sample_leaf"][1],
            ),
        }
        self.model = self.model(**suggest)
        return mean(cross_val_score(estimator=self.model, X=self.X, y=self.y, cv=self.cv))


class XGBRTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        params: Mapping[str, Tuple[float, int]],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        super().__init__(trials=trials, model=XGBRegressor, params=params, X=X, y=y, cv=cv)

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, Union[float, int]] = {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.params["n_estimators"][0], self.params["n_estimators"][1]
            ),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", self.params["learning_rate"][0], self.params["learning_rate"][1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.params["max_depth"][0], self.params["max_depth"][1]
            ),
            "min_child_weight": trial.suggest_loguniform(
                "min_child_weight",
                self.params["min_child_weight"][0],
                self.params["min_child_weight"][1],
            ),
            "gamma": trial.suggest_loguniform(
                "gamma", self.params["gamma"][0], self.params["gamma"][1]
            ),
            "subsample": trial.suggest_loguniform(
                "subsample", self.params["subsample"][0], self.params["subsample"][1]
            ),
            "colsample_bytree": trial.suggest_loguniform(
                "colsample_bytree",
                self.params["colsample_bytree"][0],
                self.params["colsample_bytree"][1],
            ),
            "reg_alpha": trial.suggest_loguniform(
                "reg_alpha", self.params["alpha"][0], self.params["alpha"][1]
            ),
            "reg_lambda": trial.suggest_loguniform(
                "reg_lambda", self.params["lambda"][0], self.params["lambda"][1]
            ),
        }
        self.model = self.model(**suggest)
        return mean(cross_val_score(estimator=self.model, X=self.X, y=self.y, cv=self.cv))
