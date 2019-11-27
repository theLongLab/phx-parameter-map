# base_tuner.py

from abc import abstractmethod
from typing import Any, Mapping, NoReturn, Union

from optuna import create_study, optimize, Study, Trial
from pandas import DataFrame
from sklearn.base import BaseCrossValidator, BaseEstimator
from xgboost.compat import XGBModelBase


class BaseTuner:
    def __init__(
        self,
        trials: int,
        model: Union[BaseEstimator, XGBModelBase],
        params: Mapping[str, tuple],
        X: DataFrame,
        y: DataFrame,
        cv: BaseCrossValidator,
    ) -> None:
        self.model: Union[BaseEstimator, XGBModelBase] = model
        self.trials: int = trials
        self.params: Mapping[str, tuple] = params
        self.X: DataFrame = X
        self.y: DataFrame = y
        self.cv = cv

    @abstractmethod
    def objective(self, trial: Trial) -> Union[float, NoReturn]:
        raise NotImplementedError

    def tune(self, direction: str) -> Union[BaseEstimator, XGBModelBase]:
        study: Study = create_study(direction=direction)
        study.optimize(self.objective, n_trials=self.trials)
        return self.model(**study.best_params).fit(self.X, self.y)
