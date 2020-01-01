# base_tuner.py

from abc import abstractmethod
from typing import Any, Mapping, NoReturn, Optional, Union

from optuna import create_study, Study, Trial
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from xgboost.compat import XGBModelBase


class BaseTuner:
    def __init__(
        self,
        trials: int,
        model: Union[BaseEstimator, XGBModelBase],
        params: Mapping[str, Union[str, tuple]],
        X: DataFrame,
        y: Union[DataFrame, Series],
        cv: BaseCrossValidator,
    ) -> None:
        self.model: Union[BaseEstimator, XGBModelBase] = model
        self.trials: int = trials
        self.params: Mapping[str, Union[str, tuple]] = params
        self.X: DataFrame = X
        self.y: DataFrame = y
        self.cv: BaseCrossValidator = cv

    @abstractmethod
    def objective(self, trial: Trial) -> Union[float, NoReturn]:
        raise NotImplementedError

    def tune(
        self, seed: Optional[int], direction: Optional[str] = None, n_jobs: Optional[int] = None
    ) -> Union[BaseEstimator, XGBModelBase]:
        self.study: Study = create_study(direction=direction)
        self.study.optimize(self.objective, n_trials=self.trials, n_jobs=n_jobs, random_state=seed)
        return self.model.__class__(**self.study.best_params).fit(self.X, self.y)
