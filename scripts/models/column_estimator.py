from sklearn.base import BaseEstimator
from scripts.pipeline.preprocess import categorize
from scripts.pipeline.data_pipe import DEFAULT_BORDERS

from numpy import unique, array
from typing import List


class ColumnEstimator(BaseEstimator):
    def __init__(self, column_idx: int = 0, borders: List[int] = DEFAULT_BORDERS):
        self.column_idx = column_idx
        self.borders = borders

    def fit(self, X: array, y: array):
        self.classes_ = unique(y)
        return self

    def predict(self, X: array):
        return array(
            list(
                map(
                    lambda y: categorize(y, DEFAULT_BORDERS),
                    list(X[:, self.column_idx]),
                )
            )
        )
