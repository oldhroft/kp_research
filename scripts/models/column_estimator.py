from sklearn.base import BaseEstimator
from ..pipeline.preprocess import categorize

from numpy import unique, array


class ColumnEstimator(BaseEstimator):
    def __init__(self, column_idx: int = 0):
        self.column_idx = column_idx

    def fit(self, X: array, y: array):
        self.classes_ = unique(y)
        return self

    def predict(self, X: array):
        return array(list(map(categorize, list(X[:, self.column_idx]))))
