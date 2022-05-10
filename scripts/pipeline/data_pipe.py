class DataPipe(object):

    def __init__(self,  steps) -> None:
        self.steps = steps

    def fit_transform(self, X):

        self.fitted_steps = []
        for step, arg, is_fittable in self.steps:

            if is_fittable:
                X = step.fit_transform(X, **arg)
                step = step.transform
            else:
                X = step(X, **arg)
            self.fitted_steps.append((step, arg))
        return X
    
    def transform(self, X):
        for step, arg in self.fitted_steps:
            X = step(X, **arg)
        return X

from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from numpy import array
from ..helpers.utils import add_lags 

def _select(df: DataFrame, columns: list) -> DataFrame:
    return df.loc[:, columns]

def _split_and_drop(data_tuple: tuple, drop_columns) -> tuple:
    df = data_tuple[0]
    y_columns = data_tuple[1]
    return df.drop(y_columns + drop_columns, axis=1), df.loc[:, y_columns]

def _get_feature_names(data_tuple: tuple,) -> tuple:
    return data_tuple[0], data_tuple[1], data_tuple[0].columns
    
def _shuffle(data_tuple: tuple, shuffle: bool, random_state: int) -> DataFrame:
    if shuffle:
        X = data_tuple[0]
        cols = data_tuple[1]
        return X.sample(frac=1., random_state=random_state), cols
    else:
        return data_tuple

def _as_numpy(data_tuple: tuple) -> tuple:
    return tuple(map(array, data_tuple))

class _StandardScalerXY(StandardScaler):
    
    def fit(self, data_tuple: tuple):
        return super().fit(data_tuple[0])
    
    def transform(self, data_tuple: tuple, ):
        X_scaled = super().transform(data_tuple[0])
        return tuple([X_scaled, *data_tuple[1: ]])
    
    def fit_transform(self, data_tuple: tuple):
        return self.fit(data_tuple).transform(data_tuple)


class LagDataPipe(DataPipe):

    def __init__(self, variables, target, backward_steps, forward_steps,
                 scale=False, shuffle=True, random_state=None):
        self.steps = [
            (_select, {"columns": variables + [target]}, False),
            (
                add_lags, {
                    "lags": backward_steps, "forward": False, 
                    "trim": True, "subset": variables, 
                    "return_cols": False},
                False),
            (
                add_lags, {
                    "lags": forward_steps, "forward": True, 
                    "trim": True, "subset": target, 
                    "return_cols": True},
                False),
            (_shuffle, {"random_state": random_state, "shuffle": shuffle}, False),
            (_split_and_drop, {"drop_columns": [target]}, False),
            (_get_feature_names, {}, False),
            (_as_numpy, {}, False)
        ]
        if scale: self.steps.append( (_StandardScalerXY(), {}, True))

def _reshape(data_tuple, n_features, time_steps):
    X = data_tuple[0].reshape((-1, time_steps, n_features))
    y = data_tuple[1]
    return X, y
def _pack_with_array(data_tuple, array):
    return tuple((*data_tuple, array))

class SequenceDataPipe(LagDataPipe):
    
    def __init__(self, variables, target, backward_steps, forward_steps,
                 scale=False, shuffle=True, random_state=None):
        
        super().__init__(variables, target, backward_steps, forward_steps,
                         scale=scale, shuffle=shuffle, random_state=random_state)
        
        self.steps.extend([
            (
                _reshape, {
                    "n_features": len(variables), 
                    "time_steps": backward_steps + 1
                }, False),
            (_pack_with_array, {"array": variables}, False)
        ])



    

        
