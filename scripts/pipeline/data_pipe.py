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
from ..helpers.utils import add_lags 

def _select(df: DataFrame, columns: list) -> DataFrame:
    return df.loc[:, columns]

def _split_and_drop(data_tuple: tuple, drop_columns) -> tuple:
    df = data_tuple[0]
    y_columns = data_tuple[1]
    return df.drop(y_columns + drop_columns, axis=1), df.loc[:, y_columns]

class _StandardScalerXY(StandardScaler):
    
    def fit(self, data_tuple: tuple):
        return super().fit(data_tuple[0])
    
    def transform(self, data_tuple: tuple, ):
        columns = data_tuple[0].columns
        X_scaled = super().transform(data_tuple[0])
        return DataFrame(X_scaled, columns), data_tuple[1]
    
    def fit_transform(self, data_tuple: tuple):
        return self.fit(data_tuple).transform(data_tuple)


class LagDataPipe(DataPipe):

    def __init__(self, columns, target, backward_steps, forward_steps, scale=False):
        self.steps = [
            (_select, {"columns": columns + [target]}, False),
            (
                add_lags, {
                    "lags": backward_steps, "forward": False, 
                    "trim": True, "subset": columns, 
                    "return_cols": False},
                False),
            (
                add_lags, {
                    "lags": forward_steps, "forward": True, 
                    "trim": True, "subset": target, 
                    "return_cols": True},
                False),
            (_split_and_drop, {"drop_columns": [target]}, False) ,
        ]
        if scale: self.steps.append( (_StandardScalerXY(), {}, True))



    

        
