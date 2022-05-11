import os

def create_folder(name: str) -> None:
    if not os.path.exists(name):
        os.mkdir(name)

def _choose_suffix_name(forward: bool, suffix_name: str) -> str:
    if suffix_name is not None:
        return suffix_name
    else:
        return 'lead' if forward else 'lag'


from pandas import DataFrame, Series

def _trim(df: DataFrame, forward: bool, trim: bool, lags: int) -> DataFrame:
    if trim and forward:
        return df.iloc[: -lags]
    elif trim:
        return df.iloc[lags: ]
    else:
        return df

def add_lags(df: DataFrame, subset: list=None, forward: bool=False,
             lags: int=1, trim: bool=False, suffix_name: str=None,
             return_cols=False) -> DataFrame:

    suffix_name = _choose_suffix_name(forward, suffix_name)

    x = df.copy()

    digits = len(str(lags))

    columns = []

    if not isinstance(lags, int):
        raise ValueError(f'Lags should be int, {type(lags)} type prodided')
    elif lags < 0:
        raise ValueError(f'Lags should be non-negative')
    elif lags == 0:
        return x
    elif subset is None:
        for i in range(1, lags + 1):
            lag = - i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f'_{suffix_name}_{index}'

            x = x.join(x.shift(lag).add_suffix(column_suffix))
        
        columns = x.columns.tolist()

    elif isinstance(subset, list):
        for i in range(1, lags + 1):
            lag = - i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f'_{suffix_name}_{index}'
            tmp = x.loc[:, subset].shift(lag).add_suffix(column_suffix)
            columns.extend(tmp.columns)
            x = x.join(tmp)

    elif isinstance(subset, str):
        for i in range(1, lags + 1):
            lag = - i if forward else i
            index = str(i).zfill(digits)
            column_name = f'{subset}_{suffix_name}_{index}'
            columns.append(column_name)

            x = x.join(x.loc[:, subset].shift(lag).rename(column_name))
    else:
        raise ValueError(f'Subset should be str or list, providided type {type(subset)}')

    if return_cols:
        return _trim(x, forward, trim, lags), columns
    else:
        return _trim(x, forward, trim, lags)


from sklearn.metrics import confusion_matrix
from types import FunctionType

def columnwise_score(scoring_func: FunctionType, 
                     preds_df: DataFrame, 
                     true_df: DataFrame,
                     **kwargs) -> Series:

    score = Series(dtype='float64')
    preds_df = DataFrame(preds_df)
    true_df = DataFrame(true_df)
    for (column_pred, y_pred), (columns_true, y_true) in zip(preds_df.iteritems(), 
                                                             true_df.iteritems()):
        score.loc[column_pred] = scoring_func(y_true, y_pred, **kwargs)
    
    return score


def columnwise_confusion_matrix(preds_df: DataFrame, y_true_df: DataFrame, 
                                categories: list) -> list:
    
    preds_df = DataFrame(preds_df)
    y_true_df = DataFrame(y_true_df)

    all_matrices = {}
    for (column_pred, y_pred), (columns_true, y_true) in zip(preds_df.iteritems(), 
                                                             y_true_df.iteritems()):

        matrix = DataFrame(confusion_matrix(y_true, y_pred),
                           index=categories, columns=categories)
        all_matrices[column_pred] = matrix

    return all_matrices


from sklearn.base import clone
import time
from itertools import product
from sklearn.metrics import SCORERS
from numpy import array

def _create_param_grid(params: dict) -> map:
    return map(lambda x: dict(zip(params.keys(), x)), 
               product(*params.values()))

def validate(model, params: list,
             X_train: array, y_train: array,
             X_val: array, y_val: array,
             scoring: str, verbose: int=1,) -> list:
    
    best_score = 0
    best_param = None

    scorer = SCORERS[scoring]

    for param in _create_param_grid(params):

        start_time = time.time()
        model_param = clone(model).set_params(**param)
        if verbose > 0: print(f'Fitting param = {param}')
        model_param.fit(X_train, y_train)
        score = scorer(model_param, X_val, y_val)
        if score > best_score: 
            best_score = score
            best_param = param

        end_time = time.time()
        if verbose > 0: print(f'Param {param}, time {end_time - start_time:.2f}')

    return best_score, best_param

from pandas import Series, DataFrame
from pandas import get_dummies
from numpy import mean
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import callbacks as callbacks
try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import random as random
    set_seed = random.set_seed

def validate_keras_cv(model: FunctionType, input_shape: tuple, n_classes: int,
                      cv_params: dict, params: dict, scoring: FunctionType,
                      X: array, y: array,
                      callback_params: dict, scoring_params: dict,
                      init_params: dict, fit_params: dict,
                      verbose: bool, seed: int,) -> list:
    
    best_score = 0
    best_param = None

    for param in _create_param_grid(params):
        start_time = time.time()
        if seed is not None: set_seed(seed)
        if verbose: print('Fitting param {}'.format(param))
        full_params = init_params.copy()
        full_params.update(param)
        sub_scores = []
        cv = StratifiedKFold(**cv_params)
        for train_idx, test_idx in cv.split(X, y):
            callbacks_list = [
                callbacks.EarlyStopping(**callback_params),
            ]
            
            model_param = model(input_shape, n_classes, **full_params)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            y_train = array(get_dummies(y_train))
            model_param.fit(X_train, y_train, callbacks=callbacks_list,
                            **fit_params)
            y_preds = model_param.predict(X_test).argmax(axis=1)
            sub_scores.append(scoring(y_preds, y_test, **scoring_params))
        
        score = mean(sub_scores)

        if score > best_score:
            best_score = score
            best_param = param
   
        end_time = time.time()
        if verbose: print('Param {}, time {:.2f}'.format(param, 
                                                         end_time - start_time))

    return best_param, best_score

def validate_keras(model: FunctionType, input_shape: tuple, n_classes: int,
                   params: dict, scoring: FunctionType,
                   X_train: array, y_train: array,
                   X_val: array, y_val: array,
                   callback_params: dict, scoring_params: dict,
                   init_params: dict, fit_params: dict,
                   verbose: bool, seed: int,) -> list:
    
    best_score = 0
    best_param = None

    for param in _create_param_grid(params):

        start_time = time.time()
        if seed is not None: set_seed(seed)
        if verbose: print('Fitting param {}'.format(param))
        full_params = init_params.copy()
        full_params.update(param)

        callbacks_list = [
            callbacks.EarlyStopping(**callback_params),
        ]
        model_param = model(input_shape, n_classes, **full_params)
        y_train = array(get_dummies(y_train))

        model_param.fit(X_train, y_train, callbacks=callbacks_list,
                        **fit_params)

        y_preds = model_param.predict(X_val).argmax(axis=1)
        score = scoring(y_preds, y_val, **scoring_params)

        if score > best_score:
            best_score = score
            best_param = param

        end_time = time.time()
        if verbose: print('Param {}, time {:.2f}'.format(param, 
                                                         end_time - start_time))

    return best_param, best_score
