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


def get_train_test(df: DataFrame, columns: list, 
                   forward_steps: int, backward_steps: int, 
                   last: str='24m', last_val=None) -> tuple:

    ts_df_back, lag_cols = add_lags(df, lags=backward_steps, forward=False, 
                                    trim=True, subset=columns, 
                                    return_cols=True)
    lag_cols.extend(columns)
    ts_df_back_test = ts_df_back.set_index('dttm').last(last)

    index_test = ts_df_back_test.index
    ts_df_back_train = ts_df_back.set_index('dttm').drop(index_test)


    if last_val is not None:
        ts_df_back_val = ts_df_back_train.last(last_val)
        index_val = ts_df_back_val.index
        ts_df_back_train = ts_df_back.set_index('dttm').drop(index_val)
        df_val, lead_cols = add_lags(ts_df_back_val, lags=forward_steps,
                                       forward=True, trim=True, 
                                       subset='category', return_cols=True)

    df_train, lead_cols = add_lags(ts_df_back_train, lags=forward_steps,
                                   forward=True, trim=True, 
                                   subset='category', return_cols=True)
    df_test, lead_cols = add_lags(ts_df_back_test, lags=forward_steps, 
                                  forward=True, trim=True, 
                                  subset='category', return_cols=True)
    if last_val is not None:
        return (
            df_train.reset_index(), lag_cols,
            df_val.reset_index(), lag_cols,
            df_test.reset_index(), lead_cols)
    else:
        return (
            df_train.reset_index(), lag_cols,
            df_test.reset_index(), lead_cols)


from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import f1_score

from types import FunctionType

def columnwise_score(scoring_func: FunctionType, 
                     preds_df: DataFrame, 
                     true_df: DataFrame,
                     **kwargs) -> Series:

    score = Series(dtype='float64')
    for (column_pred, y_pred), (columns_true, y_true) in zip(preds_df.iteritems(), 
                                                             true_df.iteritems()):
        score.loc[column_pred] = scoring_func(y_pred, y_true, **kwargs)
    
    return score


def columnwise_confusion_matrix(preds_df: DataFrame, y_true_df: DataFrame, 
                                categories: list) -> list:

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

def _create_param_grid(params: dict) -> map:
    return map(lambda x: dict(zip(params.keys(), x)), 
               product(*params.values()))

def validate(model, params: list,
             X_train: DataFrame, y_train: Series,
             X_val: DataFrame, y_val: Series,
             scoring: str, verbose: int=1, ) -> list:
    
    best_score = 0
    best_param = None

    scorer = SCORERS[scoring]

    for param in _create_param_grid(params):

        start_time = time.time()
        model_param = clone(model).set_params(**param)
        if verbose > 0: print(model_param.get_params())
        if verbose > 0: print(f'Fitting param = {param}')
        model_param.fit(X_train, y_train)
        score = scorer(model_param, X_val, y_val)
        if score > best_score: 
            best_score = score
            best_param = param

        end_time = time.time()
        if verbose > 0: print(f'Param {param}, time {end_time - start_time:.2f}')

    return best_score, best_param

from pandas import get_dummies

def validate_keras_cv(model: FunctionType, get_callbacks: FunctionType, shape: tuple,
                      cv, params: list, param_name: str, scoring: FunctionType,
                      X: DataFrame, y: Series,
                      verbose: bool=True, seed: int=None,
                      scoring_kwargs: dict={}, fit_kwargs: dict={}, 
                      callback_kwargs: dict={}) -> list:
    
    scores = []

    for param in _create_param_grid(params):

        start_time = time.time()
        
        if seed is not None: set_seed(seed)
        if verbose: print('Fitting param {} = {}'.format(param_name, param))
        sub_scores = []
        for train_idx, test_idx in cv.split(X, y):
            callbacks_ = get_callbacks(**callback_kwargs)
            model_param = model(shape, **param)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            y_train = get_dummies(y_train)
            model_param.fit(X_train, y_train, callbacks=callbacks_,
                            **fit_kwargs)
            y_preds = model_param.predict(X_test).argmax(axis=1)
            sub_scores.append(scoring(y_preds, y_test, **scoring_kwargs))
        
        scores.append(sub_scores)
   
        end_time = time.time()
        if verbose: print('Param {} = {}, time {:.2f}'.format(param_name, 
                                                              param, 
                                                              end_time - start_time))

from seaborn import heatmap
from matplotlib.pyplot import subplots, tight_layout

def plot_all_confusion_matrices(matrices: list, 
                                h: int=2, w: int= 2, **kwargs) -> None:

    n = len(matrices)

    f, ax = subplots(n, 1, sharex=True)

    for i, (key, matrix) in enumerate(matrices.items()):
        heatmap(matrix.astype('int'), ax=ax[i], **kwargs)
        ax[i].set_title(key)

    f.set_figheight(n * h)
    f.set_figwidth(w)

    tight_layout()

