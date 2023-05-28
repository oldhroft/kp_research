import os
from typing import Any, List, Dict, Callable

def create_folder(name: str) -> None:
    if not os.path.exists(name):
        os.mkdir(name)


def add_to_environ(data: List[str]) -> None:
    for item in data:
        key, value = item.split("=")
        os.environ[key] = value


def decorate_class(decorator: Callable) -> Callable:
    def _decorate_class(cls: type):

        method_list = [
            func
            for func in dir(cls)
            if callable(getattr(cls, func)) and not func.startswith("__")
        ]

        for method in method_list:
            setattr(cls, method, decorator(getattr(cls, method)))

        return cls

    return _decorate_class


from pandas import DataFrame, Series


def _trim(df: DataFrame, forward: bool, trim: bool, lags: int) -> DataFrame:
    if trim and forward:
        return df.iloc[:-lags]
    elif trim:
        return df.iloc[lags:]
    else:
        return df


def add_lags(
    df: DataFrame,
    subset: list = None,
    forward: bool = False,
    lags: int = 1,
    trim: bool = False,
    suffix_name: str = None,
    return_cols=False,
) -> DataFrame:

    if suffix_name is None:
        suffix_name = "lead" if forward else "lag"

    x = df.copy()

    digits = len(str(lags))

    columns = []

    if not isinstance(lags, int):
        raise ValueError(f"Lags should be int, {type(lags)} type prodided")
    elif lags < 0:
        raise ValueError(f"Lags should be non-negative")
    elif lags == 0:
        return x
    elif subset is None:
        for i in range(1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f"_{suffix_name}_{index}"

            x = x.join(x.shift(lag).add_suffix(column_suffix))

        columns = x.columns.tolist()

    elif isinstance(subset, list):
        for i in range(1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_suffix = f"_{suffix_name}_{index}"
            tmp = x.loc[:, subset].shift(lag).add_suffix(column_suffix)
            columns.extend(tmp.columns)
            x = x.join(tmp)

    elif isinstance(subset, str):
        for i in range(1, lags + 1):
            lag = -i if forward else i
            index = str(i).zfill(digits)
            column_name = f"{subset}_{suffix_name}_{index}"
            columns.append(column_name)

            x = x.join(x.loc[:, subset].shift(lag).rename(column_name))
    else:
        raise ValueError(
            f"Subset should be str or list, providided type {type(subset)}"
        )

    if return_cols:
        return _trim(x, forward, trim, lags), columns
    else:
        return _trim(x, forward, trim, lags)


from sklearn.metrics import confusion_matrix


def columnwise_score(
    scoring_func: Callable, preds_df: DataFrame, true_df: DataFrame, **kwargs
) -> Series:

    score = Series(dtype="float64")
    preds_df = DataFrame(preds_df)
    true_df = DataFrame(true_df)
    for (column_pred, y_pred), (columns_true, y_true) in zip(preds_df.items(), 
                                                             true_df.items()):
        score.loc[column_pred] = scoring_func(y_true, y_pred, **kwargs)

    return score


def columnwise_confusion_matrix(
    preds_df: DataFrame, y_true_df: DataFrame, categories: list
) -> Dict[Any, Any]:

    preds_df = DataFrame(preds_df)
    y_true_df = DataFrame(y_true_df)

    all_matrices = {}
    for (column_pred, y_pred), (columns_true, y_true) in zip(preds_df.items(), 
                                                             y_true_df.items()):

        matrix = DataFrame(
            confusion_matrix(y_true, y_pred), index=categories, columns=categories
        )
        all_matrices[column_pred] = matrix

    return all_matrices


import time
from itertools import product
from sklearn.metrics import SCORERS
from numpy import array


def _create_param_grid(params: dict) -> map:
    return map(lambda x: dict(zip(params.keys(), x)), product(*params.values()))


def validate(
    model: Callable,
    init_params: dict,
    params: dict,
    X_train: array,
    y_train: array,
    X_val: array,
    y_val: array,
    scoring: str,
    verbose: int = 1,
) -> tuple:

    best_score = 0
    best_param = None

    scorer = SCORERS[scoring]
    results = []

    for param in _create_param_grid(params):

        start_time = time.time()
        full_params = init_params.copy()
        full_params.update(param)
        model_param = model(**full_params)
        if verbose > 0:
            print(f"Fitting param = {param}")
        model_param.fit(X_train, y_train)
        score = scorer(model_param, X_val, y_val)
        if score > best_score:
            best_score = score
            best_param = param
        results.append({**param, "score": score})

        end_time = time.time()
        if verbose > 0:
            print(f"Param {param}, time {end_time - start_time:.2f}")

    return best_score, best_param, DataFrame(results)


from pandas import Series, DataFrame
from pandas import get_dummies
from numpy import mean
from sklearn.model_selection import BaseCrossValidator
from tensorflow.keras import callbacks as callbacks

try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import random as random_tf

    set_seed = random_tf.set_seed

import random


def _random_param_grid(params: dict, n_iter: int, seed: int) -> list:
    grid = list(_create_param_grid(params))
    random.seed(seed)
    random.shuffle(grid)
    return grid[:n_iter]


def validate_keras_cv(
    model: Callable,
    init_params: dict,
    cv: BaseCrossValidator,
    params: dict,
    scoring: Callable,
    X: array,
    y: array,
    callback_params: dict,
    scoring_params: dict,
    fit_params: dict,
    verbose: bool,
    seed: int,
    n_iter: int = None,
) -> tuple:
    best_score = 0
    best_param = None
    results = []

    if n_iter is None:
        grid = list(_create_param_grid(params))
    else:
        grid = _random_param_grid(params, n_iter=n_iter, seed=seed)

    for param in grid:
        start_time = time.time()
        if seed is not None:
            set_seed(seed)
        if verbose:
            print("Fitting param {}".format(param))
        full_params = init_params.copy()
        full_params.update(param)
        sub_scores = []

        i = 0
        for train_idx, test_idx in cv.split(X, y):
            callbacks_list = [
                callbacks.EarlyStopping(**callback_params),
            ]
            random.seed(seed)
            random.shuffle(train_idx)
            model_param = model(**full_params)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            y_train = array(get_dummies(y_train))
            model_param.fit(X_train, y_train, callbacks=callbacks_list, **fit_params)
            y_preds = model_param.predict(X_test).argmax(axis=1)
            score = scoring(y_preds, y_test, **scoring_params)
            sub_scores.append(score)
            result_dict = {**param, "score": score, "split": i}
            i += 1
            results.append(result_dict)

        score = mean(sub_scores)

        if score > best_score:
            best_score = score
            best_param = param

        end_time = time.time()
        if verbose:
            print("Param {}, time {:.2f}".format(param, end_time - start_time))

    return best_param, best_score, DataFrame(results)


def validate_keras(
    model: Callable,
    init_params: dict,
    params: dict,
    scoring: Callable,
    X_train: array,
    y_train: array,
    X_val: array,
    y_val: array,
    callback_params: dict,
    scoring_params: dict,
    fit_params: dict,
    verbose: bool,
    seed: int,
) -> tuple:

    best_score = 0
    best_param = None

    results = []

    for param in _create_param_grid(params):

        start_time = time.time()
        if seed is not None:
            set_seed(seed)
        if verbose:
            print("Fitting param {}".format(param))
        full_params = init_params.copy()
        full_params.update(param)

        callbacks_list = [
            callbacks.EarlyStopping(**callback_params),
        ]
        model_param = model(**full_params)
        y_train = array(get_dummies(y_train))

        model_param.fit(X_train, y_train, callbacks=callbacks_list, **fit_params)

        y_preds = model_param.predict(X_val).argmax(axis=1)
        score = scoring(y_preds, y_val, **scoring_params)

        if score > best_score:
            best_score = score
            best_param = param

        results.append({**param, "score": score})

        end_time = time.time()
        if verbose:
            print("Param {}, time {:.2f}".format(param, end_time - start_time))

    return best_param, best_score, DataFrame(results)
