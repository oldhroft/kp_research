import os
from datetime import datetime
import joblib
import yaml

from numpy import array, squeeze, vectorize
from pandas import DataFrame, get_dummies, read_csv
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from tensorflow.keras import callbacks as callbacks

try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import random as random

    set_seed = random.set_seed

from scripts.helpers.utils import (
    columnwise_confusion_matrix,
    columnwise_score,
    create_folder,
)
from scripts.models import (
    nn_model_factory,
    sk_model_factory,
    gcv_factory,
    sk_model_factory_reg,
)
from scripts.pipeline.preprocess import preprocess_3h, categorize


def read_data(path, date_from=None, val=False, regression=False):
    df = read_csv(path, encoding="cp1251", na_values="N").pipe(
        lambda x: preprocess_3h(x, regression=regression)
    )

    if date_from is not None:
        df = df.loc[df.dttm >= date_from].reset_index(drop=True)

    if not regression:
        categories = list(df.category.unique())
    else:
        categories = list(map(categorize, df.category.unique()))

    df_test = df.loc[df.year >= 2020].reset_index(drop=True)
    df_train = df.loc[df.year < 2020].reset_index(drop=True)

    if val:
        df_val = df_train.loc[df.year >= 2018].reset_index(drop=True)
        df_train = df_train.loc[df.year < 2018].reset_index(drop=True)
        return df_train, df_val, df_test, categories
    else:
        return df_train, df_test, categories


from importlib import import_module
from typing import Dict, Any, List


def get_data_pipeline(config: Dict[str, Any]) -> Any:
    cls = getattr(import_module("scripts.pipeline.data_pipe"), config["pipe_name"])
    return cls(**config["pipe_params"])


def build_data_pipelines(config: dict, structure: dict):
    pipes = {}

    for model_name, cfg in config.items():
        cfg_cp = cfg.copy()
        cfg_cp["pipe_params"]["out_folder"] = structure["data_path"]
        pipe = get_data_pipeline(cfg_cp)
        pipe.load()

        pipes[pipe.name] = pipe

    return pipes


def create_folder_structure(root: str) -> Dict[str, str]:
    os.environ["FOLDER"] = root
    structure = {"root": root}
    create_folder(root)
    for sub_folder in ["matrix", "model", "history", "vars", "cv_results", "data"]:
        path = os.path.join(root, sub_folder)
        create_folder(path)
        structure[f"{sub_folder}_path"] = path

    return structure


def _convert_to_results(gcv: GridSearchCV) -> DataFrame:
    cv_results = gcv.cv_results_
    n_splits = gcv.n_splits_

    results = []

    for j, param in enumerate(cv_results["params"]):
        for i in range(n_splits):
            score = cv_results[f"split{i}_test_score"][j]

            results.append(
                {
                    **param,
                    "score": score,
                    "split": i,
                }
            )

    return DataFrame(results)


def grid_search(
    params, model_name, init_params, X_train, y_train, cv, gcv_name, gcv_params
):
    model = sk_model_factory.get(model_name, **init_params)
    gcv = gcv_factory.get(
        "gcv", estimator=model, param_grid=params, cv=cv, **gcv_params
    )
    gcv.fit(X_train, y_train)

    results = _convert_to_results(gcv)

    return gcv.best_params_, gcv.best_score_, results


def fit(model_name: str, init_params: dict, X_train: Any, y_train: Any):
    model = sk_model_factory.get(model_name, **init_params)
    model = MultiOutputClassifier(model)
    model.fit(X_train, y_train)
    return model


def fit_reg(model_name: str, init_params: dict, X_train: Any, y_train: Any):
    model = sk_model_factory_reg.get(model_name, **init_params)
    model = MultiOutputRegressor(model)
    model.fit(X_train, y_train)
    return model


def fit_keras(
    model_name: str,
    init_params: dict,
    fit_params: dict,
    callback_params: dict,
    X_train: Any,
    y_train: Any,
    seed: int,
):
    models = []
    histories = {}

    for col in range(y_train.shape[1]):
        set_seed(seed)
        y_dummy = array(get_dummies(y_train[:, col]))
        model = nn_model_factory.get(model_name, **init_params)
        callbacks_list = [
            callbacks.EarlyStopping(**callback_params),
        ]
        history = model.fit(X_train, y_dummy, callbacks=callbacks_list, **fit_params)
        models.append(model)
        histories[col] = history

    return models, histories


def score(
    model: Any,
    model_name: str,
    X_test: Any,
    y_test: Any,
    structure: Dict[str, str],
    proc_name: str,
) -> None:
    preds = squeeze(model.predict(X_test))
    f1_macro_res = columnwise_score(f1_score, preds, y_test, average="macro")
    fname = os.path.join(structure["root"], f"{proc_name}_{model_name}_f1.csv")
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds, y_test, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(
                structure["matrix_path"], f"{proc_name}_{model_name}_{key}.xlsx"
            )
        )


from scripts.pipeline.preprocess import categorize


def score_reg(
    model, model_name, X_test, y_test, structure, proc_name, borders: List[int]
):
    preds = squeeze(model.predict(X_test))
    vectorized = vectorize(lambda x: (categorize(x, borders)))
    preds_cat = vectorized(preds)
    y_test_cat = vectorized(y_test)
    f1_macro_res = columnwise_score(f1_score, preds_cat, y_test_cat, average="macro")
    fname = os.path.join(structure["root"], f"{proc_name}_{model_name}_f1.csv")
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds_cat, y_test_cat, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(
                structure["matrix_path"], f"{proc_name}_{model_name}_{key}.xlsx"
            )
        )


def score_keras(model, model_name, X_test, y_test, structure, proc_name):
    preds = {}

    for i in range(y_test.shape[1]):
        preds[i] = model[i].predict(X_test, verbose=0).argmax(axis=1)
    preds = DataFrame(preds)
    print(preds.shape)
    f1_macro_res = columnwise_score(f1_score, preds, y_test, average="macro")
    fname = os.path.join(structure["root"], f"{proc_name}_{model_name}_f1.csv")
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds, y_test, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(
                structure["matrix_path"], f"{proc_name}_{model_name}_{key}.xlsx"
            )
        )


def save_cv_results(
    results: DataFrame,
    model_name: str,
    structure: Dict[str, str],
    proc_name: str,
) -> None:
    filename = os.path.join(
        structure["cv_results_path"], f"{proc_name}_{model_name}_cv_results.csv"
    )
    results.to_csv(filename)


def save_history(
    history: Any,
    model_name: str,
    structure: Dict[str, str],
    proc_name: str,
) -> None:
    for col, item in history.items():
        history_ = DataFrame(item.history)
        filename = os.path.join(
            structure["history_path"], f"{proc_name}_{model_name}_{col}_history.csv"
        )
        history_.to_csv(filename)


def save_model(
    model: Any,
    model_name: str,
    structure: Dict[Any, Any],
    proc_name: str,
):
    filename = os.path.join(
        structure["model_path"], f"{proc_name}_{model_name}_model.pkl"
    )
    joblib.dump(model, filename)


def save_model_keras(
    model: Any,
    model_name: str,
    structure: Dict[str, str],
    proc_name: str,
):
    for i, model_ in enumerate(model):
        filename = os.path.join(
            structure["model_path"], f"{proc_name}_{model_name}_{i}_model.h5"
        )
        model_.save(filename)


def save_vars(config: dict, proc_name: str, model_name: str, structure: dict) -> None:
    config["dttm"] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(
        os.path.join(structure["vars_path"], f"vars_{proc_name}_{model_name}.yaml"),
        "w",
        encoding="utf-8",
    ) as file:
        yaml.dump(config, file)


from scripts.models.factory import ModelFactory


def check_config(config: Dict[str, Any], factory: ModelFactory):
    length = 0
    not_in_factory = []
    for item in config["models"]:
        if item not in factory:
            not_in_factory.append(item)
            length += 1

    if length > 0:
        raise ValueError(f'Models {",".join(not_in_factory)} are not implemented')
