import os
import joblib
import yaml
from datetime import datetime

from pandas import DataFrame, read_csv
from pandas import get_dummies
from numpy import array

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from tensorflow.keras import callbacks as callbacks
try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import random as random
    set_seed = random.set_seed

from scripts.pipeline.preprocess import preprocess_3h
from scripts.helpers.utils import columnwise_score, columnwise_confusion_matrix, create_folder
from scripts.models.models import *


MODEL_DICT = {
    'xgboost': XGBClassifier(),
    'randomforest': RandomForestClassifier(),
    'ridge': make_pipeline(StandardScaler(), RidgeClassifier()),
    'dummy': DummyClassifier(strategy='most_frequent')
}

NN_MODEL_DICT = {
    'perceptron': get_sequential_model,
    'lstm': get_lstm_model,
    "gru": get_gru_model,
}

def read_data(val=False):
    df = read_csv(
        './data/All_browse_data_без_погружения_19971021_20211231_с_пропусками.csv', 
        encoding='cp1251', na_values='N').pipe(preprocess_3h)
    
    categories = list(df.category.unique())

    df = df.set_index('dttm')

    df_test = df.last('730d')
    df_train = df.drop(df_test.index)

    if val:
        df_val = df_train.last('730d')
        df_train = df_train.drop(df_val.index)
        return df_train.reset_index(), df_val.reset_index(), df_test.reset_index(), categories
    else:
        return df_train.reset_index(), df_test.reset_index(), categories

from importlib import import_module

def get_data_pipeline(config):
    cls = getattr(import_module('scripts.pipeline.data_pipe'), 
                  config['pipe_name'])
    return cls(**config['pipe_params'])

def create_folder_structure(root):
    matrix_path = os.path.join(root, 'matrix')
    model_path = os.path.join(root, 'models')
    history_path = os.path.join(root, 'history')
    vars_path = os.path.join(root, 'vars')
    create_folder(root)
    create_folder(matrix_path)
    create_folder(model_path)
    create_folder(history_path)
    create_folder(vars_path)
    structure = dict(root=root, model_path=model_path, vars=vars_path,
                     matrix_path=matrix_path, history_path=history_path)
    return structure

def grid_search(params, model_name, init_params, X_train, y_train, 
                cv_params, gcv_params):
    model = MODEL_DICT[model_name].set_params(**init_params)
    skf = StratifiedKFold(**cv_params)
    gcv = GridSearchCV(model, params, cv=skf, **gcv_params)
    gcv.fit(X_train, y_train)
    return gcv.best_params_, gcv.best_score_

def fit(model_name, params, X_train, y_train):
    model = MultiOutputClassifier(MODEL_DICT[model_name].set_params(**params))
    model.fit(X_train, y_train)
    return model

def fit_keras(model_name, input_shape, n_classes, init_params, 
              fit_params, callback_params, X_train, y_train, seed):
    
    models = []
    histories = {}

    for col in range(y_train.shape[1]):
        set_seed(seed)
        y_dummy = array(get_dummies(y_train[:, col]))
        model = NN_MODEL_DICT[model_name](input_shape, n_classes, **init_params)
        callbacks_list = [
            callbacks.EarlyStopping(**callback_params),
        ]
        history = model.fit(X_train, y_dummy, 
                            callbacks=callbacks_list, **fit_params)
        models.append(model)
        histories[col] = history
    
    return models, histories

def score(model, model_name, X_test, y_test, structure, proc_name):
    preds = model.predict(X_test)
    f1_macro_res = columnwise_score(f1_score, preds, y_test, average='macro')
    fname = os.path.join(structure['root'], f'{proc_name}_{model_name}_f1.csv')
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds, y_test, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(structure['matrix_path'], f'{proc_name}_{model_name}_{key}.xlsx'))

def score_keras(model, model_name, X_test, y_test, structure, proc_name):
    preds= {}

    for i in range(y_test.shape[1]):
        preds[i] = model[i].predict(X_test).argmax(axis=1)
    preds = DataFrame(preds)
    f1_macro_res = columnwise_score(f1_score, preds, y_test, average='macro')
    fname = os.path.join(structure['root'], f'{proc_name}_{model_name}_f1.csv')
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds, y_test, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(structure['matrix_path'], f'{proc_name}_{model_name}_{key}.xlsx'))

def save_history(history, model_name: str, structure: str, proc_name: str,) -> None:
    for col, item in history.items():
        history_ = DataFrame(item.history)
        filename = os.path.join(structure['history_path'], 
                                f'{proc_name}_{model_name}_{col}_history.csv')
        history_.to_csv(filename)

def save_model(model, model_name: str, structure: str, proc_name: str,):
    filename = os.path.join(structure['model_path'], 
                            f'{proc_name}_{model_name}_model.pkl')
    joblib.dump(model, filename)

def save_model_keras(model, model_name: str, structure: str, proc_name: str,):

    for i, model_ in enumerate(model):
        filename = os.path.join(structure['model_path'], 
                                f'{proc_name}_{model_name}_{i}_model')
        model_.save(filename)

def save_vars(config: dict, proc_name: str, model_name: str, structure: dict) -> None:
    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(structure["vars"], f'vars_{proc_name}_{model_name}.yaml'),
            'w', encoding='utf-8') as file:
        yaml.dump(config, file)    
