import os
from datetime import datetime
import joblib
import yaml

from numpy import array, squeeze
from pandas import DataFrame, get_dummies, read_csv
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras import callbacks as callbacks

try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import random as random
    set_seed = random.set_seed

from scripts.helpers.utils import (columnwise_confusion_matrix,
                                   columnwise_score, create_folder)
from scripts.models import nn_model_factory, sk_model_factory
from scripts.pipeline.preprocess import preprocess_3h


def read_data(path, val=False):
    if path is None:
        path = './data/All_browse_data_без_погружения_19971021_20211231_с_пропусками.csv'
    df = read_csv(
        path, encoding='cp1251', na_values='N').pipe(preprocess_3h)
    
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

def create_folder_structure(root: str) -> dict:

    os.environ['FOLDER'] = root
    structure = {'root': root}
    create_folder(root)
    for sub_folder in ['matrix', 'model', 'history', 'vars','cv_results']:
        path = os.path.join(root, sub_folder)
        create_folder(path)
        structure[f'{sub_folder}_path'] = path

    return structure

def _convert_to_results(gcv: GridSearchCV) -> DataFrame:
    cv_results = gcv.cv_results_
    n_splits = gcv.n_splits_
    
    results = []
    
    for j, param in enumerate(cv_results['params']):
        
        for i in range(n_splits):
            score = cv_results[f'split{i}_test_score'][j]
            
            results.append({
                **param, 'score': score, 'split': i,
            })
    
    return DataFrame(results)

def grid_search(params, model_name, init_params, X_train, y_train, 
                cv_params, gcv_params):

    model = sk_model_factory.get(model_name, **init_params)
                
    skf = StratifiedKFold(**cv_params)
    gcv = GridSearchCV(model, params, cv=skf, **gcv_params)
    gcv.fit(X_train, y_train)

    results = _convert_to_results(gcv)

    return gcv.best_params_, gcv.best_score_, results

def fit(model_name, init_params, X_train, y_train):

    model = sk_model_factory.get(model_name, **init_params)
    model = MultiOutputClassifier(model)
    model.fit(X_train, y_train)
    return model

def fit_keras(model_name, init_params, fit_params, callback_params, 
              X_train, y_train, seed):
    
    models = []
    histories = {}

    for col in range(y_train.shape[1]):
        set_seed(seed)
        y_dummy = array(get_dummies(y_train[:, col]))
        model = nn_model_factory.get(model_name, **init_params)
        callbacks_list = [
            callbacks.EarlyStopping(**callback_params),
        ]
        history = model.fit(X_train, y_dummy, 
                            callbacks=callbacks_list, **fit_params)
        models.append(model)
        histories[col] = history
    
    return models, histories

def score(model, model_name, X_test, y_test, structure, proc_name):
    preds = squeeze(model.predict(X_test))
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

def save_cv_results(results: DataFrame, model_name: str, structure: str, proc_name: str,) -> None:
    filename = os.path.join(structure['cv_results_path'], 
                            f'{proc_name}_{model_name}_cv_results.csv')
    results.to_csv(filename)

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
    with open(os.path.join(structure["vars_path"], f'vars_{proc_name}_{model_name}.yaml'),
            'w', encoding='utf-8') as file:
        yaml.dump(config, file)

from scripts.models.factory import ModelFactory
def check_config(config: dict, factory: ModelFactory):
    length = 0
    not_in_factory = []
    for item in config['models']:
        if item not in factory:
            not_in_factory.append(item)
            length += 1
    
    if length > 0:
        raise ValueError(f'Models {",".join(not_in_factory)} are not implemented')
