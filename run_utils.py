import os

from pandas import DataFrame, read_csv
from pandas import get_dummies
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from tensorflow.keras import callbacks as callbacks
try:
    from tensorflow.random import set_seed
except ImportError:
    from tensorflow import random as random
    set_seed = random.set_seed


from scripts.helpers.preprocess import preprocess_3h
from scripts.helpers.utils import columnwise_score, columnwise_confusion_matrix
from scripts.helpers.utils_nn import get_sequential_model

MODEL_DICT = {
    'xgboost': XGBClassifier(),
    'randomforest': RandomForestClassifier(),
    'ridge': make_pipeline(StandardScaler(), RidgeClassifier()),
    'dummy': DummyClassifier(strategy='most_frequent')
}

NN_MODEL_DICT = {
    'perceptron': get_sequential_model
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

def fit(model_name, params, X_train, y_train):
    model = MultiOutputClassifier(MODEL_DICT[model_name].set_params(**params))
    model.fit(X_train, y_train)
    return model

def fit_keras(model_name, input_shape, n_classes, init_params, 
              fit_params, callback_params, X_train, y_train, seed):
    
    models = []
    histories = {}
    y_train = DataFrame(y_train)
    for col in y_train.columns:
        set_seed(seed)
        y_dummy = get_dummies(y_train[col]).values
        model = NN_MODEL_DICT[model_name](input_shape, n_classes, **init_params)
        callbacks_list = [
            callbacks.EarlyStopping(**callback_params),
        ]
        history = model.fit(X_train, y_dummy, callbacks=callbacks_list, **fit_params)
        models.append(model)
        histories[col] = history
    
    return models, histories

def score(model, model_name, X_test, y_test, root, matrix_path, proc_name):
    preds = model.predict(X_test)
    preds = DataFrame(preds)
    f1_macro_res = columnwise_score(f1_score, preds, y_test, average='macro')
    fname = os.path.join(root, f'{proc_name}_{model_name}_f1.csv')
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds, y_test, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(matrix_path, f'{proc_name}_{model_name}_{key}.xlsx'))

def score_keras(model, model_name, X_test, y_test, root, matrix_path, proc_name):
    preds= {}
    y_test = DataFrame(y_test)
    for i, col in enumerate(y_test.columns):
        preds[col] = model[i].predict(X_test).argmax(axis=1)
    preds = DataFrame(preds)
    f1_macro_res = columnwise_score(f1_score, preds, y_test, average='macro')
    fname = os.path.join(root, f'{proc_name}_{model_name}_f1.csv')
    f1_macro_res.to_csv(fname)
    confusion_matrices = columnwise_confusion_matrix(preds, y_test, [0, 1, 2])
    for key, matrix in confusion_matrices.items():
        matrix.to_excel(
            os.path.join(matrix_path, f'{proc_name}_{model_name}_{key}.xlsx'))

def save_history(history, model_name: str, history_path: str, proc_name: str,) -> None:
    for col, item in history.items():
        history_ = DataFrame(item.history)
        filename = os.path.join(history_path, f'{proc_name}_{model_name}_history.csv')
        history_.to_csv(filename)
