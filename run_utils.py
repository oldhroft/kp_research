import os

from pandas import DataFrame, read_csv
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import categorize, preprocess_3h
from utils import columnwise_score, columnwise_confusion_matrix

MODEL_DICT = {
    'xgboost': XGBClassifier(),
    'randomforest': RandomForestClassifier(),
    'ridge': make_pipeline(StandardScaler(), RidgeClassifier()),
}

def read_data():
    df = read_csv(
        'All_browse_data_без_погружения_19971021_20211231_с_пропусками.csv', 
        encoding='cp1251', na_values='N').pipe(preprocess_3h)
    
    categories = list(df.category.unique())
    return df, sorted(categories)

def fit(model_name, params, X_train, y_train):
    model = MultiOutputClassifier(MODEL_DICT[model_name].set_params(**params))
    model.fit(X_train, y_train)
    return model

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