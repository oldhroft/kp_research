from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L


def get_sequential_model(input_shape: tuple, n_classes: int=3, units_array: list=[10], 
                         optimizer: str='adam') -> Sequential:

    model = Sequential([
        L.Input(shape=input_shape),
        *(L.Dense(units=units, activation='relu') for units in units_array),
        L.Dense(units=n_classes, activation='softmax')  
    ])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def get_lstm_model(input_shape, n_classes, units_array, optimizer, ):
    
    model = Sequential([
        L.Input(shape=input_shape),
        *(L.LSTM(i, return_sequences=True, ) 
          for i in units_array['rnn'][:-1]),
        L.LSTM(units_array['rnn'][-1]),
        *(L.Dense(units=units, activation='relu') 
          for units in units_array['dense']),
        L.Dense(n_classes, activation='softmax')
    ], )
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def get_gru_model(input_shape, n_classes, units_array, optimizer, ):
    
    model = Sequential([
        L.Input(shape=input_shape),
        *(L.GRU(i, return_sequences=True, ) 
          for i in units_array['rnn'][:-1]),
        L.GRU(units_array['rnn'][-1]),
        *(L.Dense(units=units, activation='relu') 
          for units in units_array['dense']),
        L.Dense(n_classes, activation='softmax')
    ], )
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

NN_MODEL_DICT = {
    'perceptron': get_sequential_model,
    'lstm': get_lstm_model,
    "gru": get_gru_model,
}

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
from .column_estimator import ColumnEstimator

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

MODEL_DICT = {
    'xgboost': XGBClassifier(),
    'randomforest': RandomForestClassifier(),
    'ridge': make_pipeline(StandardScaler(), RidgeClassifier()),
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'dummy': DummyClassifier(strategy='most_frequent'),
    'catboost': CatBoostClassifier(),
    'lightgbm': LGBMClassifier(),
    'rf_xgboost': make_pipeline(
        SelectFromModel(RandomForestClassifier(random_state=17)),
        XGBClassifier()
    ),
    'rf_lr': make_pipeline(
        SelectFromModel(RandomForestClassifier(random_state=17)),
        LogisticRegression()
    ),
    'columnestimator': ColumnEstimator(),
    'smote_randomforest': make_pipeline_imb(
        SMOTE(), RandomForestClassifier()
    ),
    'smote_xgboost': make_pipeline_imb(
        SMOTE(), XGBClassifier()
    ),
    'smote_ridge': make_pipeline_imb(
        SMOTE(), StandardScaler(), RidgeClassifier()
    ),
    'smote_lr': make_pipeline_imb(
        SMOTE(), StandardScaler(), LogisticRegression()
    ),
    'smote_catboost': make_pipeline_imb(
        SMOTE(), CatBoostClassifier()
    ), 
    'smote_lightgbm': make_pipeline_imb(
        SMOTE(), LGBMClassifier(),
    ), 

}

        