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

from .factory import register_model
from ._models import sk_model_factory, nn_model_factory, cv_factory, gcv_factory
from ..helpers.utils import decorate_class

@decorate_class(staticmethod)
@decorate_class(register_model(sk_model_factory))
class SkLearnModels():
    def xgboost():
        return XGBClassifier()

    def randomforest():
        return RandomForestClassifier()

    def ridge():
        return make_pipeline(StandardScaler(), RidgeClassifier())

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression())

    def dummy():
        return DummyClassifier()

    def catboost():
        return CatBoostClassifier()

    def lightgbm():
        return LGBMClassifier()

    def rf_xgboost():
        return make_pipeline(
            SelectFromModel(RandomForestClassifier(random_state=17)),
            XGBClassifier()
        )

    def rf_lr():
        return make_pipeline(
            SelectFromModel(RandomForestClassifier(random_state=17)),
            LogisticRegression()
        )

    def columnestimator():
        return ColumnEstimator()

    def smote_randomforest():
        return make_pipeline_imb(
            SMOTE(), RandomForestClassifier()
        )

    def smote_xgboost():
        return make_pipeline_imb(
            SMOTE(), XGBClassifier()
        )

    def smote_ridge():
        return make_pipeline_imb(
            SMOTE(), StandardScaler(), RidgeClassifier()
        )
    
    def smote_lr():
        return make_pipeline_imb(
            SMOTE(), StandardScaler(), LogisticRegression()
        )

    def smote_catboost():
        return make_pipeline_imb(
            SMOTE(), CatBoostClassifier()
        )
    
    def smote_lightgbm():
        return make_pipeline_imb(
            SMOTE(), LGBMClassifier(),
        )

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L

@decorate_class(staticmethod)
@decorate_class(register_model(nn_model_factory))
class KerasModels:
    def perceptron(input_shape: tuple, n_classes: int=3, units_array: list=[10], 
                    optimizer: str='adam') -> Sequential:

        model = Sequential([
            L.Input(shape=input_shape),
            *(L.Dense(units=units, activation='relu') for units in units_array),
            L.Dense(units=n_classes, activation='softmax')  
        ])

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model

    def lstm(input_shape, n_classes, units_array, optimizer, ):
        
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

    def gru(input_shape, n_classes, units_array, optimizer, ):
        
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
        
    def bi_lstm(input_shape, n_classes, units_array, optimizer, ):
        
        model = Sequential([
            L.Input(shape=input_shape),
            *(L.Bidirectional(L.LSTM(i, return_sequences=True, ))
            for i in units_array['rnn'][:-1]),
            L.Bidirectional(L.LSTM(units_array['rnn'][-1])),
            *(L.Dense(units=units, activation='relu') 
            for units in units_array['dense']),
            L.Dense(n_classes, activation='softmax')
        ], )
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def bi_gru(input_shape, n_classes, units_array, optimizer, ):
        
        model = Sequential([
            L.Input(shape=input_shape),
            *(L.Bidirectional(L.GRU(i, return_sequences=True, ))
            for i in units_array['rnn'][:-1]),
            L.Bidirectional(L.GRU(units_array['rnn'][-1])),
            *(L.Dense(units=units, activation='relu') 
            for units in units_array['dense']),
            L.Dense(n_classes, activation='softmax')
        ], )
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

@decorate_class(staticmethod)
@decorate_class(register_model(cv_factory))
class CV:

    def skf(**kwargs):
        return StratifiedKFold(**kwargs)
    
    def tss(**kwargs):
        return TimeSeriesSplit(**kwargs)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

@decorate_class(staticmethod)
@decorate_class(register_model(gcv_factory))
class GCV:

    def gcv(**kwargs):
        return GridSearchCV(**kwargs)
    
    def rscv(**kwargs):

        if 'param_grid' in kwargs:
            kwargs['param_distributions'] = kwargs['param_grid']
            kwargs.pop('param_grid')
            
        return RandomizedSearchCV(**kwargs)