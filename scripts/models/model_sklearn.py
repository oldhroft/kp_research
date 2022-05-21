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
from ._models import sk_model_factory
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
