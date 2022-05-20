from pandas import read_csv
from numpy import sqrt, array

from ..models.column_estimator import ColumnEstimator
from ..pipeline.data_pipe import LagDataPipe
from ..pipeline.preprocess import preprocess_3h, categorize
from ..helpers.yaml_utils import load_yaml


def test_column_estimator():
    CONF = load_yaml('scripts/tests/test_yamls/test_vars.yaml')
    ce = ColumnEstimator(0)
    df = read_csv('scripts/tests/test_data/test.csv').pipe(preprocess_3h)
    data_pipe = LagDataPipe(**CONF)
    X_train, y_train, features = data_pipe.fit_transform(df)
    X_train, y_train, features = data_pipe.transform(df)
    ce.fit(X_train, y_train)
    preds = ce.predict(X_train)
    true_preds = array(list(map(categorize, list(X_train[:, 0]))))

    assert sqrt(((preds - true_preds) ** 2).sum()) < 1e-6

    