from turtle import shape
from pandas import read_csv

from scripts.pipeline.data_pipe import *
from scripts.pipeline.preprocess import _select

from scripts.pipeline.preprocess import preprocess_3h

DF = read_csv("scripts/tests/test_data/test.csv").pipe(preprocess_3h)


def test_simple_pipe():
    steps = [(_select, {"columns": ["Kp*10", "doyCos", "doySin"]}, False)]
    pipe = DataPipe(steps)
    result = pipe.fit_transform(DF)

    assert result.shape == (DF.shape[0], 3)


from scripts.helpers.yaml_utils import load_yaml

CONF = load_yaml("scripts/tests/test_yamls/test_vars.yaml")


def test_lag_data_pipe():
    data_pipe = LagDataPipe(**CONF)
    X_train, y_train, features = data_pipe.fit_transform(DF)
    X_train, y_train, features = data_pipe.transform(DF)
    n_records = DF.shape[0] - CONF["forward_steps"] - CONF["backward_steps"]
    assert y_train.shape == (n_records, CONF["forward_steps"])
    n_features = (CONF["backward_steps"] + 1) * len(CONF["variables"])
    assert X_train.shape == (n_records, n_features)
    assert len(features) == n_features


def test_sequence_data_pipe():
    data_pipe = SequenceDataPipe(**CONF)
    X_train, y_train, features = data_pipe.fit_transform(DF)
    X_train, y_train, features = data_pipe.transform(DF)
    n_records = DF.shape[0] - CONF["forward_steps"] - CONF["backward_steps"]
    assert y_train.shape == (n_records, CONF["forward_steps"])
    n_features = len(CONF["variables"])
    time_steps = CONF["backward_steps"] + 1
    assert X_train.shape == (n_records, time_steps, n_features)
    assert len(features) == n_features
