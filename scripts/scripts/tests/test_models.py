import os

from scripts.models.models import *
from scripts.models import sk_model_factory, nn_model_factory, cv_factory, gcv_factory
from scripts.helpers.yaml_utils import load_yaml

LOCATION = os.path.dirname(os.path.realpath(__file__))


def test_sklearn_models():

    method_list = [
        func
        for func in dir(SkLearnModels)
        if callable(getattr(SkLearnModels, func)) and not func.startswith("__")
    ]

    for method in method_list:
        assert method in sk_model_factory


def test_sklearn_models_build():

    method_list = [
        func
        for func in dir(SkLearnModels)
        if callable(getattr(SkLearnModels, func)) and not func.startswith("__")
    ]

    for method in method_list:
        assert hasattr(getattr(SkLearnModels, method)(), "fit")
        assert hasattr(getattr(SkLearnModels, method)(), "predict")


CONF = load_yaml(os.path.join(LOCATION, "test_yamls/nn_conf.yaml"))


def test_model_buold_gru():
    model = KerasModels.gru(**CONF["rnn"]["init_params"])
    assert len(model.layers) == 4, "GRU shape not correct"


def test_model_buold_lstm():
    model = KerasModels.lstm(**CONF["rnn"]["init_params"])
    assert len(model.layers) == 4, "LSTM shape not correct"


def test_model_buold_perceptron():
    model = KerasModels.perceptron(**CONF["perceptron"]["init_params"])
    assert len(model.layers) == 3, "Perceptron shape not correct"


def test_model_buold_gru_from_factory():
    model = nn_model_factory.get("gru", **CONF["rnn"]["init_params"])
    assert len(model.layers) == 4, "GRU shape not correct"


def test_model_buold_lstm_from_factory():
    model = nn_model_factory.get("lstm", **CONF["rnn"]["init_params"])
    assert len(model.layers) == 4, "LSTM shape not correct"


def test_model_buold_bi_gru_from_factory():
    model = nn_model_factory.get("bi_gru", **CONF["rnn"]["init_params"])
    assert len(model.layers) == 4, "GRU shape not correct"


def test_model_buold_bi_lstm_from_factory():
    model = nn_model_factory.get("bi_lstm", **CONF["rnn"]["init_params"])
    assert len(model.layers) == 4, "LSTM shape not correct"


def test_model_buold_perceptron_from_factory():
    model = nn_model_factory.get("perceptron", **CONF["perceptron"]["init_params"])
    assert len(model.layers) == 3, "Perceptron shape not correct"


def test_cv():

    method_list = [
        func
        for func in dir(CV)
        if callable(getattr(CV, func)) and not func.startswith("__")
    ]

    for method in method_list:
        assert method in cv_factory


from sklearn.model_selection import BaseCrossValidator


def test_cv_build():
    method_list = [
        func
        for func in dir(CV)
        if callable(getattr(CV, func)) and not func.startswith("__")
    ]

    for method in method_list:
        assert isinstance(getattr(CV, method)(n_splits=3), BaseCrossValidator)


def test_gcv():

    method_list = [
        func
        for func in dir(GCV)
        if callable(getattr(GCV, func)) and not func.startswith("__")
    ]

    for method in method_list:
        assert method in gcv_factory


from sklearn.ensemble import RandomForestClassifier


def test_gcv_build():
    model = RandomForestClassifier()
    grid = {"max_depth": [2, 4, 5, 6, 7]}
    method_list = [
        func
        for func in dir(GCV)
        if callable(getattr(GCV, func)) and not func.startswith("__")
    ]

    for method in method_list:
        assert hasattr(getattr(GCV, method)(estimator=model, param_grid=grid), "fit")
        assert hasattr(
            getattr(GCV, method)(estimator=model, param_grid=grid), "predict"
        )
