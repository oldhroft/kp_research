from ..models.model_sklearn import *
from ..models import sk_model_factory, nn_model_factory

from ..helpers.yaml_utils import load_yaml

def test_sklearn_models():

    method_list = [
        func for func in dir(SkLearnModels) 
        if callable(getattr(SkLearnModels, func)) and not func.startswith("__")]
    
    for method in method_list:
        assert method in sk_model_factory

def test_sklearn_models_build():

    method_list = [
        func for func in dir(SkLearnModels) 
        if callable(getattr(SkLearnModels, func)) and not func.startswith("__")]
    
    for method in method_list:
        assert hasattr(getattr(SkLearnModels, method)(), 'fit')
        assert hasattr(getattr(SkLearnModels, method)(), 'predict')

CONF = load_yaml('scripts/tests/test_yamls/nn_conf.yaml')

def test_model_buold_gru():
    model = KerasModels.gru(**CONF['rnn']['init_params'])
    assert len(model.layers) == 4, "GRU shape not correct"

def test_model_buold_lstm():
    model = KerasModels.lstm(**CONF['rnn']['init_params'])
    assert len(model.layers) == 4, "LSTM shape not correct"

def test_model_buold_perceptron():
    model = KerasModels.perceptron(**CONF['perceptron']['init_params'])
    assert len(model.layers) == 3, "Perceptron shape not correct"

def test_model_buold_gru_from_factory():
    model = nn_model_factory.get("gru", **CONF['rnn']['init_params'])
    assert len(model.layers) == 4, "GRU shape not correct"

def test_model_buold_lstm_from_factory():
    model = nn_model_factory.get("lstm", **CONF['rnn']['init_params'])
    assert len(model.layers) == 4, "LSTM shape not correct"

def test_model_buold_perceptron_from_factory():
    model = nn_model_factory.get("perceptron", **CONF['perceptron']['init_params'])
    assert len(model.layers) == 3, "Perceptron shape not correct"