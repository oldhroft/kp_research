from ..models.models import *
from ..helpers.yaml_utils import load_yaml

CONF = load_yaml('scripts/tests/test_yamls/nn_conf.yaml')

def test_model_buold_gru():
    model = get_gru_model(**CONF['rnn']['init_params'])
    assert len(model.layers) == 4, "GRU shape not correct"

def test_model_buold_lstm():
    model = get_lstm_model(**CONF['rnn']['init_params'])
    assert len(model.layers) == 4, "LSTM shape not correct"

def test_model_buold_perceptron():
    model = get_sequential_model(**CONF['perceptron']['init_params'])
    assert len(model.layers) == 3, "Perceptron shape not correct"

    

