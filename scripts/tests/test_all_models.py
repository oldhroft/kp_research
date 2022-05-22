from ..models import sk_model_factory, nn_model_factory

def test_sk_model_factory_build():

    assert 'xgboost' in sk_model_factory
    assert 'randomforest' in sk_model_factory
    assert 'lr' in sk_model_factory

def test_nn_model_factory_build():

    assert 'gru' in nn_model_factory
    assert 'lstm' in nn_model_factory
    assert 'perceptron' in nn_model_factory
