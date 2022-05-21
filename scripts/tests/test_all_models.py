from ..models import sk_model_factory

def test_sk_model_factory_build():

    assert 'xgboost' in sk_model_factory
    assert 'randomforest' in sk_model_factory
    assert 'lr' in sk_model_factory