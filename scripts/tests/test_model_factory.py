import pytest

from ..models.model_factory import *

def test_model_factory():
    factory = ModelFactory()

    def build_string(name):
        return f'String {name}'
    factory.register_builder('string', build_string)

    string_bld = factory._safe_get('string')
    assert string_bld('sample') == 'String sample'

    string1 = factory.get('string', name='1')
    assert string1 == 'String 1'

    with pytest.raises(ValueError):
        factory.get('nothing')
    
    try:
        factory.get('nothing')
    except ValueError as e:
        assert str(e) == 'No model named nothing registered'

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def test_sklearn_model_factory():

    factory = SklearnModelFactory()

    def get_xgboost():
        return XGBClassifier()
    
    def get_rf():
        return RandomForestClassifier()

    factory.register_builder('xgboost', get_xgboost)
    factory.register_builder('rf', get_rf)

    assert 'xgboost' in factory
    assert 'rf' in factory

    booster = factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17


def test_register_model():
    @register_model('xgboost', 'sklearn')
    def get_xgboost():
        return XGBClassifier()
    
    assert 'xgboost' in sk_model_factory

    booster = sk_model_factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17
    
    

