import pytest

from ..models.factory import *
from ..models import sk_model_factory

def test_model_factory():
    factory = ModelFactory()

    def build_string(name):
        return f'String {name}'
    factory.register_builder(build_string, 'string')

    string_bld = factory._safe_get('string')
    assert string_bld('sample') == 'String sample'

    string_bld = factory.get_builder('string')(name='sample')
    assert string_bld == 'String sample'

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
    
    def rf():
        return RandomForestClassifier()

    factory.register_builder(get_xgboost, 'xgboost')
    factory.register_builder(rf)

    assert 'xgboost' in factory
    assert 'rf' in factory

    booster = factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

    booster = factory.get_builder('xgboost')(random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

def test_keras_model_factory():

    factory = KerasModelFactory()

    def get_xgboost(**kwargs):
        return XGBClassifier(**kwargs)
    
    def rf(**kwargs):
        return RandomForestClassifier(**kwargs)

    factory.register_builder(get_xgboost, 'xgboost')
    factory.register_builder(rf)

    assert 'xgboost' in factory
    assert 'rf' in factory

    booster = factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

def test_cv_factory():

    factory = CVFactory()

    def get_xgboost(**kwargs):
        return XGBClassifier(**kwargs)
    
    def rf(**kwargs):
        return RandomForestClassifier(**kwargs)

    factory.register_builder(get_xgboost, 'xgboost')
    factory.register_builder(rf)

    assert 'xgboost' in factory
    assert 'rf' in factory

    booster = factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

def test_gcv_factory():

    factory = GCVFactory()

    def get_xgboost(**kwargs):
        return XGBClassifier(**kwargs)
    
    def rf(**kwargs):
        return RandomForestClassifier(**kwargs)

    factory.register_builder(get_xgboost, 'xgboost')
    factory.register_builder(rf)

    assert 'xgboost' in factory
    assert 'rf' in factory

    booster = factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

def test_register_model_error():
    with pytest.raises(ValueError):
        @register_model(1, 'xgboost')
        def get_xgboost():
            return XGBClassifier()

def test_register_model():
    @register_model(sk_model_factory, 'xgboost')
    def get_xgboost():
        return XGBClassifier()
    
    assert 'xgboost' in sk_model_factory

    booster = sk_model_factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

def test_register_model_none():
    @register_model(sk_model_factory)
    def xgboost():
        return XGBClassifier()
    
    booster = sk_model_factory.get('xgboost', random_state=17)
    assert isinstance(booster, XGBClassifier)
    assert booster.random_state == 17

    
    

