from ..models.model_sklearn import *
from ..models import sk_model_factory

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