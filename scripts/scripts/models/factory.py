from typing import Dict, Callable

class ModelFactory(object):

    def __init__(self) -> None:
        self._builders: Dict[str, Callable] = {}
    
    def register_builder(self, builder: Callable, key: str=None):
        if key is None:
            key = builder.__name__
        self._builders[key] = builder
    
    def __contains__(self, item: str):
        return item in self._builders
    
    def _safe_get(self, key: str):
        if key in self:
            return self._builders[key]
        else:
            raise ValueError(f'No model named {key} registered')
    
    def get(self, key: str, **kwargs):
        return self._safe_get(key)(**kwargs)
    
    def get_builder(self, key: str):
        return lambda **kwargs: self.get(key, **kwargs)


class SklearnModelFactory(ModelFactory):
    def get(self, key, **kwargs):
        return self._safe_get(key)().set_params(**kwargs)

class KerasModelFactory(ModelFactory):
    pass

class CVFactory(ModelFactory):
    pass

class GCVFactory(ModelFactory):
    pass

def register_model(factory, name=None):
    def _register_model(builder):

        if not isinstance(factory, ModelFactory):
            raise ValueError('Provided value is not a model facotory')

        factory.register_builder(builder, name)
        return builder
    return _register_model
    