class ModelFactory(object):

    def __init__(self) -> None:
        self._builders = {}
    
    def register_builder(self, builder, key=None):
        if key is None:
            key = builder.__name__
        self._builders[key] = builder
    
    def __contains__(self, item):
        return item in self._builders
    
    def _safe_get(self, key):
        if key in self:
            return self._builders[key]
        else:
            raise ValueError(f'No model named {key} registered')
    
    def get(self, key, **kwargs):
        return self._safe_get(key)(**kwargs)
    
    def get_builder(self, key):
        return lambda **kwargs: self.get(key, **kwargs)


class SklearnModelFactory(ModelFactory):
    def get(self, key, **kwargs):
        return self._safe_get(key)().set_params(**kwargs)

class KerasModelFactory(ModelFactory):
    pass


def register_model(factory, name=None):
    def _register_model(builder):

        if not isinstance(factory, ModelFactory):
            raise ValueError('Provided value is not a model facotory')

        factory.register_builder(builder, name)
        return builder
    return _register_model
    