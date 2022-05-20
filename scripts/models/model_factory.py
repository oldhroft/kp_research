class ModelFactory(object):

    def __init__(self) -> None:
        self._builders = {}
    
    def register_builder(self, key, builder):
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

class SklearnModelFactory(ModelFactory):
    def get(self, key, **kwargs):
        return self._safe_get(key)().set_params(**kwargs)

sk_model_factory = SklearnModelFactory()

def register_model(name, factory):
    def _register_model(builder):
        if factory == 'sklearn':
            sk_model_factory.register_builder(name, builder)
        return builder
    return _register_model






