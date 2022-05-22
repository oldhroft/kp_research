from .factory import SklearnModelFactory, KerasModelFactory

sk_model_factory = SklearnModelFactory()
nn_model_factory = KerasModelFactory()

from .model_sklearn import SkLearnModels, KerasModels