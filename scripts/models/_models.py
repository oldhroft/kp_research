from .factory import CVFactory, SklearnModelFactory, KerasModelFactory

sk_model_factory = SklearnModelFactory()
nn_model_factory = KerasModelFactory()
cv_factory = CVFactory()

from .models import SkLearnModels, KerasModels, CV