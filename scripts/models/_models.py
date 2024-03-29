from scripts.models.factory import *

sk_model_factory = SklearnModelFactory()
nn_model_factory = KerasModelFactory()
cv_factory = CVFactory()
gcv_factory = GCVFactory()
sk_model_factory_reg = SklearnModelFactoryReg()

from scripts.models.models import SkLearnModels, KerasModels, CV, SkLearnModelsReg
