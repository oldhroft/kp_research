import os
import logging
from sklearn.metrics import f1_score

from scripts.helpers.utils import add_to_environ, validate_keras_cv
from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.helpers.yaml_utils import load_yaml, dict_to_yaml_str
from scripts.models import nn_model_factory, cv_factory

from run_utils import (
    fit_keras,
    save_history,
    score_keras,
    build_data_pipelines,
)
from run_utils import create_folder_structure
from run_utils import save_vars, save_cv_results, save_model_keras, check_config

PROC_NAME = os.path.basename(__file__).split(".")[0]

if __name__ == "__main__":
    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)
    add_to_environ(arguments.conf)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_name = f"vars_{PROC_NAME}.yaml"
    vars_path = (
        os.path.join("vars", vars_name) if arguments.vars is None else arguments.vars
    )
    config_global = load_yaml(vars_path)
    check_config(config_global, nn_model_factory)

    data_pipelines = build_data_pipelines(
        config_global["models"], structure
    )

    for model_name, config in config_global["models"].items():
        if arguments.model is not None and arguments.model != model_name:
            continue

        pipe = data_pipelines[config["pipe_name"]]
        X_train, y_train, X_test, y_test = pipe.get_xy()

        logger.info(f"Model {model_name}, params:")
        logger.info(dict_to_yaml_str(config))
        config["best_params"] = {}

        pipe = data_pipelines[config["pipe_name"]]
        X_train, y_train, X_test, y_test = pipe.get_xy()

        logger.info(f"X_train shape {X_train.shape}")
        logger.info(f"X_test shape {X_test.shape}")
        config["features"] = list(pipe.features)

        shape = X_train.shape[1:]

        logger.info(f"Grid search model, {model_name}")
        model_fn = nn_model_factory.get_builder(model_name)
        init_params = config["init_params"].copy()
        init_params["input_shape"] = shape
        init_params["n_classes"] = len(pipe.categories)
        cv = cv_factory.get(config["cv"], **config["cv_params"])
        best_params, best_score, results = validate_keras_cv(
            model_fn,
            init_params,
            cv,
            config["param_grids"],
            f1_score,
            X_train,
            y_train[:, 0],
            config["callback_params"],
            config["scoring_params"],
            config["fit_params"],
            **config["gcv_params"],
        )
        save_cv_results(results, model_name, structure, PROC_NAME)
        config["best_params"] = best_params
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best score: {best_score}")

        logger.info(f"Fitting model, {model_name}")
        final_params = init_params.copy()
        final_params.update(best_params)
        model, history = fit_keras(
            model_name,
            final_params,
            config["fit_params"],
            config["callback_params"],
            X_train,
            y_train,
            config["seed"],
        )

        logger.info(f"Scoring model, {model_name}")
        score_keras(model, model_name, X_test, y_test, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)

        save_vars(config, PROC_NAME, model_name, structure)
