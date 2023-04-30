import os
import logging

from pandas import concat
from sklearn.metrics import f1_score

from scripts.helpers.utils import validate_keras, add_to_environ
from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.helpers.yaml_utils import load_yaml, dict_to_yaml_str
from scripts.models import nn_model_factory

from run_utils import fit_keras, get_data_pipeline, save_history, save_model_keras, score_keras, read_data
from run_utils import create_folder_structure, save_model_keras, save_vars, save_cv_results
from run_utils import check_config

PROC_NAME = os.path.basename(__file__).split('.')[0]

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)
    add_to_environ(arguments.conf)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)
    
    vars_name = f'vars_{PROC_NAME}.yaml'
    vars_path = os.path.join('vars', vars_name) if arguments.vars is None else arguments.vars
    config_global = load_yaml(vars_path)
    check_config(config_global, nn_model_factory)

    df_train, df_test, df_val, categories = read_data(**config_global["data"], val=True)

    for model_name, config in config_global['models'].items():
        if arguments.model is not None and arguments.model != model_name:
            continue

        logger.info(f'Model {model_name}, params:')
        logger.info(dict_to_yaml_str(config))
        config['best_params'] = {}

        logger.info(f'Data processing for {model_name}')
        data_pipeline = get_data_pipeline(config)
        X_train, y_train, features = data_pipeline.fit_transform(df_train)
        X_val, y_val, features = data_pipeline.transform(df_val)
        X_test, y_test, features = data_pipeline.transform(df_test)

        df_train_full = concat([df_train, df_val], ignore_index=True)
        X_train_full, y_train_full, features = data_pipeline.fit_transform(df_train_full)
        X_test_full, y_test_full, features = data_pipeline.transform(df_test)

        shape = X_train.shape[1: ]
        config['features'] = list(features)
        logger.info(f'X_train shape {X_train.shape}')
        logger.info(f'X_val shape {X_val.shape}')
        logger.info(f'X_test shape {X_test.shape}')
        logger.info(f'X_train_full shape {X_train_full.shape}')

        logger.info(f'Grid search model, {model_name}')
        model_fn = nn_model_factory.get_builder(model_name)

        init_params = config['init_params'].copy()
        init_params['input_shape'] = shape
        init_params['n_classes'] = len(categories)
        best_params, best_score, results = validate_keras(model_fn, init_params,
                                                          config['param_grids'],
                                                          f1_score, X_train, y_train[:, 0],
                                                          X_val, y_val[:, 0],
                                                          config['callback_params'],
                                                          config['scoring_params'],
                                                          config['fit_params'],
                                                          **config['gv_params'])
        save_cv_results(results, model_name, structure, PROC_NAME)

        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')
        config['best_params'] = best_params
        final_params = init_params.copy()
        final_params.update(best_params)

        logger.info(f'Fitting model, {model_name}')
        model, history = fit_keras(model_name, final_params, config['fit_params'],
                                   config['callback_params'],
                                   X_train_full, y_train_full, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test_full, y_test_full, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)

        save_vars(config, PROC_NAME, model_name, structure)
 