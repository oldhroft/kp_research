import os
import logging

from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.helpers.yaml_utils import load_yaml, dict_to_yaml_str
from scripts.helpers.utils import add_to_environ
from scripts.models import nn_model_factory

from run_utils import fit_keras, get_data_pipeline, save_history, read_data, score_keras
from run_utils import create_folder_structure, save_model_keras, save_vars, check_config

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
        
    df_train, df_test, categories = read_data(arguments.data)

    for model_name, config in config_global['models'].items():

        if arguments.model is not None and arguments.model != model_name:
            continue

        logger.info(f'Model {model_name}, params:')
        logger.info(dict_to_yaml_str(config))
        config['best_params'] = {}

        logger.info(f'Data processing for mode {model_name}.')
        data_pipeline = get_data_pipeline(config)
        X_train, y_train, features = data_pipeline.fit_transform(df_train)
        X_test, y_test, features = data_pipeline.transform(df_test)
        config['features'] = list(features)
        logger.info(f'X_train shape {X_train.shape}')
        logger.info(f'X_test shape {X_test.shape}')

        shape = X_train.shape[1: ]
        logger.info(f'Fitting model, {model_name}')

        init_params = config['init_params'].copy()
        init_params['input_shape'] = shape
        init_params['n_classes'] = len(categories)
        model, history = fit_keras(model_name, init_params, config['fit_params'],
                                   config['callback_params'],
                                   X_train, y_train, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test, y_test, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)

        save_vars(config, PROC_NAME, model_name, structure)
