from datetime import datetime
import os
import yaml
import logging

from scripts.helpers.logging_utils import config_logger, create_argparser

from run_utils import fit_keras, get_data_pipeline, save_history, read_data, NN_MODEL_DICT, score_keras
from run_utils import create_folder_structure, save_model_keras, save_vars

PROC_NAME = 'nnrun'

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_path = 'vars/vars_nn.yaml' if arguments.vars is None else arguments.vars
    with open(vars_path, 'r', encoding='utf-8') as file:
        config_global = yaml.safe_load(file)
        
    df_train, df_test, categories = read_data()

    for model_name, _ in NN_MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue
        config = config_global[model_name]
        config['best_params'] = {}

        logger.info(f'Data processing for mode {model_name}.')
        data_pipeline = get_data_pipeline(config)
        X_train, y_train, features = data_pipeline.fit_transform(df_train)
        X_test, y_test, features = data_pipeline.transform(df_test)
        logger.info(f'X_train shape {X_train.shape}')
        logger.info(f'X_test shape {X_test.shape}')

        shape = X_train.shape[1: ]
        logger.info(f'Fitting model, {model_name}')
        model, history = fit_keras(model_name, shape, len(categories), 
                                   config['init_params'], 
                                   config['fit_params'],
                                   config['callback_params'],
                                   X_train, y_train, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test, y_test, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)

        save_vars(config, PROC_NAME, model_name, structure)
