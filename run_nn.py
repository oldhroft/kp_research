from datetime import datetime
import os
import json
import logging

from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.pipeline.data_pipe import LagDataPipe

from run_utils import fit_keras, save_history, read_data, NN_MODEL_DICT, score_keras
from run_utils import create_folder_structure, save_model_keras

PROC_NAME = 'nnrun'

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_path = 'vars/vars_nn.json' if arguments.vars is None else arguments.vars

    with open(vars_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
        
    df_train, df_test, categories = read_data()
    data_pipeline = LagDataPipe(config['variables'], 'category', 24, 24 // 3, scale=True,
                                shuffle=True, random_state=config['random_state'])
    
    X_train, y_train = data_pipeline.fit_transform(df_train)
    X_test, y_test = data_pipeline.transform(df_test)

    shape = (X_train.shape[1], )

    for model_name, _ in NN_MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Fitting model, {model_name}')
        model, history = fit_keras(model_name, shape, len(categories), 
                                   config['init_params'][model_name], 
                                   config['fit_params'][model_name],
                                   config['callback_params'][model_name],
                                   X_train, y_train, config['random_state'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test, y_test, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)

    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(structure['root'], 'vars_nn.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)
        