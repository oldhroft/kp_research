from datetime import datetime
import os
import json
import logging
from sklearn.metrics import f1_score

from scripts.helpers.utils import validate_keras_cv
from scripts.helpers.logging_utils import config_logger, create_argparser

from run_utils import fit_keras, get_data_pipeline, save_history, score_keras, read_data, NN_MODEL_DICT
from run_utils import create_folder_structure, save_model_keras, get_data_pipeline

PROC_NAME = 'nnruncv'

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)
    
    vars_path = 'vars/vars_cv_nn.json' if arguments.vars is None else arguments.vars
    with open(vars_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    df_train, df_test, categories = read_data()

    config['best_params'] = {}
    for model_name, _ in NN_MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Data processing for {model_name}')
        data_pipeline = get_data_pipeline(config, model_name)
        X_train, y_train, features = data_pipeline.fit_transform(df_train)
        X_test, y_test, features = data_pipeline.transform(df_test)
        logger.info(f'X_train shape {X_train.shape}')
        logger.info(f'X_test shape {X_test.shape}')

        shape = X_train.shape[1: ]

        logger.info(f'Grid search model, {model_name}')
        best_params, best_score = validate_keras_cv(NN_MODEL_DICT[model_name], shape,  
                                                    len(categories), config['cv_params'],
                                                    config['param_grids'][model_name],
                                                    f1_score, X_train, y_train[:, 0],
                                                    config['callback_params'][model_name],
                                                    config['scoring_params'], 
                                                    config['init_params'][model_name],
                                                    config['fit_params'][model_name],
                                                    **config['gcv_params'])

        config['best_params'][model_name] = best_params
        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')

        logger.info(f'Fitting model, {model_name}')

        final_params = config['init_params'][model_name].copy()
        final_params.update(best_params)

        model, history = fit_keras(model_name, shape, len(categories), 
                                   final_params, config['fit_params'][model_name],
                                   config['callback_params'][model_name],
                                   X_train, y_train, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test, y_test, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)
    
    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(structure["root"], 'vars_cv_nn.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)


            






        




