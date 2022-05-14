import os
import logging
from sklearn.metrics import f1_score

from scripts.helpers.utils import validate_keras_cv
from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.helpers.yaml_utils import load_yaml, dict_to_yaml_str

from run_utils import fit_keras, get_data_pipeline, save_history, score_keras, read_data, NN_MODEL_DICT
from run_utils import create_folder_structure, get_data_pipeline
from run_utils import save_vars, save_cv_results, save_model_keras

PROC_NAME = os.path.basename(__file__).split('.')[0]

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_name = f'vars_{PROC_NAME}.yaml'
    vars_path = os.path.join('vars', vars_name) if arguments.vars is None else arguments.vars
    config_global = load_yaml(vars_path)

    df_train, df_test, categories = read_data(arguments.data)
    
    for model_name, _ in NN_MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue
        config = config_global[model_name]
        logger.info(f'Model {model_name}, params:')
        logger.info(dict_to_yaml_str(config))
        config['best_params'] = {}

        logger.info(f'Data processing for {model_name}')
        data_pipeline = get_data_pipeline(config)
        X_train, y_train, features = data_pipeline.fit_transform(df_train)
        X_test, y_test, features = data_pipeline.transform(df_test)
        logger.info(f'X_train shape {X_train.shape}')
        logger.info(f'X_test shape {X_test.shape}')

        shape = X_train.shape[1: ]

        logger.info(f'Grid search model, {model_name}')
        best_params, best_score, results = validate_keras_cv(NN_MODEL_DICT[model_name], shape,  
                                                             len(categories), config,
                                                             config['param_grids'],
                                                             f1_score, X_train, y_train[:, 0],
                                                             config['callback_params'],
                                                             config['scoring_params'], 
                                                             config['init_params'],
                                                             config['fit_params'],
                                                             **config['gcv_params'])
        save_cv_results(results, model_name, structure, PROC_NAME)
        config['best_params'] = best_params
        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')

        logger.info(f'Fitting model, {model_name}')
        final_params = config['init_params'].copy()
        final_params.update(best_params)
        model, history = fit_keras(model_name, shape, len(categories), 
                                   final_params, config['fit_params'],
                                   config['callback_params'],
                                   X_train, y_train, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test, y_test, structure, PROC_NAME)
        save_history(history, model_name, structure, PROC_NAME)
        if arguments.save_models:
            save_model_keras(model, model_name, structure, PROC_NAME)
    
        save_vars(config, PROC_NAME, model_name, structure)
