import logging
import os

from scripts.helpers.logging_utils import config_logger, create_argparser

from run_utils import get_data_pipeline, save_model,  score, read_data, MODEL_DICT, fit
from run_utils import create_folder_structure, save_vars
from scripts.helpers.yaml_utils import load_yaml, dict_to_yaml_str

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

    logger.info(f'Data processing...')
    data_pipeline = get_data_pipeline(config_global["default"])
    
    X_train, y_train, features = data_pipeline.fit_transform(df_train)
    X_test, y_test, features = data_pipeline.transform(df_test)
    
    logger.info(f'X_train shape {X_train.shape}')
    logger.info(f'X_test shape {X_test.shape}')

    for model_name, _ in MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue
        config = config_global[model_name]
        logger.info(f'Model {model_name}, params:')
        logger.info(dict_to_yaml_str(config))
        config['best_params'] = {}

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, config['init_params'], X_train, y_train)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, structure, PROC_NAME)
        if arguments.save_models:
            save_model(model, model_name, structure, PROC_NAME)
        
        save_vars(config, PROC_NAME, model_name, structure)
        