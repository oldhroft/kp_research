import yaml
import logging

from scripts.helpers.logging_utils import config_logger, create_argparser

from run_utils import get_data_pipeline, save_model,  score, read_data, MODEL_DICT, fit
from run_utils import create_folder_structure, save_vars

PROC_NAME = 'skrun'

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_path = 'vars/vars.yaml' if arguments.vars is None else arguments.vars
    with open(vars_path, 'r', encoding='utf-8') as file:
        config_global = yaml.safe_load(file)

    df_train, df_test, categories = read_data()

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
        config['best_params'] = {}

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, config['init_params'], X_train, y_train)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, structure, PROC_NAME)
        if arguments.save_models:
            save_model(model, model_name, structure, PROC_NAME)
        
        save_vars(config, PROC_NAME, model_name, structure)
     






        




