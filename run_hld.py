import os
import logging

from pandas import concat

from scripts.helpers.utils import validate
from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.helpers.yaml_utils import load_yaml

from run_utils import fit, get_data_pipeline, score, read_data, MODEL_DICT, save_model
from run_utils import create_folder_structure, save_vars

PROC_NAME = os.path.basename(__file__).split('.')[0]

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_name = f'vars_{PROC_NAME}.yaml'
    vars_path = os.path.join('vars', vars_name) if arguments.vars is None else arguments.vars
    config_global = load_yaml(vars_path)

    df_train, df_test, df_val, categories = read_data(val=True)

    logger.info(f'Data processing...')
    data_pipeline =get_data_pipeline(config_global["default"])
    X_train, y_train, features = data_pipeline.fit_transform(df_train)
    X_test, y_test, features = data_pipeline.transform(df_test)
    X_val, y_val, features = data_pipeline.transform(df_val)

    df_train_full = concat([df_train, df_val], ignore_index=True)
    X_train_full, y_train_full, features = data_pipeline.fit_transform(df_train_full)
    X_test_full, y_test_full, features = data_pipeline.transform(df_test)

    logger.info(f'X_train shape {X_train.shape}')
    logger.info(f'X_val shape {X_val.shape}')
    logger.info(f'X_test shape {X_test.shape}')
    logger.info(f'X_train_full shape {X_train_full.shape}')

    for model_name, model in MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue
        config = config_global[model_name]
        config['best_params'] = {}

        logger.info(f'Grid search model, {model_name}')
        model = model.set_params(**config['init_params'])
        best_score, best_params = validate(model, config['param_grids'],
                                           X_train, y_train[:, 0],
                                           X_val, y_val[:, 0], **config['gv_params'])

        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')
        config['best_params'] = best_params
        params = config['init_params'].copy()
        params.update(best_params)

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, params, X_train_full, y_train_full)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test_full, y_test_full, structure, PROC_NAME)

        if arguments.save_models:
            save_model(model, model_name, structure, PROC_NAME)

        save_vars(config, PROC_NAME, model_name, structure)



            






        




