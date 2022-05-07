from datetime import datetime
import os
import json
import logging

from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.pipeline.data_pipe import LagDataPipe

from run_utils import fit, score, read_data, MODEL_DICT, save_model
from run_utils import create_folder_structure, grid_search

PROC_NAME = 'skruncv'

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_path = 'vars/vars_cv.json' if arguments.vars is None else arguments.vars
    with open(vars_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    df_train, df_test, categories = read_data()

    data_pipeline = LagDataPipe(config['variables'], 'category', 24, 24 // 3,)
    X_train, y_train = data_pipeline.fit_transform(df_train)
    y_train_cv = y_train.iloc[:, 0]

    X_test, y_test = data_pipeline.transform(df_test)
    print(type(X_test))

    config['best_params'] = {}
    for model_name, _ in MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Grid search model, {model_name}')
        best_params, best_score = grid_search(config['param_grids'][model_name],
                                              model_name,
                                              config['init_params'][model_name],
                                              X_train, y_train_cv, 
                                              config['cv_params'], config['gcv_params'])
        config['best_params'][model_name] = best_params
        params = config['init_params'][model_name].copy()
        params.update(best_params)
        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')


        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, params, X_train, y_train)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, structure, PROC_NAME)
        if arguments.save_models:
            save_model(model, model_name, structure, PROC_NAME)
    
    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(structure['root'], 'vars_cv.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)


            






        




