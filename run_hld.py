from datetime import datetime
import os
import sys
import json
import logging

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils import get_train_test, create_folder, validate
from logging_utils import config_logger, create_argparser
from run_utils import fit, score, read_data, MODEL_DICT

logger = logging.getLogger(__name__)
config_logger(logger)

PROC_NAME = 'skrunhld'

def create_folder_structure(root):
    matrix_path = os.path.join(root, 'matrix')
    create_folder(root)
    create_folder(matrix_path)
    return root, matrix_path


if __name__ == '__main__':

    arguments = create_argparser().parse_args()

    with open('vars_hld.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    root, matrix_path =  create_folder_structure(arguments.folder)
    config['best_params'] = {}

    df, categories = read_data()

    df_train, lag_cols, df_val, _, _, lead_cols = get_train_test(df, config['variables'], 
                                                                 24 // 3, 24, last_val='24m')
    X_train, y_train = df_train[lag_cols], df_train[lead_cols[0]]
    X_val, y_val = df_val[lag_cols], df_val[lead_cols[0]]

    df_train, lag_cols, df_test, lead_cols = get_train_test(df, config['variables'], 
                                                                  24 // 3, 24,)
    X_train_full, y_train_full = df_test[lag_cols], df_test[lead_cols]
    X_test, y_test = df_test[lag_cols], df_test[lead_cols]

    for model_name, model in MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Grid search model, {model_name}')
        model = model.set_params(**config['init_params'][model_name])
        best_score, best_params = validate(model, config['param_grids'][model_name],
                                           X_train, y_train,
                                           X_val, y_val, **config['gv_params'])

        logger.info(f'Best params: {best_params}')
        config['best_params'][model_name] = best_params
        params = config['init_params'][model_name].copy()
        params.update(best_params)

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, params, X_train_full, y_train_full)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, root, matrix_path, PROC_NAME)

    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(root, 'vars_hld.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file)


            






        




