from datetime import datetime
import os
import json
import logging

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from scripts.helpers.utils import get_train_test, create_folder
from scripts.helpers.logging_utils import config_logger, create_argparser
from run_utils import fit, score, read_data, MODEL_DICT

PROC_NAME = 'skruncv'

def create_folder_structure(root):
    matrix_path = os.path.join(root, 'matrix')
    create_folder(root)
    create_folder(matrix_path)

    return root, matrix_path

def grid_search(params, model_name, init_params, X_train, y_train, 
                cv_params, gcv_params):
    model = MODEL_DICT[model_name].set_params(**init_params)
    skf = StratifiedKFold(**cv_params)
    gcv = GridSearchCV(model, params, cv=skf, **gcv_params)
    gcv.fit(X_train, y_train)
    return gcv.best_params_, gcv.best_score_


if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    root, matrix_path = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    with open('vars/vars_cv.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    df, categories = read_data()

    df_train, lag_cols, df_test, lead_cols = get_train_test(df, config['variables'],
                                                            24 // 3, 24)
    X_train, y_train = df_train[lag_cols], df_train[lead_cols[0]]
    y_train_full = df_train[lead_cols]
    X_test, y_test = df_test[lag_cols], df_test[lead_cols]

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
                                              X_train, y_train, config['cv_params'],
                                              config['gcv_params'])
        config['best_params'][model_name] = best_params
        params = config['init_params'][model_name].copy()
        params.update(best_params)
        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')


        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, params, X_train, y_train_full)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, root, matrix_path, PROC_NAME)
    
    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(root, 'vars_cv.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)


            






        




