from datetime import datetime
import os
import json
import logging

from pandas import concat

from scripts.helpers.utils import create_folder, validate
from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.pipeline.data_pipe import LagDataPipe

from run_utils import fit, score, read_data, MODEL_DICT

PROC_NAME = 'skrunhld'

def create_folder_structure(root):
    matrix_path = os.path.join(root, 'matrix')
    create_folder(root)
    create_folder(matrix_path)
    return root, matrix_path


if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    root, matrix_path = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    with open('vars/vars_hld.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    config['best_params'] = {}

    df_train, df_test, df_val, categories = read_data(val=True)

    data_pipeline = LagDataPipe(config['variables'], 'category', 24, 24 // 3,)
    X_train, y_train = data_pipeline.fit_transform(df_train)
    X_test, y_test = data_pipeline.transform(df_test)
    X_val, y_val = data_pipeline.transform(df_val)

    df_train_full = concat([df_train, df_val], ignore_index=True)
    X_train_full, y_train_full = data_pipeline.fit_transform(df_train_full)
    X_test_full, y_test_full = data_pipeline.transform(df_test)

    for model_name, model in MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Grid search model, {model_name}')
        model = model.set_params(**config['init_params'][model_name])
        best_score, best_params = validate(model, config['param_grids'][model_name],
                                           X_train, y_train.iloc[:, 0],
                                           X_val, y_val.iloc[:, 0], **config['gv_params'])

        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')
        config['best_params'][model_name] = best_params
        params = config['init_params'][model_name].copy()
        params.update(best_params)

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, params, X_train_full, y_train_full)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test_full, y_test_full, root, matrix_path, PROC_NAME)

    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(root, 'vars_hld.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)


            






        




