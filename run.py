from datetime import datetime
import os
import sys
import json
import logging

from helpers.utils import get_train_test, create_folder
from helpers.logging_utils import config_logger, create_argparser

from run_utils import score, read_data, MODEL_DICT, fit

PROC_NAME = 'skrun'

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

    with open('vars/vars.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    df, categories = read_data()
    
    df_train, lag_cols, df_test, lead_cols = get_train_test(df, config['variables'], 24 // 3, 24)
    X_train, y_train = df_train[lag_cols], df_train[lead_cols]
    X_test, y_test = df_test[lag_cols], df_test[lead_cols]

        
    model_name_param = sys.argv[2] if len(sys.argv) > 2 else None

    for model_name, _ in MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue        

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, config['init_params'][model_name], X_train, y_train)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, root, matrix_path, PROC_NAME)

    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(root, 'vars.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)
            






        




