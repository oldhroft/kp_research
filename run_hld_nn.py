from datetime import datetime
import os
import json
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from helpers.utils import get_train_test, create_folder, validate_keras
from helpers.logging_utils import config_logger, create_argparser

from run_utils import fit_keras, save_history, score_keras, read_data, NN_MODEL_DICT

PROC_NAME = 'nnrunhld'

def create_folder_structure(root):
    matrix_path = os.path.join(root, 'matrix')
    history_parh = os.path.join(root, 'history')
    create_folder(root)
    create_folder(matrix_path)
    create_folder(history_parh)
    return root, matrix_path, history_parh

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    root, matrix_path, history_path = create_folder_structure(arguments.folder)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    with open('vars/vars_hld_nn.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    config['best_params'] = {}

    df, categories = read_data()

    df_train, lag_cols, df_val, _, _, lead_cols = get_train_test(df, config['variables'], 
                                                                 24 // 3, 24, last_val='24m')

    df_train = df_train.sample(frac=1., random_state=config['random_state'])
    X_train, y_train = df_train[lag_cols], df_train[lead_cols[0]]
    X_val, y_val = df_val[lag_cols], df_val[lead_cols[0]]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    df_train, lag_cols, df_test, lead_cols = get_train_test(df, config['variables'], 
                                                                  24 // 3, 24,)
    df_train = df_train.sample(frac=1., random_state=config['random_state'])                                                           
    X_train_full, y_train_full = df_train[lag_cols], df_train[lead_cols]
    X_test, y_test = df_test[lag_cols], df_test[lead_cols]

    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)

    shape = (X_train.shape[1], )

    for model_name, model in NN_MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Grid search model, {model_name}')
        
        best_params, best_score = validate_keras(NN_MODEL_DICT[model_name], shape,  
                                                 len(categories), config['param_grids'][model_name],
                                                 f1_score, X_train, y_train,
                                                 X_val, y_val,
                                                 config['callback_params'][model_name],
                                                 config['scoring_params'],
                                                 config['init_params'][model_name],
                                                 config['fit_params'][model_name],
                                                 **config['gv_params'])

        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')
        config['best_params'][model_name] = best_params
        final_params = config['init_params'][model_name].copy()
        final_params.update(best_params)

        logger.info(f'Fitting model, {model_name}')
        model, history = fit_keras(model_name, shape, len(categories), 
                                   final_params, config['fit_params'][model_name],
                                   config['callback_params'][model_name],
                                   X_train_full, y_train_full, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test, y_test, root, matrix_path, PROC_NAME)
        save_history(history, model_name, history_path, PROC_NAME)

    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(root, 'vars_hld_nn.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)


            






        




