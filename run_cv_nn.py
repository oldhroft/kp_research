from datetime import datetime
import os
import json
import logging
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler

from helpers.utils import get_train_test, create_folder, validate_keras_cv
from helpers.logging_utils import config_logger, create_argparser

from run_utils import fit_keras, save_history, score_keras, read_data, NN_MODEL_DICT

PROC_NAME = 'nnruncv'

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

    with open('vars/vars_cv_nn.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    df, categories = read_data()

    df_train, lag_cols, df_test, lead_cols = get_train_test(df, config['variables'],
                                                            24 // 3, 24)
    X_train, y_train = df_train[lag_cols], df_train[lead_cols[0]]
    df_train = df_train.sample(frac=1., random_state=config['random_state'])
    y_train_full = df_train[lead_cols]
    X_test, y_test = df_test[lag_cols], df_test[lead_cols]

    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    shape = (X_train_scaled.shape[1], )

    config['best_params'] = {}
    for model_name, _ in NN_MODEL_DICT.items():
        if arguments.model is not None and arguments.model != model_name:
            continue
        if not arguments.dummy and model_name == 'dummy':
            continue

        logger.info(f'Grid search model, {model_name}')
        best_params, best_score = validate_keras_cv(NN_MODEL_DICT[model_name], shape,  
                                                    len(categories), config['cv_params'],
                                                    config['param_grids'][model_name],
                                                    f1_score, X_train_scaled, y_train,
                                                    config['callback_params'][model_name],
                                                    config['scoring_params'], 
                                                    config['init_params'][model_name],
                                                    config['fit_params'][model_name],
                                                    **config['gcv_params'])

        config['best_params'][model_name] = best_params
        logger.info(f'Best params: {best_params}')
        logger.info(f'Best score: {best_score}')

        logger.info(f'Fitting model, {model_name}')

        final_params = config['init_params'][model_name].copy()
        final_params.update(best_params)

        model, history = fit_keras(model_name, shape, len(categories), 
                                   final_params, config['fit_params'][model_name],
                                   config['callback_params'][model_name],
                                   X_train_scaled, y_train_full, config['seed'])

        logger.info(f'Scoring model, {model_name}')
        score_keras(model, model_name, X_test_scaled, y_test, root, matrix_path, PROC_NAME)
        save_history(history, model_name, history_path, PROC_NAME)
    
    config['dttm'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    with open(os.path.join(root, 'vars_cv_nn.json'), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)


            






        




