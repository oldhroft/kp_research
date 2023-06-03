import logging
import os

from run_utils import (create_folder_structure, fit, get_data_pipeline,
                       read_data, save_model, save_vars, score, check_config,
                       build_data_pipelines)
from scripts.helpers.logging_utils import config_logger, create_argparser
from scripts.helpers.utils import add_to_environ
from scripts.helpers.yaml_utils import dict_to_yaml_str, load_yaml

from scripts.models import sk_model_factory

PROC_NAME = os.path.basename(__file__).split('.')[0]

if __name__ == '__main__':

    arguments = create_argparser().parse_args()
    structure = create_folder_structure(arguments.folder)
    add_to_environ(arguments.conf)

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, arguments.folder)

    vars_name = f'vars_{PROC_NAME}.yaml'
    vars_path = os.path.join('vars', vars_name) if arguments.vars is None else arguments.vars
    config_global = load_yaml(vars_path)
    check_config(config_global, sk_model_factory)


    logger.info(f'Data processing...')

    data_pipelines = build_data_pipelines(
        config_global["models"], structure
    )
    
    for model_name, config in config_global['models'].items():

        if arguments.model is not None and arguments.model != model_name:
            continue
        
        pipe = data_pipelines[config["pipe_name"]]
        X_train, y_train, X_test, y_test = pipe.get_xy()
       

        logger.info('\n' + '=' * 60 + '\n')
        logger.info(f'Model {model_name}, params:')
        logger.info(dict_to_yaml_str(config))
        logger.info(f'X_train shape {X_train.shape}')
        logger.info(f'X_test shape {X_test.shape}')
        config['features'] = list(pipe.features)
        config['best_params'] = {}

        logger.info(f'Fitting model, {model_name}')
        model = fit(model_name, config['init_params'], X_train, y_train)

        logger.info(f'Scoring model, {model_name}')
        score(model, model_name, X_test, y_test, structure, PROC_NAME)
        if arguments.save_models:
            save_model(model, model_name, structure, PROC_NAME)
        
        save_vars(config, PROC_NAME, model_name, structure)
        