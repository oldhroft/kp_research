import logging
import sys

def config_logger(logger):
    handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)