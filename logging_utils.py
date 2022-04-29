import logging
import sys

from numpy import imag

def config_logger(logger):
    handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

from argparse import ArgumentParser
def create_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--folder', action='store', type=str, required=True)
    parser.add_argument('--model', action='store', type=str, required=False, default=None)
    parser.add_argument('--dummy', action='store_true')
    return parser

