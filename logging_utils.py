import logging
import sys

import datetime
import os

from utils import create_folder

def config_logger(logger, proc_name: str, folder: str):

    handler = logging.StreamHandler(sys.stdout)

    log_folder = os.path.join(folder, f'log_{proc_name}')
    create_folder(log_folder)
    now = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    filename = os.path.join(log_folder, f'log_{now}.log')
    file_handler = logging.FileHandler(filename, mode='w', encoding='utf-8')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

from argparse import ArgumentParser

def create_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--folder', action='store', type=str, required=True)
    parser.add_argument('--model', action='store', type=str, required=False, default=None)
    parser.add_argument('--dummy', action='store_true')
    return parser

