import logging
import sys

import datetime
import os

from .utils import create_folder

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

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

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

from argparse import ArgumentParser

def create_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--folder', action='store', type=str, required=True)
    parser.add_argument('--model', action='store', type=str, required=False, default=None)
    parser.add_argument('--dummy', action='store_true')
    return parser

