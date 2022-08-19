import logging
import glob
import os
import shutil

from ..helpers.logging_utils import config_logger


def test_config_logger():

    PROC_NAME = "test_proc"
    folder = "scripts/tests"

    logger = logging.getLogger(__name__)
    config_logger(logger, PROC_NAME, folder)
    log_folder = f"scripts/tests/log_{PROC_NAME}"

    logger.info("sample info")

    filename = glob.glob(os.path.join(log_folder, "*"))[0]

    with open(filename, "r", encoding="utf-8") as file:
        data = file.read()

    shutil.rmtree(log_folder)

    assert "sample info" in data, "Log not appeared in log file"
