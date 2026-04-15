import logging
from loguru import logger
import hashlib

import colorlog

def setup_logger(logger_path: str, verbose: bool = False, mode: str = "w") -> logging.Logger:
    path_hash = hashlib.md5(logger_path.encode('utf-8')).hexdigest()
    logger_name = f"task_logger_{path_hash}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(logger_path, mode=mode, encoding='utf-8', delay=False)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

    if verbose:
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


def get_logger(*args, **kwargs):
    return logger
