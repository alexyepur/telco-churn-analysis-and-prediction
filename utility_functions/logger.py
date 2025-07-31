import logging
import os
from datetime import datetime
import colorlog

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/churn_log_{datetime.now().strftime('%Y-%m-%d')}.log"


def get_logger(
    name: str
):
    """
    Creates and configures a logger with file and colored console handlers.
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """    
    logger = logging.getLogger(name)  # defines logger for source

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)  # set a minimum log level to INFO

        # === File Log Handler ===
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )  # sets a format of log records
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # === Color Console Handler ===
        stream_handler = colorlog.StreamHandler()  # sets a color consol logs
        stream_formatter = colorlog.ColoredFormatter(
            "%(log_color)s>> %(levelname)-8s: %(message)s",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger