import logging
import os

_logger = None

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name, log file, and logging level.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    _logger = logger
    return logger


def get_logger():
    if _logger is None:
        raise ValueError("Logger has not been set up. Please call setup_logger first.")
    return _logger

# Example usage:
# logger = setup_logger('example_logger', 'example.log')
# logger.info('This is an info message.')