import logging
import os

mylogger = None

def setup_logger(name, log_file):
    """
    Sets up a logger with the specified name, log file, and logging level.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Example usage:
# logger = setup_logger('example_logger', 'example.log')
# logger.info('This is an info message.')