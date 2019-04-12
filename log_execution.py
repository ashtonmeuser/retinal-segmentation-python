"""
Logging for long-running functions
"""

import logging
from time import time

def log_execution(func):
    """
    Decorator to log execution time of wrapped function
    """
    def wrapped(*args, **kwargs):
        logging.debug('Executing %s...', func.__name__)
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.debug('Completed %s (%.3fs)', func.__name__, end - start)
        return result
    return wrapped
