"""
Verbose printing
"""

from time import time

def vprint(verbose, message, flush=False, end='\n'):
    """
    Print if verbose
    """
    if verbose:
        print(message, flush=flush, end=end)

def time_it(method):
    """
    Print time taken to execute wrapped function
    """
    def wrapped(*args, **kw):
        start = time()
        result = method(*args, **kw)
        end = time()
        print('{}  {:.2f}s'.format(method.__name__, end - start))
        return result
    return wrapped
