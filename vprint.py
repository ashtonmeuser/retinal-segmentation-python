"""
Verbose printing
"""

def vprint(verbose, message, flush=False, end='\n'):
    """
    Print if verbose
    """
    if verbose:
        print(message, flush=flush, end=end)
