from .log import MyLogger
from .utils import quick_str_args
import time
from functools import wraps, partial

__all__ = ["wrap_log", "wrap_args"]

# quick record log
def wrap_log(outfile="./debug.log", name="", verbose=True):
    """ the outputs log save in  "./xx.log" , name is prefix"""
    def _logg(func):
        logger = MyLogger(outfile, name=name)
        @wraps(func)
        def log_func(*args, **kwargs):
            tic = time.time()
            g = func.__globals__
            g["print"] = logger.info # replace print with logger.info
            g["logger"] = logger
            if verbose:
                logger.info("Execute:", func.__name__)
            res = func(*args, **kwargs)
            toc = time.time()
            if verbose:
                logger.info("Done , cost time:{0:2f}s".format(toc-tic))
            return res
        return log_func
    return _logg

# quick method to build args from cmd 
def wrap_args(name, *args, **kwargs):
    def _args(func):
        fargs = quick_str_args(name, *args, **kwargs)
        @wraps(func)
        def _func(*args, **kwargs):
            g = func.__globals__
            g["wargs"] = fargs
            res = func(*args, **kwargs)
            return res
        return _func
    return _args
