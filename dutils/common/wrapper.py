from dutils.common import MyLogger
import time
from functools import wraps, partial

def logg(outfile="./debug.log", name=""):
    """ the outputs log save in  "./xx.log" , name is prefix"""
    def _logg(func):
        logger = MyLogger(outfile, name=name)
        @wraps(func)
        def log_func(*args, **kwargs):
            tic = time.time()
            g = func.__globals__
            g["print"] = logger.info # replace print with logger.info
            g["logger"] = logger
            logger.info("Execute:", func.__name__)
            res = func(*args, **kwargs)
            toc = time.time()
            logger.info("Done , cost time:{0:2f}s".format(toc-tic))
            return res
        return log_func
    return _logg

