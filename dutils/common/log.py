from .utils import get_today, check_path
from tabulate import tabulate
from termcolor import colored
import logging
import sys, os
import time
import pprint

LOGGED = {}
class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink", "bold"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline", "bold"])
        else:
            return log
        return prefix + " " + log

def setup_logger(output=None, rank=0, color=True, name=" ", save_all_rank=False):
    logger = logging.getLogger(name)
    if name in LOGGED:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: rank = 0  only
    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    # file logging: all ranks
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = name + '_' + get_today() + '.log'
            filename = os.path.join(output, filename)

        filename = filename if rank==0 else filename + ".rank{}".format(rank)
        if not save_all_rank and rank > 0:
            logger.setLevel(logging.ERROR)
        else:
            check_path(os.path.dirname(filename))
            fh = logging.FileHandler(filename, 'w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

    LOGGED[name] = True
    return logger

class MyLogger(object):
    def __init__(self, outfile=None, color=True, name='debug', saveall=True, rank=0):
        self.logger = setup_logger(outfile, rank, color, name, saveall)
        self.rank = rank
        self.name = name

    def error(self,info):
        self.logger.setLevel(logging.ERROR)
        self.logger.error(info)
    
    def warn(self, *info):
        self.logger.setLevel(logging.WARNING)
        info = " ".join(str(ele) for ele in info)
        self.logger.warning(info)

    def info(self, *info):
        self.logger.setLevel(logging.DEBUG)
        info = " ".join(str(ele) for ele in info)
        self.logger.info(info)

    def table(self, info):
        if isinstance(info, dict):
            table_header = ["keys", "values"]
            dict_table_list =  [
                (str(k), pprint.pformat(v))
                for k, v in info.items()
            ]
            dict_table = tabulate(dict_table_list, headers=table_header, tablefmt="fancy_grid")
            self.logger.info("\n"+dict_table)
        else:
            self.logger.info("tableprint only suppory dict type, print original format:")
            self.logger.info(info)

    def __call__(self, info, *args):
        if isinstance(info, dict):
            self.table(info)
        else:
            new_info = [info]
            new_info.extend(args)
            self.info(*new_info)
