from abc import ABCMeta, abstractmethod
import pprint, ast
import tabulate
from dutils.common import get_timestamp
import os.path as osp

class BaseExp(metaclass=ABCMeta):
    def __init__(self, **kwargs):

        self._build_independ_config()
        self.update(kwargs)
        self._build_depend_config()

        self.exp_name = self.__class__.__name__
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                # Type check
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        if src_type is bool:
                            v = eval(v)
                        else:
                            v = src_type(v)       
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
            else:
                # set new args
                setattr(self, k, v)

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if k in self._repr_keys
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def output(self, outdir, exp_name, time=True):
        if time:
            return osp.join(outdir, exp_name, get_timestamp())
        else:
            return osp.join(outdir, exp_name)



    @abstractmethod
    def _build_independ_config(self):
        # dont depend on other args
        pass

    @abstractmethod
    def _build_depend_config(self):
        pass
        
    @abstractmethod
    def get_model(self):
        return None

    @abstractmethod
    def get_optimizer(self):
        return None
