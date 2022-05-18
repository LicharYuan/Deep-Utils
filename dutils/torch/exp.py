from abc import ABCMeta, abstractmethod
import pprint, ast
from tabnanny import check
from tabulate import tabulate
from dutils.common import get_timestamp
import os.path as osp
from dutils.common.utils import check_path
import numpy as np
import yaml
from dutils.common import save_json, save_yaml, save_pkl

class BaseExp(metaclass=ABCMeta):
    def __init__(self, **kwargs):

        self._build_independ_config()
        self.update(**kwargs)
        self._build_depend_config()

        self.exp_name = self.__class__.__name__
        self.outdir = "./results"
        self._repr_keys = []
        
    def update(self, **kwargs):
        # only support update for int/float/bool
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
            if k in sorted(self._repr_keys)
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def output(self, time=True):
        if time:
            return osp.join(self.outdir, self.exp_name, get_timestamp())
        else:
            return osp.join(self.outdir, self.exp_name)

    def dump(self, save=None):
        # save config if type is int/float/str/bool/array
        exp_config = {}
        for k, v in vars(self).items():
            if isinstance(v, (int, float, bool, str, np.ndarray, list, tuple, dict)):
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                    
                exp_config[k] = v
        if save:
            dump_file = save
        else:
            check_path(osp.join(self.outdir, self.exp_name))
            dump_file = osp.join(self.outdir, self.exp_name, "exp.yaml")
            
        save_yaml(dump_file, exp_config)

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
