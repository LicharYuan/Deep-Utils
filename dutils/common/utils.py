import os
from datetime import date
from datetime import datetime
import argparse
import glob
import uuid
import numpy as np

__all__ = ['check_path', 'get_today', 'quick_str_args', 'glob_file_in_dir', 'get_timestamp', 'logical_and', 
           'gen_random_file',
        ]

def gen_random_file(root, suffix=".json", prefix=""):
    check_path(root)
    _random_name = str(uuid.uuid4())
    _random_name = prefix + _random_name + suffix
    path = os.path.join(root, _random_name)
    return path
    
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,  exist_ok=True) # mp 

def get_today():
    return date.today().strftime("%Y%m%d")

def quick_str_args(name="quick args", *args, **kwargs):
    parser = argparse.ArgumentParser(name)
    for ele in args:
        parser.add_argument(f"--{ele}", default=None, type=str)
    for key, value in kwargs.items():
        allow_types = [str, int, float]
        assert value in allow_types, "quick args only support {}".format(allow_types)
        parser.add_argument(f"--{key}", default=None, type=value)
    args = parser.parse_args()
    return args

def glob_file_in_dir(dir, must_key):
    files = glob.glob(os.path.join(dir, f"*{must_key}*"))
    return files

def get_timestamp():
    # till seconds
    timestamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    return timestamp


def logical_and(a, b, *args):
    mask_and = np.logical_and(a, b)
    for ele in args:
        mask_and = np.logical_and(mask_and, ele)
    return mask_and

def is_increase_arr(array):
    first = array[0]
    for ele in array[1:]:
        flag = ele > first
        first = ele
        if flag is False:
            return False
    return True

    
