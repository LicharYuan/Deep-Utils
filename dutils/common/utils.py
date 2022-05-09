import os
from datetime import date
from datetime import datetime
import argparse
import glob

__all__ = ['check_path', 'get_today', 'quick_str_args', 'glob_file_in_dir', 'get_timestamp']

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
        parser.add_argument(f"--{key}", default=None, type=value)
    args = parser.parse_args()
    return args

def glob_file_in_dir(dir, must_key):
    files = glob.glob(os.path.join(dir, f"*{must_key}*"))
    return files

def get_timestamp():
    # till seconds
    timestamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    return timestamp
