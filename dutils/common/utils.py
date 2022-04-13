import os
from datetime import date
from datetime import datetime

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,  exist_ok=True) # mp 

def get_today():
    return date.today().strftime("%Y%m%d")

def get_cur_time():
    return datetime.today().strftime("%Y%m%d%H%M")
