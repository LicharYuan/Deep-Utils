
# IO
import os
import json
import pickle 
import pyarrow as pa
import pickle as pkl
import numpy as np
import yaml
import uuid
__all__ = ["load_json", "save_pkl", "save_json", "load_pkl", "save_yaml", "load_yaml",
          "assert_path", "read_roidb", "read_npz", "read_npy", "merge_json_results"]


def load_json(json_file):
    assert os.path.exists(json_file)
    with open(json_file, "r") as f:
        res = json.load(f)
    return res

def save_yaml(filename, data):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def load_yaml(filename):
    with open(filename, "r") as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

def assert_path(path):
    if isinstance(path, (list, tuple)):
        for ele in path:
            assert os.path.exists(ele), ele + " is not exists"    
    else:
        assert os.path.exists(path), path + " is not exists"

def save_pkl(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)

def load_pkl(filename, check_error=True):
    if check_error:
        try:
            return  _load_pkl(filename)
        except FileNotFoundError:
            print("File not exits, check path")
            return None
    else:
        return _load_pkl(filename)

def _load_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def read_pa(path):
    with open(path, "rb") as f:
        data = pa.deserialize(f.read())
    return data

def read_roidb(path):
    with open(path, "rb") as f:
        data = pkl.load(f, encoding="latin1")
    return data

def read_npz(path):
    return np.load(path, allow_pickle=True)

def read_npy(path):
    return np.load(path)

def update_to_json(save_path, update_data):
    data = load_json(save_path)
    assert isinstance(data, dict)
    data.update(update_data)
    save_json(save_path, data)


def merge_json_results(json_files):
    data_type = type(load_json(json_files[0]))
    if data_type in (list, tuple):
        all_data = []
        _type_flag = 0
    elif data_type is dict:
        all_data = {}
        _type_flag = 1
    else:
        raise TypeError(data_type)
    
    for js_file in json_files:
        data = load_json(js_file)

        if _type_flag == 0:
            if not isinstance(data, (list, tuple)):
                print(js_file, 'data type error')
                continue
            all_data.extend(data)
        
        if _type_flag == 1:
            if not isinstance(data, dict):
                print(js_file, 'data type error')
            all_data.update(data)

    return all_data

