"""
MPFile 的作用:
在开发过程中, 常遇到的需求是, 多进程写入不同的数据, 然后同步给主进程.
每次自己都需要写对应的代码, 借助MPFile, **初始化一次** 自动的在不同的进程完成写入/读取.
"""
import uuid
from dutils.common.io import load_json
from dutils.common.utils import glob_file_in_dir
from dutils.multiprocess import get_rank
from dutils.common import gen_random_file, save_json, merge_json_results

class MPFile(object):
    def __init__(self, pool_size, save_name):
        self.pool_size = pool_size
        self.save_name = save_name
        self._tmp_dir = f"./.mpfiles_{str(uuid.uuid4())}"

    def get_rank(self):
        return get_rank()

    def write_json(self, data):
        _rank_file = gen_random_file(self._tmp_dir, prefix=self.save_name, suffix=".json")
        save_json(_rank_file, data)

    def read_json(self):
        # search all files contain save_name 
        all_jsons = glob_file_in_dir(self._tmp_dir, must_key=self.save_name)
        all_jsons = [ele for ele in all_jsons if ele.endswith(".json")]
        all_data = merge_json_results(all_jsons)
        return all_data
    
            