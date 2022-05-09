from dutils.visualize import GenAVI
from dutils.common import wrap_args, glob_file_in_dir, check_path
import os

@wrap_args("gen avi", "dir", "save")
def gen_avi():
    img_list = glob_file_in_dir(wargs.dir, must_key="")
    print(len(img_list))
    check_path(wargs.save)
    save_avi = os.path.join(wargs.save, "test.avi")
    GenAVIFunc= GenAVI(img_list, save_avi) 
    GenAVIFunc.make()


if __name__ == "__main__":
    gen_avi()

