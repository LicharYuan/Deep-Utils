from dutils.common import wrap_args, wrap_log

@wrap_log(outfile="./test_wrap_log.log", name="debug")
def test():
    print("test wrap log")

if __name__ == "main":
    test()