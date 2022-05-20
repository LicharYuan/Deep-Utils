from dutils.common import wrap_args, wrap_log

@wrap_log(outfile="./test_wrap_log.log", name="debug")
def test_log():
    print("test wrap log")

@wrap_args("test", "a", "b", c=list,) 
def test_args():
    print(wargs.a) # default type:str, value: None
    print(wargs.b)
    print(type(wargs.c))

if __name__ == "main":
    test_args()