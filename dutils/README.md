
Modules:
- common: common usage functions by other modules, io
- vis: show or save img
- process: do some thing on img or bbox
- torch: deep-train tools for pytorch
- metrics: get correaltion, statics for metrics
- examples: some examples to use

# Visualize

- Make avi for directory

```python
cd examples
python gen_avi.py --dir ./data/imgs  --save ./results 
```


# Wrapper

**Wrap will set global args inside, It may make some problems.**

- Log

```python
from dutils.common import wrap_args, wrap_log

@wrap_log(outfile="./test_wrap_log.log" name="debug")
def test():
    print("test wrap log")


if __name__ == "__main__":
    test()
```

- Args

```python
@wrap_args("test", "a", "b", c=list,) 
def test_args():
    print(wargs.a) # default type:str, value: None
    print(wargs.b)
    print(type(wargs.c))
```





