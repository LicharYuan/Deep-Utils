import multiprocessing
import os
def f(x):
    print (get_rank())
    print (os.getpid())
    return x * x

def get_rank():
    cur_process = multiprocessing.current_process()
    cur_rank = eval(str(cur_process).split(',')[0].split('-')[-1])
    return cur_rank

if __name__ == '__main__':
    p = multiprocessing.Pool()
    # p.apply_async(f)
    print(p.map(f, range(6)))