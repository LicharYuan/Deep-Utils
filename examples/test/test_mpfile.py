from dutils.multiprocess import MPFile, get_rank
import multiprocessing

def f(x, io_file):
    data = [x * x]
    io_file.write_json(data)

if __name__ == '__main__':
    p = multiprocessing.Pool(6)
    io_file = MPFile(6, 'test') # copy class
    p.starmap(f, zip(range(6), [io_file]*6))
    data = io_file.read_json()
    print(data)
    





