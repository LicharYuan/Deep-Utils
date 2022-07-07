import multiprocessing

def get_rank():
    cur_process = multiprocessing.current_process()
    cur_rank = eval(str(cur_process).split(',')[0].split('-')[-1])
    return cur_rank
    