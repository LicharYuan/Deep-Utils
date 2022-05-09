import tqdm

def niter_bar(niter):
    bar = tqdm(total=niter)
    return bar

