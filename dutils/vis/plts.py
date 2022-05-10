import matplotlib.pyplot as plt
import os
osp = os.path

def scatter(a, b, xlabel, ylabel, title=None, save=None):
    plt.figure()
    plt.scatter(a, b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save)