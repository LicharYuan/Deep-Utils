from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import os
osp = os.path

def scatter(a, b, xlabel, ylabel, title=None, save=None, size=None):
    plt.figure()
    plt.scatter(a, b, s=size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save)

def scatter3(a, b, c, xlabel, ylabel, title=None, save=None, size=None):
    assert len(ylabel) == 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(a, b, color="g", label=ylabel[0], s=size)
    ax2 = ax.twinx()
    ax2.scatter(a, c,  marker="+", color="b", label=ylabel[1], s=size)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel[0])
    ax2.set_ylabel(ylabel[1])
    
    fig.legend()
    if title is not None:
        fig.title(title)
    if save is not None:
        fig.savefig(save)