
import matplotlib.pyplot as plt
from TUtils.utils import check_path
from .color import rgba
import os
os.path = osp

class DrawPts(objest):
    def __init__(self, pts, save=None):
        self.pts = pts
        self.save = save

    def bev(self):
        plot_bev(self.pts, savepath=self.save)
    

def plot_bev(points, xid=0, yid=1, color="r", savepath=None):
    plt.figure(figsize=(100, 100))
    plt.scatter(points[:,xid], points[:,yid], s=0.5, c=color)  
    if savepath is not None:
        check_path(savepath)
        if not savepath.endswith(".jpg") or savepath.endswith(".png"):
            savepath = osp.join(savepath, "points.jpg")
        plt.savefig(savepath)

def plot_bev_list(points_list, xid=0, yid=1, color=rgba, savepath=None):
    plt.figure(figsize=(100, 100))
    for i, points in enumerate(points_list):
        plt.scatter(points[:,xid], points[:,yid], s=0.5, c=color(i)) 
    if savepath is not None:
        check_path(savepath)
        if not savepath.endswith(".jpg") or savepath.endswith(".png"):
            savepath = osp.join(savepath, "points.jpg")
        plt.savefig(savepath)

