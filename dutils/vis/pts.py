
import matplotlib.pyplot as plt
from dutils.common import check_path
from .color import rgba
import os

osp = os.path 

class DrawPts(object):
    def __init__(self, pts, save=None):
        # pts: Nx3 array
        self.pts = pts
        self.save = save

    def bev(self):
        plot_bev(self.pts, savepath=self.save)
    
    def draw_bbox(self, bboxes):
        raise NotImplementedError

    def all(self):
        plot_all(self.pts, savepath=self.save)


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

def plot_all(points, savepath=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
    if savepath is not None:
        o3d.io.write_point_cloud(savepath, pcd)
    



