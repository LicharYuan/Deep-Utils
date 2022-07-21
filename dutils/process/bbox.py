# bbox 2d and 3d
import numpy as np
import warnings
def points_in_bboxes(bboxes, points):
    # bboxes: Mx8x3, points Nx3, 3:xyz
    """
                
                2 _______ 1
                /      /
              3/______/ 4

    """
    bbox1to4 = bboxes[:, :4, :2]
    pxy = points[:,:2] 
    pz = points[:, 2]
    out_mask = np.ones_like(pxy[:, 0])
    for i, bbox in enumerate(bbox1to4):
        p1, p2, p3, p4 = bbox
        # four angle
        # print(p1, p2, p3, p4)
        theta123 = _cal_angle_for_points(p1, p2, p3)
        theta234 = _cal_angle_for_points(p2, p3, p4)
        theta341 = _cal_angle_for_points(p3, p4, p1)
        theta412 = _cal_angle_for_points(p4, p1, p2)
        # print("res:", theta123, theta234, theta341, theta412)
        theta12x = _cal_angle_for_points(p1, p2, pxy)
        theta23x = _cal_angle_for_points(p2, p3, pxy)
        theta34x = _cal_angle_for_points(p3, p4, pxy)
        theta41x = _cal_angle_for_points(p4, p1, pxy)

        # outside mask, 
        # filter x-y 任何一个点夹角大于原来的角度则不在xy框内, 对每个框都判断
        theta_mask = np.logical_or(theta12x > theta123,  theta23x > theta234)
        theta_mask = np.logical_or(theta_mask,  theta34x > theta341)
        theta_mask = np.logical_or(theta_mask,  theta41x > theta412)
        # filter z
        z_min = np.min(bboxes[i, :, 2])
        z_max = np.max(bboxes[i, :, 2])
        z_mask = np.logical_or(pz > z_max, pz < z_min)
        bbox_out_mask = np.logical_or(z_mask, theta_mask)
        out_mask = np.logical_and(out_mask, bbox_out_mask)
    mask = np.logical_not(out_mask) 
    return mask

def cal_ious_2d(bboxes1, bboxes2, eps=1e-5):
    # iou between bboxes
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious

    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        union = area1[i] + area2 - overlap
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T

    return ious

def cal_ioss_2d(bboxes1, bboxes2, eps=1e-5):
    # ios between bboxes bboxes1: Nx4; bboxes2: Mx4
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ioss = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ioss

    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ioss = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    # less bbox shape
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        self_bbox = area1
        ioss[i, :] = overlap / self_bbox
    if exchange:
        ioss = ioss.T
    return ioss


def is_partial(bbox, h, w, ratio=0.02):
    x1, y1, x2, y2 = bbox[:4]
    partial_flag = x1 < w * ratio  or x2 > w * (1-ratio) or y1 < h * ratio or y2 > h * (1-ratio)
    return partial_flag

def bbox_size_xyxy(bbox):
    x1, y1, x2, y2 = bbox
    return (y2-y1) * (x2-x1)


def bbox_xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    assert x2>x1 and y2>y1
    w = x2 - x1
    h = y2 - y1
    center_x = (x1 + x2) / 2.
    center_y = (y1 + y2) / 2.
    return center_x, center_y, w, h

def bbox_xywh2xyxy(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w / 2.
    x2 = cx + w / 2.
    y1 = cy - h / 2.
    y2 = cy + h / 2.

    if x1 < 0 or y1 < 0:
        warnings.warn('function: bbox_xywh2xyxy \
            bbox x1y1 is out of range, will set as zero, please check x2y2')
        x1 = 0
        y1 = 0

    return x1, y1, x2, y2

def expand_bbox_xyxy(bbox, ratio):
    # wont check Out-of-Range
    cx, cy, w, h = bbox_xyxy2xywh(bbox)
    w *= ratio
    h *= ratio
    bbox = bbox_xywh2xyxy([cx, cy, w, h])
    return bbox

