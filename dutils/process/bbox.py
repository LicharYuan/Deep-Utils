# bbox 2d and 3d
import numpy as np

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