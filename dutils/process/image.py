import cv2
import numpy as np

def undistort(img, intrinsic, distortion):
    dst = cv2.undistort(img,  intrinsic, distortion, None, intrinsic)
    return dst

def resize_img(img, imgh, imgw, dtype=np.float32):
    img = cv2.resize(img, (imgw, imgh), interpolation=cv2.INTER_LINEAR).astype(dtype)
    return img
