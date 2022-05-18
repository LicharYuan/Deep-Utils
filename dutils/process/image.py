import cv2
import numpy as np

def undistort(img, intrinsic, distortion):
    dst = cv2.undistort(img,  intrinsic, distortion, None, intrinsic)
    return dst

def resize_img(img, imgh, imgw, dtype=np.float32):
    img = cv2.resize(img, (imgw, imgh), interpolation=cv2.INTER_LINEAR).astype(dtype)
    return img

def normalize(inputs, mean, std):
    # inputs: C, H, W,  normalize to [-1, 1]
    inputs = inputs / 255
    mean = np.array(mean)
    std  = np.array(std)
    mean = mean[:, None, None]
    std  = std[:, None, None]
    out = (inputs - mean) / std
    return out

def stack_2images(img1, img2, weights=0.5):
    stack_img = cv2.addWeighted(img1, weights, img2, 1-weights, 0)
    return stack_img
