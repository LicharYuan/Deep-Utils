import cv2
import numpy as np
import random

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

def img_shift(img, shift_px=20):
    random_shift_x = random.randint(-shift_px, shift_px)
    random_shift_y = random.randint(-shift_px, shift_px)
    new_x = max(0, random_shift_x)
    orig_x = max(0, -random_shift_x)

    new_y = max(0, random_shift_y)
    orig_y = max(0, -random_shift_y)

    new_img = np.zeros_like(img)
    imgh, imgw = img.shape[:2]
    new_h = imgh - np.abs(random_shift_y)
    new_w = imgw - np.abs(random_shift_x)
    new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                    = img[orig_y:orig_y + new_h, orig_x:orig_x + new_w]
    
    return new_img
