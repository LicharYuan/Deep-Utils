import cv2
import numpy as np

class DrawImg(object):
    """In-place draw"""
    def __init__(self, img):
        self.img = img
        self.imgh, self.imgw = img.shape[:2]
        self.name = self.__class__.__name__ # show windows name

    def draw_mask(self, mask):
        draw_mask_on_imgs(self.img, mask[None,:])
    
    def draw_bbox(self, bbox, text=None, title=None, loc="bl", tsize=0.5):
        if not isinstance(bbox[0], np.ndarray):
            bbox = bbox[None,:]
        draw_bboxes_on_imgs(self.img, bbox, text=text, title=title, loc=loc, text_scale=tsize)
    
    def save(self, savepath):
        cv2.imwrite(savepath, self.img)
    
    def show(self, name=None):
        name = str(name) if name else self.name
        show_img(self.img, name)
    
    def add_title(self, title=None, loc="bottom"):
        fontColor = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        fontscale = 0.8
        if loc == "bottom":
            pos = (10, self.imgh - 20)
        elif loc == "top":
            pos = (10, 20)
        img = cv2.putText(self.img, title, pos, font, fontscale, fontColor, thickness)
        return img


def show_img(img, name="img", resize=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if resize:
        cv2.resizeWindow(name, 1920, 1080)
    cv2.imshow(name, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bboxes_on_imgs(img, bboxes, savepath=None, text=None, title=None, loc="bl", text_scale=0.5):
    imgh, imgw = img.shape[:2]
    for i, bbox in enumerate(bboxes):        
        x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        width = x2-x1
        height = y2-y1
        img = cv2.rectangle(img, (x1, y2), (x2, y1), (255,0,0), 2)
        if text is not None:
            bottomLeftCornerOfText = (x1, y2)
            TopLeftCornerOfText = (x1, y1)
            bottomRightCornerOfText = (x2, y2)
            TopRightCornerOfText = (x2, y1)
            if loc == "bl":
                position = bottomLeftCornerOfText
            elif loc == "tl":
                position = TopLeftCornerOfText
            elif loc == "br":
                position = bottomRightCornerOfText
            elif loc == "tr":
                position = TopRightCornerOfText
            else:
                # default
                position = bottomLeftCornerOfText
                
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontColor              = (0,0,255)
            thickness = 2 
            fontscale = text_scale
            img = cv2.putText(img, text[i], position, font, fontscale, fontColor, thickness)
        
    if title:
        fontColor = (255, 255, 255)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        fontscale = 0.8
        img = cv2.putText(img, title, (imgw//2, imgh-20), font, fontscale, fontColor, thickness)

    if savepath is not None:
        cv2.imwrite(savepath, img)

    return img
    

def draw_mask_on_imgs(img, masks, savepath=None):
    for mask in masks:
        cv2.drawContours(img, mask, -1, (0, 255, 0), 2)    
