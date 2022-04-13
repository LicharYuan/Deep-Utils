import cv2
    
class DrawImg(object):
    def __init__(self, img):
        self.img = img
        self.name = self.__class__.__name__

    def draw_mask(self, mask):
        draw_mask_on_imgs(self.img, mask[None,:])
    
    def draw_bbox(self, bbox, text=None, title=None):
        draw_bboxes_on_imgs(self.img, bbox[None,:], text=text, title=title)
       
    def save(self, savepath):
        cv2.imwrite(savepath, self.img)
    
    def show(self, name=None):
        name = name if name else self.name
        show_img(self.img, name)
        
        
def show_img(img, name="img", resize=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if resize:
        cv2.resizeWindow(name, 1920, 1080)
    cv2.imshow(name, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bboxes_on_imgs(img, bboxes, savepath=None, text=None, title=None):
    # text will add in leftCorner of bboxes
    cimg = img
    imgh, imgw = img.shape[:2]
    for i, bbox in enumerate(bboxes):        
        x1,y1,x2,y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        width = x2-x1
        height = y2-y1
        cimg = cv2.rectangle(cimg, (x1, y2), (x2, y1), (255,0,0), 2)
        if text is not None:
            bottomLeftCornerOfText = (x1, y2)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontColor              = (0,0,255)
            thickness = 2 
            fontscale = 0.5
            cimg = cv2.putText(cimg, text[i], bottomLeftCornerOfText, font, fontscale, fontColor, thickness)
        
    if title:
        fontColor = (255, 255, 255)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        fontscale = 0.5
        cimg = cv2.putText(cimg, title, (imgw//2, imgh-20), font, fontscale, fontColor, thickness)

    if savepath is not None:
        cv2.imwrite(savepath, cimg)

    return cimg

def draw_mask_on_imgs(img, masks, savepath=None):
    for mask in masks:
        cv2.drawContours(img, mask, -1, (0, 255, 0), 2)    

