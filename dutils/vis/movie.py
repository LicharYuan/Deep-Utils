
from dutils.process.image import resize_img
import imageio
import skvideo
import os
from skimage.transform import resize
import cv2
import numpy as np
from PIL import Image

class GenAVI(object):
    def __init__(self, img_list, dst, numh=1, numw=1):
        # put same frame in list if multi
        # format be like: [imgsA_list, imgsB_list, ...]
        assert len(img_list) > 1, "number of imgs must large than 1"
        self.img_list = img_list
        self.show_multi = numh * numw > 1
        self.numh = numh
        self.numw = numw
        self.dst = dst
        self._lens = len(img_list)
        if self.show_multi:
            assert len(img_list) == numh * numw,  f"{len(img_list)}, {numh}, {numw}"
    
    def __repr__(self):
        _msg =  f"Show Multi={self.show_multi}, \n" + \
                f"Length of ImgList={self._lens}, \n" + \
                f"numh, numw={self.numh}, {self.numw} ."                
        return _msg
    
    def make(self, fps=10):
        if self.show_multi:
            make_multi_avi(self.img_list, self.numh, self.numw, self.dst, fps=10)
        else:
            make_avi(self.img_list, avi_path=self.dst, fps=fps)
    
def show_two_gif(targets, preds, new_path="./merge.gif"):
    # use hstack merge gif
    gif1 = imageio.get_reader(targets)
    gif2 = imageio.get_reader(preds)
    number_of_frames = min(gif1.get_length(), gif2.get_length()) 
    new_gif = imageio.get_writer(new_path)
    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()    
    new_gif.close()

def show_multi_gif(new_path, *args):
    # * 目前只单行显示
    gif_list = []
    for ele in args:
        gif_list.append(imageio.get_reader(ele))
    gif_frame_list = []
    for gif in gif_list:
        gif_frame_list.append(gif.get_length())
    frame = min(gif_frame_list)
    new_gif = imageio.get_writer(new_path)
    for ids in range(frame):
        new_frame = []
        for img in gif_list:
            frame_img = img.get_next_data()
            new_frame.append(frame_img)
            img.close()
        new_frame = np.hstack(new_frame)
        new_gif.append_data(new_frame)

    new_gif.close()

def show_multi_avi(new_path, *args):
    # * 目前只单行显示
    video_list = []
    num = 0
    for ele in args:
        video_list.append(cv2.VideoCapture(ele))
        num += 1
        print(ele, num)
    fps_list = []
    h_list = []
    w_list = []
    f_list = []
    for video in video_list:
        fps_list.append(video.get(cv2.CAP_PROP_FPS))
        h_list.append(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_list.append(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_list.append(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = min(fps_list)
    h = int(min(h_list))
    w = int(min(w_list))
    frame = int(min(f_list))
    img_size = (w*num, h)

    out = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'XVID'), fps, img_size)
    for j in range(frame):
        for i, video  in enumerate(video_list):
            ret, frame_video = video.read()
            frame_video = cv2.resize(frame_video, (w, h))
            if i == 0:
                new_video = cv2.resize(frame_video, (w*num, h))
            
            new_video[:, i*w:(i+1)*w] = frame_video
        out.write(new_video)

    out.release()

def make_gif(imgs_list, gif_path="./tmp.gif", img_size=None, **kwargs):
    imgs = []
    for i, ele in enumerate(imgs_list):
        if img_size:
            img = imageio.imread(ele)
            img = resize(img, img_size)
            imgs.append(img)
        else:
            imgs.append(imageio.imread(ele))
    imageio.mimsave(gif_path, imgs, **kwargs)
    return gif_path

def make_dir_gif(fdir,  gif_path="./tmp.gif", img_size=None):
    imgs = []
    files = os.listdir(fdir)
    for ele in files:
        if ele.endswith(".jpg") or ele.endswith(".png") or ele.endswith(".bmp"):
            img_path = os.path.join(fdir, ele)
            imgs.append(img_path)
    make_gif(imgs, gif_path, img_size)

def make_avi(img_list, avi_path="./tmp.avi", img_size=None, fps=10,):
    assert avi_path.endswith(".avi")
    if img_size is None:
        # use first img shape as default
        img_size = cv2.imread(img_list[0]).shape[:2]  
        h, w = img_size
        img_size = (w, h)
    out = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, img_size)
    for img in img_list:
        img = cv2.resize(cv2.imread(img), img_size)
        out.write(img)
    out.release()


def make_multi_avi(img_lists, numh, numw, avi_path="./tmp.avi", img_size=None, fps=10, ):
    # same frame [[imga1, imgb1, imgc1, imgd1], ... ]
    assert avi_path.endswith(".avi") 
    if img_size is None:
        # use first img shape as default
        img_size = cv2.imread(img_lists[0][0]).shape[:2]  
    h, w = img_size
    avi_size = (w*numw, h*numh, ) # weight, height
    num_frame = min([len(ele) for ele in img_lists])
    # num_frame = 10
    out = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, avi_size)
    for frame_idx in range(num_frame):
        each_pair = [ele[frame_idx] for ele in img_lists]
        frame_img = np.empty([h*numh, w*numw, 3], dtype=np.uint8)
        for idx, img in enumerate(each_pair):
            try:
                img = cv2.imread(img)
                img = resize_img(img, h, w)
            except SystemError as e:
                print(idx, each_pair, "loc error")
                exit()
            _indexh = idx // h
            _indexw = idx % w
            frame_img[_indexh*h:_indexh*h+h, _indexw*w:_indexw*w+w] = img            
        
        out.write(frame_img)  
    out.release()

def make_dir_avi(fdir, avi_path="./tmp.avi", img_size=None, fps=10):
    imgs = []
    files = os.listdir(fdir)
    for ele in files:
        if ele.endswith(".jpg") or ele.endswith(".png") or ele.endswith(".bmp"):
            img_path = os.path.join(fdir, ele)
            imgs.append(img_path)
    make_avi(imgs, avi_path, img_size, fps=fps)

def make_left_gif_right_static(gif_imgs, static_img, img_size, gif_path="./left_gif_right_static.gif", **kwargs):
    imgs = []
    if isinstance(static_img, str):
        static_img = imageio.imread(static_img)
    # static_img = static_img[:,:,[2,1,0]]
    for i, ele in enumerate(gif_imgs):
        # uimg already load
        if isinstance(ele, str):
            ele = imageio.imread(ele)
        img = resize(ele, img_size)
        # img = img[:,:,[2,1,0]] 
        # imageio.imsave(f"./debug_{str(i)}.jpg", img)
        # static_img = resize(static_img, img_size)
        static_img = cv2.resize(static_img, (img_size[1], img_size[0]))
        merge_img = (255. * np.hstack([img, static_img])).astype(np.uint8)
        # merge_img = 
        imgs.append(merge_img)
    
    imageio.mimsave(gif_path, imgs, **kwargs)
