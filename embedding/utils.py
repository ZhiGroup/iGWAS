import cv2
import numpy as np
from skimage.transform import resize


# def find_circle(im):
#     if im.ndim == 3:
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
#     im = cv2.medianBlur(im.astype('uint8'), 101) * np.where(im==0, 0, 1).astype('uint8')
#     circle = cv2.HoughCircles(im,
#                           cv2.HOUGH_GRADIENT, 2, max(im.shape[0], im.shape[1]), param1=10,
#                           param2=20, minRadius=max(im.shape[0], im.shape[1])//3, maxRadius=0)[0, 0]
#     return np.round(circle).astype('int')

# def crop_im_by_circle(im, mask=None):
#     h, w = im.shape[:2]
#     try:
#         x, y, r = find_circle(im)
#     except:
#         return None, None, None
#     margin = 5
#     left = max(0, x-r-margin)
#     right = min(w, r+x+margin)
#     top = max(0, y-r-margin)
#     bot = min(h, r+y+margin)
#     return im[top:bot, left:right, ...],\
#            None if mask is None else mask[top:bot, left:right, ...],\
#            [min(x, 1) for x in [x/r, y/r, (w-x)/r, (h-y)/r]]
def crop_im_by_circle(img):
    downscale = 128
    h, w = img.shape[:2]
    im = cv2.resize(img, (downscale, downscale), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros(im.shape[:2], dtype='uint8')
    bgdModel = np.zeros((1, 65))
    fgdModel = np.zeros((1, 65))
    rect = (1, 1, downscale-2, downscale-2)
    cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    kernel = np.ones((5, 5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask[32:96, 32:96] = 1
    cv2.grabCut(im,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    left, up, width, height = cv2.boundingRect(mask%2)
    left = int(left*w/128)
    up = int(up*h/128)
    width = int(width*w/128)
    height = int(height*h/128)
    if width > 0 and height > 0:
        return cv2.resize(img[up: up+height, left:left+width, :], (512, 512))
    return

def im_resize(im, size, anti_aliasing):
    w, h = size
    return resize(im.astype('d'), (w, h, im.shape[2]), mode='constant', anti_aliasing=anti_aliasing).astype('f')