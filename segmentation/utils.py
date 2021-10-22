import cv2
import numpy as np
from glob import glob
import os
import re
from collections import defaultdict
from skimage.transform import resize
from skimage.io import imread
import h5py


def find_circle(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
    im = cv2.medianBlur(im.astype('uint8'), 101) * np.where(im==0, 0, 1).astype('uint8')
    circle = cv2.HoughCircles(im,
                          cv2.HOUGH_GRADIENT, 2, max(im.shape[0], im.shape[1]), param1=10,
                          param2=20, minRadius=max(im.shape[0], im.shape[1])//3, maxRadius=0)[0, 0]
    return np.round(circle).astype('int')


def graphcut_crop(img):
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


def crop_im_by_circle(im, mask=None):
    h, w = im.shape[:2]
    try:
        x, y, r = find_circle(im)
    except:
        return None, None, None
    margin = 5
    left = max(0, x-r-margin)
    right = min(w, r+x+margin)
    top = max(0, y-r-margin)
    bot = min(h, r+y+margin)
    return im[top:bot, left:right, ...],\
           None if mask is None else mask[top:bot, left:right, ...],\
           [min(x, 1) for x in [x/r, y/r, (w-x)/r, (h-y)/r]]

def crop_im_by_circ_ratio(ims, ratios_a, ratios_b):
    """
    Crop parts from a image
    """
    fg, bg, mask = ims
    l = [None] * 3
    h, _ = fg.shape[:2]
    r = int(h/2)
    r1, r2, r3, r4 = [min(x, 1) for x in ratios_b/ratios_a]
    left = int(r*(1-r1))
    n_right = max(int(r*(1-r3)), 1)
    top = int(r*(1-r2))
    n_bot = max(int(r*(1-r4)), 1)
    l[0] = fg[top:-n_bot, left:-n_right, ...]
    l[2] = mask[top:-n_bot, left:-n_right, ...]
    h, _ = bg.shape[:2]
    r = int(h/2)
    r1, r2, r3, r4 = [min(x, 1) for x in ratios_a/ratios_b]
    left = int(r*(1-r1))
    n_right = max(int(r*(1-r3)), 1)
    top = int(r*(1-r2))
    n_bot = max(int(r*(1-r4)), 1)
    l[1] = bg[top:-n_bot, left:-n_right, ...]
    return l

def illumination_correction(im):
    bg = cv2.medianBlur(im.astype('uint8'), 31).astype('f')
    bg -= bg.mean()
    return im-bg, bg

def normalize(im):
    return (im - im.min(axis=(0, 1)))/(im.max(axis=(0, 1))-im.min(axis=(0, 1)))

def prepare_dict(folders, pattern):
    flist = sum([glob(f+'/*') for f in folders], [])
    d = defaultdict(list)
    for f in flist:
        fid = re.match(pattern, os.path.basename(f))
        if fid:
            d[fid.group(1)].append(f)
    return d

def im_resize(im, size, anti_aliasing):
    w, h = size
    return resize(im.astype('d'), (w, h, im.shape[2]), mode='constant', anti_aliasing=anti_aliasing).astype('f')

def prepare_datasets(datasets, target_hdf5):
    with h5py.File(target_hdf5, 'w') as f:
        img_cnt = 0
        for img_folders, mask_folders, img_pattern, mask_pattern in datasets.values():
            img_dict = prepare_dict(img_folders, img_pattern)
            mask_dict = prepare_dict(mask_folders, mask_pattern)
            ccs = np.zeros((3,)) # cumulative channel sum
            ccss = np.zeros((3,)) # cumulative channel square sum
            for im_id in mask_dict:
                print(im_id)
                img_cnt += 1
                mean_mask = combine_mask(mask_dict[im_id])
                im = cv2.imread(img_dict[im_id][0])
                im, mean_mask, ratios = crop_im_by_circle(im, mean_mask)
                if im:
                    im = im_resize(im, (512, 512), 1) * 255
                    ccs += (im - 128).sum((0, 1))/2**18
                    ccss += ((im - 128)**2).sum((0, 1))/2**18
                    fg, bg = illumination_correction(im)
                    mean_mask = im_resize(mean_mask, (512, 512), 0)
                    f.create_group(im_id)
                    f[im_id].create_dataset('img', data=im)
                    f[im_id].create_dataset('mask', data=mean_mask)
                    f[im_id].create_dataset('ratios', data=ratios)
                    f[im_id].create_dataset('fg', data=fg)
                    f[im_id].create_dataset('bg', data=bg)
            mean = ccs/img_cnt + 128
            var = (ccss - ccs**2/img_cnt)/img_cnt
            f.attrs['mean'] = mean
            f.attrs['std'] = np.sqrt(var)

def prepare_target_domain(folders, img_pattern, target_hdf5, num=1000):
    with h5py.File(target_hdf5, 'w') as f:
        img_dict = prepare_dict(folders, img_pattern)
        cnt = 0
        for im_id in img_dict:
            im = cv2.imread(img_dict[im_id][0])
            im, _, ratios = crop_im_by_circle(im)
            if sum(ratios) < 3.5:
                continue
            cnt += 1
            print(im_id)
            im = im_resize(im, (512, 512), 1) * 255
            fg, bg = illumination_correction(im)
            f.create_group(im_id)
            f[im_id].create_dataset('fg', data=fg)
            f[im_id].create_dataset('bg', data=bg)
            f[im_id].create_dataset('ratios', data=ratios)
            if cnt > num:
                return
            

def combine_mask(masks):
    def threshold(im):
        return np.where(im>0.5, 1, 0)
    mean_mask = np.mean(np.concatenate(list(map(lambda x: threshold(imread(x, 1))[..., np.newaxis],
                                                masks)), -1), axis=-1)
    mean_mask_f = np.where(mean_mask > 0, mean_mask, 0)[..., np.newaxis]
    mean_mask_b = np.where(mean_mask < 1, 1 - mean_mask, 0)[..., np.newaxis]
    mean_mask = np.concatenate((mean_mask_f, mean_mask_b), axis=-1)
    return mean_mask

def show(im, savefile=None, figsize=(40, 40)):
    from matplotlib.pyplot import figure, imshow, axis, savefig
    figure(figsize=figsize)
    if im.ndim == 3:
        im = im[..., ::-1]
    imshow((normalize(im)*255).astype('i'))
    if savefile:
        axis('off')
        savefig(savefile)