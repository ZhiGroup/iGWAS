import numpy as np
import torch
import random
from utils import im_resize, crop_im_by_circ_ratio
import cv2
import h5py


class RandomCropResize:
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, mask, **kwargs):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            mask_crop = mask[y1:h-y1, x1:w-x1]

            img_crop = im_resize(img_crop, (w, h), 1)
            mask_crop = im_resize(mask_crop, (w,h), 0)
            return img_crop, mask_crop
        else:
            return img, mask



class RandomFlip:
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img, mask, **kwargs):
        if random.random() < 0.5:
            img = img[:, ::-1, :] # horizontal flip
            mask = mask[:, ::-1, :] # horizontal flip
        return img, mask

class RandomRotate:
    def __call__(self, img, mask, **kwargs):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            M = cv2.getRotationMatrix2D((w//2, h//2), random.random()*360, 1)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        return img, mask

class RandomBackground:
    def __init__(self, target_h5):
        self.target = target_h5
    def __call__(self, img, mask, fg, ratios, **kwargs):
        if random.random() < 0.5:
            with h5py.File(self.target, "r") as f:
                random_target = random.choice(list(f.keys()))
                bg = f[random_target]['bg'][:]
                bg_ratios = f[random_target]['ratios'][:]
                fg, bg, mask = crop_im_by_circ_ratio([fg, bg, mask], ratios, bg_ratios)
                bg = im_resize(bg, img.shape[:2], 1)
                fg = im_resize(fg, img.shape[:2], 1)
                mask = im_resize(mask, img.shape[:2], 0)
                img = fg + bg
        return img, mask
                
        
class Normalize:
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, img, mask, **kwargs):
        img = img.astype(np.float32)
        img -= self.mean.reshape(1, 1, 3)
        img /= self.std.reshape(1, 1, 3)
        return img, mask

class ToTensor:
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __call__(self, img, mask, **kwargs):
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img.astype('f'))
        mask_tensor =  torch.from_numpy(mask.astype('f'))
        return img_tensor, mask_tensor

class Compose:
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        for t in self.transforms:
            img, mask = t(**kwargs)
            kwargs['img'] = img
            kwargs['mask'] = mask
        return img, mask