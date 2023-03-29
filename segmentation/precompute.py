# -*- coding: utf-8 -*-
from esp_model import ESPNet
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from utils import crop_im_by_circle, im_resize
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import h5py
import os
import re


device = "cuda:0"
batch_size = 64


class MyDataset(Dataset):
    def __init__(self, pattern_dict):
        '''
        pattern_dict contains "img", "mask" and other data attributes as keys,
        and has correspoinding regular expression pattern and
        processing function as values.
        '''
        self.files = []
        for dataset_path, regex_ext in pattern_dict.items():
            for f in sorted(os.listdir(dataset_path)):
                if re.match(regex_ext, f):
                    self.files.append(dataset_path.strip('/') + '/' + f)
        self.data = np.zeros((len(self.files), 512, 512, 3), dtype='f')
        for i, f in enumerate(self.files):
            img = cv2.imread(f)
            img = crop_im_by_circle(img)
            img = im_resize(img, (512, 512), 1)
            if f.split('.')[0].split('_')[-1] == 'right':
                img = img[:, ::-1, :].copy()
            self.data[i] = img
        self.data -= self.data.mean(axis=(2, 3), keepdims=True)
        self.data /= self.data.std(axis=(2, 3), keepdims=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx].transpose(2, 0, 1)),
                self.files[idx].split('/')[-1].split('_')[0])  # subject id

if __name__ == "__main__":
    f = h5py.File('precomputed.h5', 'w')
    esp = ESPNet().to(device).eval()
    esp.load_state_dict(torch.load("esp_model_wts.pt"))
    d = MyDataset({'Dataset/Kaggle/train': '*.jpeg', 'Dataset/Kaggle/test': '*.jpeg'})
    f.create_dataset('imgs', (len(d), 1, 512, 512), dtype='f')
    dataloader = iter(DataLoader(d, batch_size=batch_size))
    for step in range(int((len(d)+batch_size-1)/batch_size)):
        print(step)
        with torch.no_grad():
            im = next(dataloader)[0].to(device)
            seg = F.softmax(esp(im), 1)[:, :1, ...].cpu().numpy()
            f['imgs'][step*batch_size:(step+1)*batch_size, ...] = seg
    f.close()
