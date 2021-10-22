#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import torch
import h5py
import re
import pickle as pkl
from model import Embedding, AddMarginProduct
import pandas
from itertools import chain
import numpy as np
from glob import glob


class MyDataset(Dataset):
    def __init__(self, tsfm, h5_file='precomputed.h5', lim=40000):
        '''
        pattern_dict contains "img", "mask" and other data attributes as keys,
        and has correspoinding regular expression pattern and
        processing function as values.
        '''
        self.f = h5py.File(h5_file, 'r')
        self.lim = lim
        df1 = pandas.read_csv('trainLabels.csv')
        df2 = pandas.read_csv('retinopathy_solution.csv')
        db_labels = dict(zip(list(df1.image) + list(df2.image),
                             list(df1.level) + list(df2.level)))
        labels = list(pkl.load(open('fundus_im_quality.pkl',
                                    'rb')).keys())[:lim]
        ids = list(map(lambda x: re.match('.*?(\d+)', x).group(1), labels))
        img_names = list(map(lambda x: re.match('(?:train|test)/(.*)?.jpeg',
                                                x).group(1), labels))
        distinct_ids = set(ids)
        id_mapper = dict(zip(distinct_ids, range(len(ids))))
        self.ids = list(map(lambda x: id_mapper[x], ids))
        self.levels = list(map(lambda x: db_labels[x], img_names))
        self.num_class = max(self.ids) + 1
        self.tsfm = tsfm
        self.data = self.f['imgs']

    def __len__(self):
        return self.lim

    def __getitem__(self, idx):
        return (self.tsfm(self.data[idx, ...]), torch.tensor(self.ids[idx]),
                torch.tensor(self.levels[idx]))

    def __del__(self):
        self.f.close()


class RandomFlip:
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img, **kwargs):
        if random.random() < 0.5:
            img = img[:, :, ::-1] # horizontal flip
        return img


class RandomRotate:
    def __call__(self, img, **kwargs):
        h, w = img.shape[-2:]
        M = cv2.getRotationMatrix2D((w//2, h//2), random.random()*360, 1)
        img = cv2.warpAffine(img.transpose(1, 2, 0), M, (w, h))
        return img[np.newaxis, ...]

class Normalize:
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img, **kwargs):
        img = img.repeat(3, 0)
        img -= self.mean.reshape(3, 1, 1)
        img /= self.std.reshape(3, 1, 1)
        return img


class ToTensor:
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __call__(self, img, **kwargs):
        img_tensor = torch.from_numpy(img.astype('f'))
        return img_tensor


class Compose:
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, **kwargs)
        return img


if __name__ == "__main__":
    hidden_dim = 128
    dr_prediction_weights = 0.1
    device = torch.device("cuda:0")
    model = Embedding(hidden_dim).to(device)
    epoch_start = 0
    tsfm = Compose([RandomFlip(),
                    RandomRotate(),
                    Normalize(),
                    ToTensor()
                    ])
    dataset = MyDataset(tsfm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    metric_fc = AddMarginProduct(hidden_dim, dataset.num_class, device)
    optimizer = torch.optim.Adam(chain(model.parameters(),
                                       metric_fc.parameters()), lr=1e-4)
    loss = torch.nn.CrossEntropyLoss().to(device)
    if glob('*.pth'):
        model_wts, metric_wts = sorted([(int(re.match('.*?_wts_epoch_(\d+).pth', x).group(1)), x) for x in glob('*.pth')], reverse=True)[:2]
        model.load_state_dict(torch.load(model_wts[1]))
        metric_fc.load_state_dict(torch.load(metric_wts[1]))
        epoch_start = model_wts[0]+1
    for epoch in range(epoch_start, 500):
        for data, label1, label2 in dataloader:
            data = data.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            feature, out1 = model(data)
            output = metric_fc(feature, label1)
            l = loss(output, label1) + loss(out1, label2) * dr_prediction_weights
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'model_wts_epoch_{}.pth'.format(epoch))
        torch.save(metric_fc.state_dict(), 'metric_fc_wts_epoch_{}.pth'.format(epoch))
        
        
        
        
        
        
    