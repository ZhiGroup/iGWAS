#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import copy
import torch
from model import ESPNet
from weighted_crossentropy import WeightedCategoricalCrossentropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import dataLoaders
import transforms
import h5py
import pickle as pkl
import numpy as np


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=1000):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            dataset_sizes = 0

            # Iterate over data.
            for images, masks in dataloaders[phase]:
                inputs = images.to(device)
                labels = masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                dataset_sizes += 1
                # statistics
            epoch_loss = running_loss / dataset_sizes

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'model_wts.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = torch.device("cuda")
    model = ESPNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    loss = WeightedCategoricalCrossentropy(20, device)
    target_h5 = 'DataSet/Pooled/targets.h5'
    data_h5 = 'DataSet/Pooled/datasets.h5'
    f = h5py.File(data_h5)
    train_tsfm = transforms.Compose([transforms.RandomBackground(target_h5),
                                     transforms.Normalize(f.attrs['mean'], f.attrs['std']),
                                     transforms.RandomFlip(),
                                     transforms.RandomRotate(),
                                     transforms.RandomCropResize(32),
                                     transforms.ToTensor()
                                     ])
    val_tsfm = transforms.Compose([transforms.Normalize(f.attrs['mean'], f.attrs['std']),
                                   transforms.ToTensor()])
    data_loaders = dataLoaders(f, {'train': train_tsfm, 'val': val_tsfm})
    train_model(model, loss, optimizer, scheduler, data_loaders, device)
    f.close()
    

