from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class MyDataset(Dataset):
    def __init__(self, h5_file_handle, transform):
        self.data = h5_file_handle
        self.im_ids = list(h5_file_handle.keys())
        self.transform = transform

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im = self.data[self.im_ids[idx]]
        image, mask = self.transform(**{attr: im[attr][...] for
                                        attr in im.keys()})
        return image, mask

def dataLoaders(h5_file_handle, transforms):
    fnum = len(list(h5_file_handle.keys()))
    split = int(fnum*0.7)
    idx = {'train': SubsetRandomSampler(np.arange(split)),
           'val': SubsetRandomSampler(np.arange(split, fnum))}
    data_loaders = {x: DataLoader(MyDataset(h5_file_handle, transforms[x]), batch_size=32, sampler=idx[x])
                    for x in ('train', 'val')}
    return data_loaders
    
