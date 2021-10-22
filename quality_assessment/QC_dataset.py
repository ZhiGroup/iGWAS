from torch.utils.data import Dataset
import pickle as pkl
import torch


class QCDataset(Dataset):
    def __init__(self):
        self.data = torch.load("fundus_QC_data.pth")
        self.label = list(pkl.load(open("quality_label.pkl", 'rb')).values())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
        
        