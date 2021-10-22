from torchvision.models import inception_v3
import torch
import torch.nn as nn
from torch.nn import functional as F


class QAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True)
        self.inception.fc = nn.Linear(2048, 1024)
        self.inception.aux_logits = False
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        act1 = self.inception(x)
        act2 = self.fc1(F.relu(act1))
        out = self.fc2(F.relu(act2))
        return torch.sigmoid(out)
    
