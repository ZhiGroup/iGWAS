from torchvision.models import inception_v3
import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.nn import functional as F

# adapted from https://github.com/ronghuaiyang/arcface-pytorch

class Embedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.inception = inception_v3(pretrained=True)
        self.inception.aux_logits = False
        self.inception.fc = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, dim)
        self.act = nn.PReLU()
        self.fc3 = nn.Linear(dim, 5)

    def forward(self, x):
        act1 = self.inception(x)
        act2 = self.fc1(self.act(act1))
        out = self.fc2(self.act(act2))
        out2 = self.fc3(out)
        return out, out2

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, device, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features).to(device))
        nn.init.xavier_uniform_(self.weight)
        self.device = device

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(self.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
 