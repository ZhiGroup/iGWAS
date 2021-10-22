from torch import nn
import torch
import torch.nn.functional as F


class WeightedCategoricalCrossentropy(nn.Module):
    """ weighted_categorical_crossentropy
        
        mean reduction of weighted categorical crossentropy

        Args:
            * weights<tensor|nparray|list>: category weights
            * device<str|None>: device-name. if exists, send weights to specified device
    """
    def __init__(self, weight, device=None):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.weight=torch.tensor(weight)
        if device:
            self.weight=self.weight.to(device)

    def forward(self, inpt, targ):
        return weighted_categorical_crossentropy(inpt,targ,self.weight)


def weighted_categorical_crossentropy(inpt,targ,fg_weight):
    """ weighted_categorical_crossentropy

    Args:
        * inpt <tensor>: network prediction 
        * targ <tensor>: network target
        * weights<tensor|nparray|list>: category weights
    Returns:
        * mean reduction of weighted categorical crossentropy
    """
    weight=fg_weight.float()
    losses=((targ * F.log_softmax(inpt, 1))).float()
    fg = (targ[1]==1).float()
    weighted_losses_transpose=fg*weight*losses + losses*(1-fg)
    return -weighted_losses_transpose.mean()