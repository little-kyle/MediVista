
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
import math

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def mkdir(FolderPath, rm = False):
    if not os.path.isdir(FolderPath):
        os.makedirs(FolderPath)
    elif rm and os.path.exists(FolderPath):
        shutil.rmtree(path=FolderPath)
        os.makedirs(FolderPath)
    # elif os.path.exists(FolderPath):
    # FolderPath = FolderPath + ' ' + timenow
    # shutil.rmtree(path=FolderPath)
    # os.makedirs(FolderPath)
    return FolderPath

class DSC():
    def __init__(self, epsilon=1e-5, ignore_index=None, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    def __call__(self, input, target, cal_mean = True):
#         print(input.shape)
        input = (input > 0.5).float()
        target = (target > 0.5).float()
        dice = compute_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index)
        if cal_mean:
            return dice.mean().item()
        return dice.item()
    
def compute_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)    

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)