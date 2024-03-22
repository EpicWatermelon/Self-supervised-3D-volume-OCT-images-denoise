import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss
class L0_Loss(nn.Module):
    def __init__(self, total_epoch, reduction = 'mean'):
        super(L0_Loss, self).__init__()
        self.total_epoch = total_epoch
        self.eps = 1e-8
        self.reduction = reduction

    def forward(self, x, y, current_epoch):
        # power is annealed linearly from 2 to 0 during training; current_epoch is from 0 to (total_epoch - 1)
        power = 2 * (self.total_epoch - current_epoch) / self.total_epoch
        # base is the sum of difference and epsilon
        base = torch.abs(x - y) + self.eps
        # compute loss
        ret = base ** power
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret
