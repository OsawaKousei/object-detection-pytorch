# 0-75 epochs: 10e-3 -> 10e-2
# 75-105 epochs: 10e-2 -> 10e-3
# 105-135 epochs: 10e-3 -> 10e-4

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1):
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self) -> list:
        if self.last_epoch < 75:
            return [base_lr * (10**1) for base_lr in self.base_lrs]
        elif self.last_epoch < 105:
            return [base_lr * (10**0) for base_lr in self.base_lrs]
        else:
            return [base_lr * (10**-1) for base_lr in self.base_lrs]
