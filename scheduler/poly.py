from torch.optim.lr_scheduler import _LRScheduler

# From https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/schedulers/schedulers.py
# Also https://github.com/cmpark0126/pytorch-polynomial-lr-decay/blob/master/torch_poly_lr_decay/torch_poly_lr_decay.py

# LARS paper uses poly(2) lr schedule policy instead of multi step

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, T_max, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.T_max = T_max
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.T_max:
            return [base_lr for base_lr in self.base_lrs]
        factor = (1 - self.last_epoch / float(self.T_max)) ** self.gamma
        return [base_lr * factor for base_lr in self.base_lrs]
