import math

from torch.optim.lr_scheduler import _LRScheduler

class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_cycles, num_training_steps, warmup_method='log', last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_cycles = num_cycles
        self.num_training_steps = num_training_steps
        self.warmup_method = warmup_method
        self._validate_warmup_method()
        super().__init__(optimizer, last_epoch, verbose)

    def _validate_warmup_method(self):
        valid_methods = ['log', 'exp',]
        if self.warmup_method not in valid_methods:
            raise ValueError(f"Invalid warmup_method. Expected one of: {valid_methods}")

    def get_lr(self):
        mult = None
        if self.last_epoch < self.num_warmup_steps:
            if self.warmup_method == 'log':
                mult = 0.10 ** (self.num_warmup_steps - self.last_epoch)
            else:
                mult = 2 ** -(self.num_warmup_steps - self.last_epoch)
        else:
            progress = float(self.last_epoch - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))
        
        return [base_lr * mult for base_lr in self.base_lrs]

        