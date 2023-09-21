import torch
from IPython import embed; from sys import exit

class TopKAccuracy:
    def __init__(self, topk=(1,), ignore_idxs=(59,60,61)):
        self.topk = topk
        self.ignore_idxs = torch.tensor(ignore_idxs).unsqueeze(1)
        self.reset()

    def reset(self):
        self.total = 0
        self.correct = {k: 0 for k in self.topk}

    def update(self, outputs, targets):
        num_char = outputs.shape[2]
        targets = targets[:, 1:].flatten()
        outputs = outputs.view(-1, num_char)
        self.ignore_idxs = self.ignore_idxs.to(targets.device)
        mask = (targets != self.ignore_idxs).all(0)
        targets = targets[mask]
        outputs = outputs[mask]
        
        maxk = max(self.topk)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        for k in self.topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            self.correct[k] += correct_k.item()

        self.total += len(targets)

    def get_result(self):
        return {k: v/self.total for k, v in self.correct.items()}
