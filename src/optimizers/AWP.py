import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20

from src.globals import PHRASE_PAD_VALUE


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma, output_device, start_epoch=0):
        super(AWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.output_device = output_device
        self.start_epoch = start_epoch

    def calc_awp(self, inputs_adv, targets, splits=1):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        output, _ = self.proxy(inputs_adv, targets[:, :-1])
        loss = - F.cross_entropy(output.permute(0,2,1), targets[:, 1:])
        output_dim = output.shape[-1]
        mask = torch.ones(output_dim).to(self.output_device)
        mask[PHRASE_PAD_VALUE] = 0.
        loss = loss * mask
        loss = loss.mean()
        loss = loss / splits
        
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

    def _restore(self, curr_epoch):
        if curr_epoch >= self.start_epoch:
            add_into_weights(self.model, self.diff, coeff=-1.0 * self.gamma)

    def __call__(self, X, y, curr_epoch, splits=1):
        if curr_epoch >= self.start_epoch:
            awp = self.calc_awp(X, y, splits)
            self.perturb(awp)
            self.diff = awp