import torch.nn as nn

def activation_factory(activation_str):

    if activation_str == 'swish':
        return nn.SiLU()
    elif activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError(f'not supported {activation_str=}')
    
def initialization_factory(module, init_str):
    if init_str == 'xavier':
        init_fn = nn.init.xavier_uniform_
    elif init_str == 'he':
        init_fn = nn.init.kaiming_uniform_
    else:
        raise NotImplementedError(f'not supported {init_str=}')
    init_fn(module.weight)
    
    