from torch import nn
from .time_lagged import TIME_LAGGED_AE
from .slow_fast_evolve import EVOLVER


def weights_normal_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None: nn.init.zeros_(m.bias)