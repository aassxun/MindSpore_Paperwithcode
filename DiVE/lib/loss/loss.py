import mindspore as ms
import mindspore.nn as nn
import numpy as np
from .custom_loss import *
from .general_loss import *
from .kd_loss import *
from .longtail_loss import *

class MultiLoss(nn.Cell):
    r"""
        It's a implementation of multi-loss Optimization. Use one backbone and one classifier, but multi-loss,
    mainly classification loss.

    Usage:
    Fill the configures in the xxx.yaml
    -MULTI_LOSS.TYPE        which is a list of loss type
    -MULTI_LOSS.WEIGHT      which is a list of loss weight

    Notes: The length of TYPE and WEIGHT should match
    """
    def __init__(self, para_dict=None):
        super(MultiLoss, self).__init__()
        cfg = para_dict["cfg"]
        self.criterions = cfg.LOSS.MULTI_LOSS.TYPE
        self.weights = cfg.LOSS.MULTI_LOSS.WEIGHT

        assert len(self.criterions) == len(self.weights), '[ERROR] number of criterions and weights not match!'

        self.criterions = [eval(item)(para_dict=para_dict) for item in self.criterions]

    def construct(self, output, target):
        losses = [item(output, target) * self.weights[i] for i, item in enumerate(self.criterions)]
        # print(losses)
        # exit()
        loss = sum(losses)
        return loss