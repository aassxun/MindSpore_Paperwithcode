import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from ..loss_base import CrossEntropy
from loss import *


class DiVEKLD(CrossEntropy):
    """
    Knowledge Distillation

    References:
        Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
        Yin-Yin He et al., Distilling Virtual Examples for Long-tailed Recognition. ICCV 2021

    Equation:
        loss = (1-alpha) * ce(logits_s, label) + alpha * kld(logits_s, logits_t)

    """

    def __init__(self, para_dict=None):
        super(DiVEKLD, self).__init__(para_dict)

        self.power = para_dict["cfg"].LOSS.DiVEKLD.POWER if para_dict["cfg"].LOSS.DiVEKLD.POWER_NORM else 1.0
        self.T = para_dict["cfg"].LOSS.DiVEKLD.TEMPERATURE
        self.alpha = para_dict["cfg"].LOSS.DiVEKLD.ALPHA
        self.base_loss = eval(para_dict["cfg"].LOSS.DiVEKLD.BASELOSS)(para_dict)
        self.log_softmax = ops.LogSoftmax(1)
        self.softmax = ops.Softmax(1)
        self.sum3 = ops.ReduceSum(True)
        self.kl_div = ops.KLDivLoss()

    def construct(self, inputs_s, inputs_t, targets, **kwargs):
        logp_s = self.log_softmax(inputs_s / self.T)
        soft_t = self.softmax(self.softmax(inputs_t / self.T)) ** self.power
        soft_t /= self.sum3(soft_t, 1)
        # soft_t.detach_()
        kl_loss = (self.T ** 2) * self.kl_div(logp_s, soft_t)
        loss = self.alpha * kl_loss + (1 - self.alpha) * self.base_loss(inputs_s, targets)

        return loss

    def update(self, epoch):
        self.base_loss.update(epoch)
