import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
import mindspore.numpy as msnp
from ..loss_base import CrossEntropy


class FocalLoss(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.

        Equation: Loss(x, class) = - (1-sigmoid(p^t))^gamma \log(p^t)

    Focal loss tries to make neural networks to pay more attentions on difficult samples.

    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
    """

    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__(para_dict)
        self.gamma = self.para_dict['cfg'].LOSS.FocalLoss.GAMMA  # hyper-parameter
        self.sigmoid = nn.Sigmoid()
        self.binary_cross_entropy_with_logits = ops.BinaryCrossEntropy('none')

    def construct(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets]) \
            if self.weight_list is not None else \
            ms.Tensor(msnp.ones(targets.shape[0]))
        label = self.onehot(targets, self.num_classes)
        p = self.sigmoid(inputs)
        focal_weights = msnp.float_power((1 - p) * label + p * (1 - label), self.gamma)
        loss = self.binary_cross_entropy_with_logits(inputs, label) * focal_weights
        loss = (loss * weight.view(-1, 1)).sum() / inputs.shape[0]
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = ms.Tensor(np.array([1 for _ in self.num_class_list]))
        else:
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = ms.Tensor(np.array([min(self.num_class_list) / N for N in self.num_class_list]))
            else:
                self.weight_list = ms.Tensor(np.array([1 for _ in self.num_class_list]))
