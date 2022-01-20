import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from ..loss_base import CrossEntropy
import mindspore.numpy as msnp

class InfluenceBalancedLoss(CrossEntropy):
    r"""
    References:
    Seulki et al., Influence-Balanced Loss for Imbalanced Visual Classification, ICCV 2021.
    """

    def __init__(self, para_dict=None):
        super(InfluenceBalancedLoss, self).__init__(para_dict)

        ib_weight = 1.0 / np.array(self.num_class_list)
        ib_weight = ib_weight / np.sum(ib_weight) * self.num_classes
        self.ib_weight = ms.Tensor(ib_weight)
        self.use_vanilla_ce = False
        self.alpha = self.para_dict['cfg'].LOSS.InfluenceBalancedLoss.ALPHA
        self.sum3=ops.ReduceSum()
        self.softmax=ops.Softmax(1)

    def construct(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """

        if self.use_vanilla_ce:
            return super()(inputs, targets)

        assert 'feature' in kwargs, 'Feature is required in InfluenceBalancedLoss. \
                You should feed the features from lib/core/combiner.py, \
                and can see \
                    https://github.com/pseulki/IB-Loss/blob/751cd39e43dee4f6cb9fff2d3fb24acd633a22c3/models/resnet_cifar.py#L130 \
                for more details'

        feature = self.sum3(msnp.abs(kwargs['feature']), 1).view(-1, 1)
        grads = self.sum3(msnp.abs(self.softmax(inputs) - self.onehot(targets, self.num_classes)), 1)
        ib = grads * feature.view(-1)
        ib = self.alpha / (ib + 1e-3)
        ib_loss = super()(inputs, targets)*ib
        return ib_loss.mean()


    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.ib_weight
        else:
            self.weight_list = msnp.ones(self.ib_weight.shape)
            start = (epoch-1) // self.drw_start_epoch
            self.use_vanilla_ce = True
            if start:
                self.use_vanilla_ce = False
                self.weight_list = self.ib_weight