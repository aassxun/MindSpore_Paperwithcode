import mindspore as ms
import mindspore.ops as ops
from ..loss_base import CrossEntropy
import mindspore.numpy as msnp

class BalancedSoftmaxCE(CrossEntropy):
    r"""
    References:
    Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.

    Equation: Loss(x, c) = -log(\frac{n_c*exp(x)}{sum_i(n_i*exp(i)})
    """

    def __init__(self, para_dict=None):
        super(BalancedSoftmaxCE, self).__init__(para_dict)
        self.bsce_weight = ms.Tensor(self.num_class_list)
        self.unsqueeze = ops.ExpandDims()

    def construct(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        logits = inputs + msnp.log(self.unsqueeze(self.weight_list, 0).expand(inputs.shape[0], -1))
        loss = super()(logits, targets)
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.bsce_weight
        else:
            self.weight_list = msnp.ones(self.bsce_weight.shape)
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.bsce_weight
