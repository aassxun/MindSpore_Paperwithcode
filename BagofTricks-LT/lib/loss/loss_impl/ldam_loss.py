import mindspore as ms
import numpy as np
import mindspore.ops as ops
from ..loss_base import CrossEntropy
import mindspore.numpy as msnp


class LDAMLoss(CrossEntropy):
    """
    LDAMLoss is modified from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).

    References:
    Cao et al., Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS 2019.

    Args:
        scale(float, double) : the scale of logits, according to the official codes.
        max_margin(float, double): margin on loss functions. See original paper's Equation (12) and (13)

    Notes: There are two hyper-parameters of LDAMLoss codes provided by official codes,
          but the authors only provided the settings on long-tailed CIFAR.
          Settings on other datasets are not avaliable (https://github.com/kaidic/LDAM-DRW/issues/5).
    """

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__(para_dict)
        s = self.para_dict['cfg'].LOSS.LDAMLoss.SCALE
        max_m = self.para_dict['cfg'].LOSS.LDAMLoss.MAX_MARGIN
        self.max_m = max_m
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = ms.Tensor(m_list[np.newaxis, :])
        self.m_list = m_list
        assert s > 0
        self.s = s
        # betas to control the **class-balanced loss (CVPR 2019)** weights according to LDAMLoss official codes
        if self.drw:
            self.betas = [0, 0.9999]
        else:
            self.betas = [0, 0]
        self.view = ops.Reshape()
        self.matmul = ops.MatMul(transpose_b=True)
        self.cast = ops.Cast()

    def update(self, epoch):
        """
        Adopt the class-balanced loss as default re-weighting method in drw according to LDAM official codes.
        Args:
            epoch: int
        """
        idx = 1 if epoch >= self.drw_start_epoch else 0
        per_cls_weights = (1.0 - self.betas[idx]) / (1.0 - np.power(self.betas[idx], self.num_class_list))
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight_list = ms.Tensor(per_cls_weights)

    def construct(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        index_float = self.onehot(targets)
        index = self.cast(index_float, ms.int32)
        batch_m = self.matmul(self.m_list, index_float)
        batch_m = self.view(batch_m, (-1, 1))
        x_m = inputs - batch_m
        outputs = msnp.where(index, x_m, inputs)
        return super(LDAMLoss, self)(self.s * outputs, targets)
