import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype


class CrossEntropy(nn.Cell):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        self.num_classes = self.para_dict["num_classes"]
        self.num_class_list = self.para_dict['num_class_list']

        self.weight_list = None
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.unsqueeze = ops.ExpandDims(0)
        self.sum2 = ops.ReduceSum()
        self.mul3 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

        # settings of defferred re-balancing by re-weighting (DRW)
        self.drw = self.para_dict['cfg'].TRAIN.TWO_STAGE.DRW
        self.drw_start_epoch = self.para_dict['cfg'].TRAIN.TWO_STAGE.START_EPOCH  # start from 1

    def construct(self, logit, label, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            label: ground truth labels with shape (batch_size)
        """
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        if self.weight_list is not None:
            weight = self.mul3(self.squeeze(self.weight_list), label)
            weight = self.sum2(weight, -1)
            loss = self.mul3(loss, weight)
        loss = self.mean(loss, -1)

        return loss

    def update(self, epoch):
        """
        Adopt cost-sensitive cross-entropy as the default
        Args:
            epoch: int. starting from 1.
        """
        pass
