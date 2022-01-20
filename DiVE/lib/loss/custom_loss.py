import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
import numpy as np


class CrossEntropyFlat(nn.Cell):
    def __init__(self, para_dict=None):
        super(CrossEntropyFlat, self).__init__()
        cfg = para_dict["cfg"]
        self.s = cfg.LOSS.SCALE
        self.alpha = cfg.LOSS.CrossEntropyFlat.alpha
        self.cast = ops.Cast()
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def construct(self, output, target, reduction=True):
        loss_a = F.cross_entropy(self.s * output, target)
        target_b = target.clone()
        target_b *= 0
        target_b += 1
        target_b = self.cast(target_b, ms.int64)
        loss_b = self.cross_entropy(self.s * output, target_b)
        loss = loss_a * self.alpha + loss_b * (1 - self.alpha)
        return loss


class GroupSoftmax(nn.Cell):
    def __init__(self, para_dict=None):
        super(GroupSoftmax, self).__init__()
        cfg = para_dict["cfg"]
        self.s = cfg.LOSS.SCALE
        self.num_classes = para_dict["num_classes"]
        self.split1 = self.num_classes // 3
        self.split2 = self.num_classes // 3 * 2
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def construct(self, output, target):
        target_split1 = (target < self.split1).nonzero().view(-1)
        target_split2 = ((target >= self.split1) * (target < self.split2)).nonzero().view(-1)
        target_split3 = (target >= self.split2).nonzero().view(-1)

        loss1 = self.cross_entropy(self.s * output[target_split1, :self.split1], target[target_split1])
        loss2 = self.cross_entropy(self.s * output[target_split2, self.split1:self.split2],
                                   target[target_split2] - self.split1)
        loss3 = self.cross_entropy(self.s * output[target_split3, self.split2:], target[target_split3] - self.split2)
        return loss1 + loss2 + loss3


class BCE_KL(nn.Cell):
    r"""Binary Cross Entropy

    Equation:
    p_j = sigmoid(z_j)
    loss = - sigma(y_j * log(p_j) + (1 - y_j) * log(1 - p_j))

    Experiment Results:
    CIFAR100-LT100 ~37.39%
    """

    def __init__(self, para_dict=None):
        super(BCE_KL, self).__init__()
        self.s = 1
        # self.device = para_dict["device"]
        self.num_classes = para_dict["num_classes"]
        self.one_hot = ops.OneHot()
        self.cast = ops.Cast()
        self.binary_cross_entropy_with_logits = ops.BinaryCrossEntropy()
        self.sigmoid = ops.Sigmoid()

    def construct(self, output, output_t, target, reduction='mean'):
        if len(list(target.size())) == 1:
            target = self.one_hot(target, self.num_classes)
        elif target.dtype == ms.int64:
            target = self.cast(target, ms.float32)
        bce_loss = self.binary_cross_entropy_with_logits(output, target)

        target_t = self.sigmoid(output_t)
        kl_loss = self.binary_cross_entropy_with_logits(output, target_t)

        loss = 0.1 * bce_loss + 0.9 * kl_loss
        return loss


class MarginCrossEntropy(nn.Cell):
    def __init__(self, para_dict=None):
        super(MarginCrossEntropy, self).__init__()
        cfg = para_dict["cfg"]
        self.s = cfg.LOSS.SCALE
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def construct(self, output, target, reduction=True):
        margin = msnp.abs(output[range(output.size(0)), target] / 2)
        output[range(output.size(0)), target] -= margin
        loss = self.cross_entropy(self.s * output, target)
        return loss


class BalancedCrossEntropyLabelSmooth(nn.Cell):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, para_dict=None):
        super(BalancedCrossEntropyLabelSmooth, self).__init__()
        self.para_dict = para_dict
        self.num_classes = para_dict["num_classes"]
        self.num_class_list = para_dict["num_class_list"]
        self.epsilon = 0.8
        # self.device = para_dict['device']
        self.logsoftmax = ops.LogSoftmax(axis=1)

        self.n_cls = ms.Tensor(self.num_class_list)
        n_mean = self.n_cls.mean()
        self.weight = n_mean - self.n_cls
        self.weight[self.weight < 0] = 0
        self.weight = self.weight / self.weight.sum()
        self.weight = self.weight

        self.mins_ratio = (self.n_cls - n_mean) / self.n_cls
        self.mins_ratio[self.mins_ratio < 0] = 0
        self.unsqueeze = ops.ExpandDims()

    def construct(self, inputs, targets):
        r"""
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets_onehot = msnp.zeros(log_probs.size()).scatter_(1, self.unsqueeze(targets, 1), 1)
        mins_ratios = self.mins_ratio[targets]
        targets = (self.epsilon * mins_ratios).view(-1, 1) * self.weight + (
                1 - (self.epsilon * mins_ratios).view(-1, 1)) * targets_onehot
        loss = (- targets * log_probs).sum(1).mean(0)
        return loss


class BatchKL(nn.Cell):
    r"""negative KL in a Batch

    loss defined by myself

    thoughts: let label distribution in a batch to be diverse, so get attention to the tail classes

    """

    def __init__(self, para_dict=None):
        super(BatchKL, self).__init__()
        self.para_dict = para_dict
        self.num_classes = para_dict["num_classes"]
        # self.device = para_dict['device']
        self.t = para_dict['cfg'].LOSS.BATCHKL.temperature
        self.softmax = ops.Softmax(1)
        self.unsqueeze = ops.ExpandDims()
        print('batchkl temperature = ', self.t)

    def construct(self, inputs, targets):
        r"""
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        probs = self.softmax(inputs / self.t)
        # log_probs = F.log_softmax(inputs / self.t, dim=1)
        log_probs = msnp.log(probs.clamp(min=1e-5, max=1.))

        KL_loss = ((probs * log_probs).sum(1).view(-1, 1) - torch.mm(probs, log_probs.transpose(0, 1)))

        target_onehot = msnp.zeros(inputs.size()).scatter_(1, self.unsqueeze(targets, 1), 1)
        mask = torch.mm(target_onehot, target_onehot.T)
        mask = 1 - mask

        KL_loss = (KL_loss * mask).sum() * (self.t ** 2) / mask.sum()
        loss = - KL_loss
        return loss


class Detached_CrossEntropy(nn.Cell):
    def __init__(self, para_dict=None):
        super(Detached_CrossEntropy, self).__init__()
        cfg = para_dict["cfg"]
        self.s = cfg.LOSS.SCALE
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def construct(self, output, target):
        logits_mean = output.mean(1)
        logits_std = output.std(1)
        output -= logits_mean.view(-1, 1)
        output /= logits_std.view(-1, 1)
        detached_target_logits = output[range(output.size(0)), target]
        output[range(output.size(0)), target] = detached_target_logits
        print(self.s * output)
        loss = self.cross_entropy(self.s * output, target)
        return loss
