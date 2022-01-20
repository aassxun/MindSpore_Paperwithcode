import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore.ops import functional as F
import numpy as np


class CrossEntropy(nn.Cell):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        cfg = para_dict["cfg"]
        self.s = cfg.LOSS.SCALE
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def construct(self, output, target, reduction=True):
        loss = self.cross_entropy(self.s * output, target)
        return loss


class BCE(nn.Cell):
    r"""Binary Cross Entropy

    Equation:
    p_j = sigmoid(z_j)
    loss = - sigma(y_j * log(p_j) + (1 - y_j) * log(1 - p_j))

    Experiment Results:
    CIFAR100-LT100 ~37.39%
    """

    def __init__(self, para_dict=None):
        super(BCE, self).__init__()
        self.s = 1
        # self.device = para_dict["device"]
        self.num_classes = para_dict["num_classes"]
        self.binary_cross_entropy_with_logits = ops.BinaryCrossEntropy()
        self.one_hot = ops.OneHot()
        self.cast = ops.Cast()

    def construct(self, output, targe):
        if len(list(target.size())) == 1:
            target = self.one_hot(target, self.num_classes)
        elif target.dtype == ms.int32:
            target = self.cast(target, ms.float32)
        loss = self.binary_cross_entropy_with_logits(output, target)
        return loss


class MaxEntropy(nn.Cell):
    r"""MaxEntropy Loss
    Reference:
    Dubey Abhimanyu et al. Maximum-Entropy Fine-Grained Classification. Neurips 2018.
    https://arxiv.org/abs/1809.05934

    Equation:
    loss = \sum_i^c p(i) log p(i)
    """

    def __init__(self, para_dict=None):
        super(MaxEntropy, self).__init__()
        cfg = para_dict["cfg"]
        self.s = cfg.LOSS.SCALE
        self.softmax = ops.Softmax()

    def construct(self, output, target):
        probs = self.softmax(output)
        log_probs = msnp.log(probs)

        loss = (probs * log_probs).sum(1).mean()

        return loss


class GCE(nn.Cell):
    r"""Gradient-boosting Cross Entropy Loss

    Refernece:
    Sun et al. Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes. AAAI2020.
    https://arxiv.org/pdf/1912.06842v1.pdf

    Equation:
    modify the softmax function of standard Corss Entropy.
    softmax = exp(x_i) /(exp(x_i) + sigma(exp(x_j))), where exp(x_j) is in the top-k biggest among all x

    Notes:
        In the original paper, it's used in fine-grained recognition for faster coverage and higher accuracy.
        In long-tail experiments, we found that the categories which has similar appearance are more likely to confuse
        together.
        GCE helps a little.(~39.67% for CIFAR100-LT100)
    """

    def __init__(self, para_dict=None):
        super(GCE, self).__init__()
        cfg = para_dict["cfg"]

        self.s = cfg.LOSS.SCALE
        self.top_k = cfg.LOSS.GCE.TOP_K
        # self.device = para_dict["device"]
        self.epoch = 0
        self.step_epoch = cfg.LOSS.GCE.DRW_EPOCH

        self.weight = ms.Tensor(para_dict["num_class_list"])
        self.sort = ops.Sort(1, True)
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.nll_loss = ops.NLLLoss()

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_top_k_log_softmax_prob(self, logits):
        max_logit = logits.max(axis=1, keepdims=True)[0]
        # max_logit = max_logit.detach()
        logits -= max_logit
        exp_logits = msnp.exp(logits)

        # a trick, sort twice to find the element idx in a sorted list
        _, idx = self.sort(exp_logits)
        _, rank = self.sort(idx)
        rank = rank.clone()

        mask = (rank < self.top_k)
        # mask = mask.detach()

        log_prob = logits - msnp.log((exp_logits * mask).sum(axis=1, keepdims=True) + 1e-5)
        return log_prob

    def construct(self, output, target):
        output *= self.s

        # max_logit = output.max(axis=1, keepdims=True)[0]
        # max_logit = max_logit.detach()
        # output -= max_logit

        # output += torch.log(torch.clamp(self.weight, min=1e-5, max=1.)).view(1,-1).detach()

        if self.epoch < self.step_epoch:
            loss = self.cross_entropy(output, target)
        else:
            log_prob = self.get_top_k_log_softmax_prob(output)
            loss = self.nll_loss(log_prob, target)

        # target_onehot = torch.zeros(output.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(self.device)
        #
        # loss = - (target_onehot * log_prob).sum(1).mean(0)

        return loss


class RCE(nn.Cell):
    r"""Reversed Cross Entropy
    Used in Symmetric Cross Entropy
    See in class SCELoss
    """

    def __init__(self, para_dict=None):
        super(RCE, self).__init__()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = para_dict['num_classes']
        self.softmax = ops.Softmax(1)
        self.one_hot = ops.OneHot()
        self.sum = ops.ReduceSum()

    def construct(self, pred, labels):
        pred = self.softmax(pred)
        # pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = self.one_hot(labels, self.num_classes)
        # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * self.sum(pred * msnp.log(label_one_hot), axis=1)).mean()
        return loss


class SCELoss(nn.Cell):
    r"""Symmetric Cross Entropy

    Reference:
    Wang et al. Symmetric Cross Entropy for Robust Learning with Noisy Labels. ICCV2019
    https://arxiv.org/abs/1908.06112v1

    Equation:
    loss = alpha * CE + (1 - alpha) * RCE
    -CE: Cross Entropy  (y * log p)
    -RCE: Reversed Cross Entropy (p * log y)

    Notes:
    Used in Noisy Label Problem, but found usful in long-tailed dataset problem too.
    The hyper-parameters alpha & beta seted both 0.5 always produce good results.

    Experiment Results:
    CIFAR100-LT100
    alpha   beta    top-1 acc
    =========================
    0.5     0.5     39.62%
    =========================
    """

    def __init__(self, alpha=0.5, beta=0.5, para_dict=None):
        super(SCELoss, self).__init__()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = para_dict['num_classes']
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()
        self.reversed_cross_entropy = RCE(para_dict=para_dict)

    def reset_epoch(self, epoch):
        r'''
        gradually change the hyper-parameters, but not work yet.
        Not in the original paper
        '''
        # self.alpha = 1 - ((epoch - 1) / 200) ** 2
        # self.beta = 1 - self.alpha
        pass

    def construct(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        rce = self.reversed_cross_entropy(pred, labels)

        loss = self.alpha * ce + self.beta * rce
        return loss


class MSELoss(nn.Cell):
    r"""Mean Square Error Loss
    loss = 1/N * sigma(y_hat_i - y_i)^2
    usage: Regression task
    """

    def __init__(self, para_dict=None):
        super(MSELoss, self).__init__()
        max_n = max(para_dict["num_class_list"])
        num_class_list = [item / max_n for item in para_dict["num_class_list"]]
        self.num_class_list = ms.Tensor(num_class_list)

    def construct(self, output, target):
        output = output.view(output.shape[0])
        target = self.num_class_list[target]
        loss = F.squeeze(output - target)
        return loss


class MultiMarginLoss(nn.Cell):
    def __init__(self, para_dict=None):
        super(MultiMarginLoss, self).__init__()
        self.multi_margin_loss = nn.MultiMarginLoss()

    def construct(self, output, target):
        output = output
        loss = self.multi_margin_loss(output, target)
        return loss


class CrossEntropyLabelSmooth(nn.Cell):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, para_dict=None):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.para_dict = para_dict
        self.num_classes = para_dict["num_classes"]
        self.epsilon = para_dict.get("epsilon", 0.1)
        self.device = para_dict['device']
        self.logsoftmax = ops.LogSoftmax(axis=1)
        self.unsqueeze = ops.ExpandDims()

    def construct(self, inputs, targets):
        r"""
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = msnp.zeros(log_probs.size()).scatter_(1, self.unsqueeze(targets, 1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(1).mean(0)
        return loss


class NCE(nn.Cell):
    r"""Negtive Learning

    Reference:
    NLNL: Negative Learning for Noisy Labels
    https://arxiv.org/abs/1908.07387?context=cs

    Equation:
    target' = Y/target
    loss_smaple_i = - sigma(target'_ij * log(1-output_ij))
    loss = 1/N * sigma(loss_sample_i)

    Args:
        self.ln_neg: number of negtive classes for a sample(random sample the negtive classes,
        ex. 110 for cifar100 in the original paper)
        self.weight: weight of each class

    Notes:
        Use for Noisy Label Learning in the original paper, trying to use it in long-tailed dataset problem, see details
        in class NL_CE.
    """

    def __init__(self, para_dict=None):
        super(NCE, self).__init__()
        self.num_classes = para_dict["num_classes"]
        # self.device = para_dict["device"]

        if self.num_classes == 100:
            self.ln_neg = 10

        elif self.num_classes == 10:
            self.ln_neg = 1
        else:
            raise NotImplementedError
        self.weight = ms.Tensor(self.num_classes).zero_() + 1.

        self.weight = self.weight
        self.unsqueeze = ops.ExpandDims()
        self.softmax=ops.Softmax()
        self.nll_loss=ops.NLLLoss()

    def construct(self, output, target):
        target_neg = (self.unsqueeze(target, -1).repeat(1, self.ln_neg)
                      + ms.Tensor((len(target), self.ln_neg), ms.int64).random_(1, self.num_classes)) % self.num_classes

        assert target_neg.max() <= self.num_classes - 1
        assert target_neg.min() >= 0
        assert (target_neg != target.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(target) * self.ln_neg

        s_neg = msnp.log(1. - self.softmax(output))
        s_neg *= self.weight[target].unsqueeze(-1).expand(s_neg.size())

        loss = self.nll_loss(s_neg.repeat(self.ln_neg, 1), target_neg.T.view(-1)) * float(
            (target_neg >= 0).sum()) / target.shape[0]
        return loss
