import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
import mindspore.numpy as msnp


class FocalLoss(nn.Cell):
    r"""Focal Loss

    Reference:
    Lin et al. Focal Loss for Dense Object Detection. ICCV2017.
    https://arxiv.org/abs/1708.02002

    Equation:
    p_j = sigmoid(z_j)
    loss = - sigma(y_j * alpha * (1 - p_j)^gamma * log(p_j) +
                   (1 - y_j) * (1 - alpha) * p_j^gamma * log(1 - p_j))
    alpha: balance factor
    gamma: hard sample mining

    """

    def __init__(self, para_dict=None, gamma=1., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.s = 1
        # self.device = para_dict["device"]
        self.gamma = gamma
        self.alpha = alpha
        self.one_hot = ops.OneHot()
        self.binary_cross_entropy_with_logits = ops.BinaryCrossEntropy('none')

    def construct(self, output, target, **kwargs):
        if len(list(target.size())) == 1:
            target = self.one_hot(target, self.num_classes)
        BCLoss = self.binary_cross_entropy_with_logits(output, target)

        modulator = msnp.exp(- self.gamma * target * output
                             - self.gamma * msnp.log1p(1 + torch.exp(-1.0 * output)))

        loss = modulator * BCLoss
        if 'weights' in kwargs:
            weighted_loss = kwargs['weights'] * loss
        else:
            weighted_loss = self.alpha * loss
        focal_loss = msnp.sum(weighted_loss)

        focal_loss /= weighted_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Cell):
    r"""ClassBalancedSoftmaxLoss

    Reference:
    Cui et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR2019.
    https://arxiv.org/pdf/1901.05555.pdf

    Equation:
    effective number: (1 - beta^n) / (1 - beta)
    weight = 1 / effective number

    Insights: authors think that samples are not individual, they can cover with each other(for some probs),
    so the sample number number of a class can not stand for the real effect number, with the number of samples increase,
    the prob of covers increase.

    This method can use with softmax cross entropy, sigmoid cross entropy and focal loss.

    TODO:
    Experiment Results:(not sure)
    CIFAR100-LT100  (beta=0.9)
    Method      top-1 acc
    =====================
    CB-focal    38.77%
    CB-sigmoid  38.73%
    CB-softmax  38.73%
    =====================
    """

    def __init__(self, para_dict=None, loss_type='sigmoid', betas=0.9, **kwargs):
        super(ClassBalancedLoss, self).__init__()
        self.s = 1
        # self.device = para_dict["device"]
        self.num_classes = para_dict["num_classes"]
        self.num_class_list = para_dict["num_class_list"]

        effective_num = 1.0 - np.power(betas, self.num_class_list)
        weights = (1.0 - betas) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.num_classes
        self.weights = ms.Tensor(weights)
        self.one_hot = ops.OneHot()
        self.unsqueeze0 = ops.ExpandDims(0)
        self.unsqueeze1 = ops.ExpandDims(1)
        self.binary_cross_entropy_with_logits = ops.BinaryCrossEntropy()

        assert loss_type in ['focal', 'sigmoid', 'softmax'], '[ERROR] loss type error!!'
        self.loss_type = loss_type
        if loss_type == "focal":
            self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 1.
            self.focal_loss = FocalLoss(para_dict=para_dict, gamma=self.gamma)

    def construct(self, output, target):
        target_onehot = F.one_hot(target, self.num_classes)

        weights = self.unsqueeze0(self.weights) * target_onehot
        weights = weights.sum(1)
        weights = self.unsqueeze1(weights)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(output=output, target=target_onehot, weights=weights)
        elif self.loss_type == "sigmoid":
            cb_loss = self.binary_cross_entropy_with_logits(output, target_onehot)
        elif self.loss_type == "softmax":
            pred = output.softmax(dim=1)
            cb_loss = self.binary_cross_entropy(input=pred, target=target_onehot, weight=weights, reduction='sum') / \
                      weights.sum()
        return cb_loss


class CSCE(nn.Cell):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        # self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None
        self.s = cfg.LOSS.SCALE
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = ms.Tensor(per_cls_weights)

    def reset_epoch(self, epoch):
        idx = (epoch - 1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def construct(self, x, target, **kwargs):
        return self.cross_entropy(self.s * x, target)


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
class LDAMLoss(nn.Cell):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        # self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = ms.Tensor(m_list)
        self.m_list = m_list
        assert s > 0

        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch - 1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = ms.Tensor(per_cls_weights[np.newaxis, :])
        self.cast = ops.Cast()
        self.one_hot = ops.OneHot()
        self.matmul = ops.MatMul(transpose_b=True)
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def construct(self, x, target):
        index_float = self.one_hot(target)
        index = self.cast(index_float, ms.int32)
        batch_m = self.matmul(self.m_list, index_float)
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = msnp.where(index, x_m, x)
        return self.cross_entropy(self.s * output, target)


class BSCE(nn.Cell):
    r"""Balanced Softmax Cross Entropy loss

    Reference:
    Balanced Meta-Softmax for Long-Tailed Visual Recognition
    https://arxiv.org/abs/2007.10740

    Equation:
    p_j = n_j * exp(x_j) / sigma(n_i * exp(x_i))
    loss = - 1 / N * sigma(sigma(y_ij * log p_ij))

    p_j: probablity of class j
    n_j: number of class j samples in trainset(effective number is better after test in my experiments)

    Notes:
    The proof is in the supplementary of this paper.
    Also, there is a Meta-Sampler in this paper, which use a meta set to choose the best smapler each epoch.

    Experiment Results(use effective number):
    CIFAR100-LT100
    top-1 acc (beta=0.9999)
    ==========
    42.5%     (40.24 beta=0.99) (41.36 beta=0.999)
    44.69% (+mixup)
    ==========
    This method produce the best results yet, and doesn't have too much hyper-parameters.
    """

    def __init__(self, para_dict=None):
        super(BSCE, self).__init__()
        cfg = para_dict["cfg"]
        self.num_classes = para_dict["num_classes"]
        # self.device = para_dict["device"]
        # para_dict["num_class_list"] = [max(20, item) for item in para_dict["num_class_list"]]
        # print('weight = ', para_dict)
        self.num_class_list = para_dict["num_class_list"]

        self.weight = ms.Tensor(para_dict["num_class_list"])

        self.weight = self.weight ** cfg.LOSS.BSCE.tau

        self.s = cfg.LOSS.SCALE
        self.nll_loss = ops.NLLLoss()

    def construct(self, output, target, reduction=True):
        logits = output * self.s

        max_logit = logits.max(axis=1, keepdims=True)[0]
        # max_logit = max_logit.detach()
        logits -= max_logit
        exp_logits = msnp.exp(logits) * self.weight.view(-1, self.weight.shape[0])

        prob = msnp.log(msnp.clamp(exp_logits / exp_logits.sum(1, keepdim=True), min=1e-5, max=1.))
        # if reduction is True:
        loss = self.nll_loss(prob, target)
        return loss


class CDT(nn.Cell):
    r"""Class-Dependent Temperatures

    Reference:
    Ye et al.Identifying and Compensating for Feature Deviation in Imbalanced Deep Learning.
    https://arxiv.org/pdf/2001.01385.pdf

    Equation:
    p_j = exp(x_j / a_j) / sigma(x_i / a_i)
    loss = - 1 / N * sigma(sigma(y_ij * log p_ij))

    p_j: probablity of class j
    a_j: a function relate to number of class j samples in trainset((n_max/n_j)^\gamma in paper)

    Notes:
    There is a Classifier normalization with class sizes in the original paper too.
    Like /tau norm, weight_j = weight_j / (N_j)^/tau, where N_j is the number of class j samples in trainset.

    Experiment Results:
    CIFAR100-LT100
    gamma   top-1 acc
    =================
    (0.0    38.17%)
    0.05    41.45%
    0.1     41.52%
    0.15    42.01% -> best
    0.2     39.80%
    =================

    ImageNet-LT
    0.15    37.02%
    """

    def __init__(self, para_dict=None):
        super(CDT, self).__init__()
        self.num_classes = para_dict["num_classes"]
        self.device = para_dict["device"]
        self.num_class_list = para_dict["num_class_list"]

        self.gamma = 0.15
        max_n = max(self.num_class_list)
        self.weight = [((max_n / item) ** self.gamma) for item in self.num_class_list]
        self.weight = ms.Tensor(self.weight)
        self.cross_entropy = ops.SoftmaxCrossEntropyWithLogits()

    def forward(self, output, target):
        output = output / self.weight.view(-1, self.weight.shape[0])
        loss = self.cross_entropy(output, target)
        return loss


class SEQL(nn.Cell):
    """Softmax Equalization Loss

    Reference:
    Tan et al. Equalization Loss for Long-Tailed Object Recognition. CVPR2020.
    https://arxiv.org/abs/2003.05176v2

    Equation:
    loss = - 1 / N * sigma(sigma(y_ij * log p_ij))
    p_j = exp(x_j) / sigma(w_k * exp(x_k)
    w_k = 1 - beta * T_lamda(N_k) * (1 - y_k)

    beta: ∈{0, 1}, randomly chose from {0, 1} with prob gamma
    N_k: the number of class j samples
    T_lamda: a function mapping N_k to {0, 1}, formly if N_k is samller than a threshold(mean it's a tail class) than
    mapping to 1

    Notes(Motivations):
        The method is a varity of Equalization Loss in the original paper to tackle long-tail problem, which call
    Softmax Equalization Loss.
        The authors think that the gradients for tail categories when optimize the head categories are bad to those tail
    categories, so they simply ignore these gradients randomly.

    Experiment Results
    CIFAR100-LT100
    gamma   threshold   top-1 acc
    =============================
    0.95    3e-3        39.94%
    0.9     5e-3        39.81%
    =============================
    """

    def __init__(self, para_dict=None):
        super(SEQL, self).__init__()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.f = np.array(para_dict['num_class_list'])
        self.threshold = 3e-3
        self.t_lambda = (self.f / np.sum(self.f)) < self.threshold  # < threshold ---> 1;  > threshold ---> 0
        self.t_lambda = ms.Tensor(self.t_lambda[np.newaxis, :])
        self.beta_prob = 0.95
        self.unsqueeze = ops.ExpandDims()
        self.nll_loss = ops.NLLLoss()
        # print(self.t_lambda)

    def get_log_softmax_w(self, logits, w):
        max_logit = logits.max(axis=1, keepdims=True)[0]
        max_logit = max_logit.detach()
        logits -= max_logit
        log_prob = logits - msnp.log((w * msnp.exp(logits)).sum(axis=1, keepdims=True) + 1e-5)
        return log_prob

    def construct(self, output, target):
        output = output
        target_onehot = msnp.zeros(output.size()).scatter_(1, self.unsqueeze(target, 1), 1)
        beta = (ops.uniform(output.size(), 0, 1) < self.beta_prob)

        w = 1 - beta * self.t_lambda * (1 - target_onehot)

        # print(w)
        # exit()
        log_prob = self.get_log_softmax_w(output, w)
        # print(log_prob)
        # loss = -(target_onehot * log_prob).sum() / target.shape[0]
        loss = self.nll_loss(log_prob, target)
        # print(loss)
        return loss
