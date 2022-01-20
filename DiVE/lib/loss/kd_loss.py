import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.numpy as msnp
from .longtail_loss import *
from .general_loss import *


class DistillKL(nn.Cell):
    """
    Knowledge Distillation

    Equation:
    loss = alpha * KL(logits, logits_t) + beta * CrossEntropy(logits, label)
    """

    def __init__(self, para_dict=None):
        super(DistillKL, self).__init__()

        cfg = para_dict["cfg"]
        self.cfg = cfg
        self.s = cfg.LOSS.SCALE
        self.t = cfg.LOSS.DISTILLKL.temperature
        self.alpha = cfg.LOSS.DISTILLKL.alpha
        self.refine = cfg.LOSS.DISTILLKL.refine

        self.criterion = eval(cfg.LOSS.DISTILLKL.criterion)(para_dict)
        self.log_softmax = ops.LogSoftmax(1)
        self.softmax = ops.Softmax(1)
        self.unsqueeze = ops.ExpandDims()
        self.kl_div = ops.KLDivLoss('sum')

        # self.device = para_dict["device"]
        self.num_classes = para_dict["num_classes"]
        self.num_class_list = para_dict["num_class_list"]

    def construct(self, logits, logits_t, label):
        """
        :param logits: shape of (batch_size, n_class)
        :param logits_t: shape of (batch_size, n_class)
        :param label: shape of (batch_size, )
        :return: loss: shape of (1, )
        """
        t = self.t
        s = self.s
        alpha = self.alpha

        # ce_loss = F.cross_entropy(logits, label)
        ce_loss = self.criterion(logits, label)

        if self.cfg.CLASSIFIER.TYPE == 'FCNorm' and self.cfg.TEACHER.CLASSIFIER.TYPE == 'FC':
            logits *= 30

        p_s = self.log_softmax(logits * s / t)
        # p_s = self.balance_log_softmax(logits / t, t)
        p_t = self.softmax(logits_t * s / t)  # / self.weight.view(1,-1)

        # refine mistakes in teacher
        if self.refine:
            label_onehot = msnp.zeros(logits.size()).scatter_(1, label.unsqueeze(1).data.cpu(), 1).to(self.device)
            p_t = p_t * 0.5 + label_onehot * 0.5

        if self.cfg.LOSS.DISTILLKL.sqrt_norm is True:
            # p_t_org = p_t.clone().detach()
            if self.cfg.LOSS.DISTILLKL.power == 0.5:
                p_t = msnp.sqrt(p_t)
                p_t /= p_t.sum(1, keepdim=True)
            else:
                p_t = p_t ** self.cfg.LOSS.DISTILLKL.power
                p_t /= p_t.sum(1, keepdim=True)

            # p_t = p_t * (self.head[label].view(-1,1)) + p_t_org * ((1 - self.head[label]).view(-1,1))

        p_t /= p_t.sum(dim=1, keepdim=True)
        # p_t = p_t.detach()

        # kl_loss = ((F.kl_div(p_s, p_t, reduction='none').sum(1) * (p_t.argmax(1)==label)) * (t ** 2)).sum() / (p_t.argmax(1)==label).sum()
        kl_loss = self.kl_div(p_s, p_t) * (t ** 2) / logits.shape[0]

        loss = alpha * kl_loss + (1 - alpha) * ce_loss
        return loss


class PKTCosSim(nn.Cell):
    r"""Learning Deep Representations with Probabilistic Knowledge Transfer

    Reference:
        Nikolaos Passalis & Anastasios Tefas. Learning Deep Representations with Probabilistic Knowledge Transfer. ECCV 2018.
        https://arxiv.org/pdf/1803.10837.pdf
        https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py
        https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/pkt.py

    Equation:
        kcos_{a,b} = (cos(a,b) + 1) / 2
        feat_prob_{i,j}^t = kcos(feat_t_{i}, feat_t_{j}) / \sum_{k} kcos(feat_t_{i}, feat_t_{j})
        feat_prob_{i,j}^s = kcos(feat_s_{i}, feat_s_{j}) / \sum_{k} kcos(feat_s_{i}, feat_s_{j})
        loss = \sum_{i} KL(feat_prob_t_{i}, feat_prob_s_{i})

        i,j,k is the sample id

    Note:
        PKTCosSim are usually too small, multiply a large weight on it.
    Experimental Results:
        on CIFAR100-LT100, Loss = BSCE + PKTCosSim * 100000

    """

    def __init__(self, para_dict=None):
        super(PKTCosSim, self).__init__()
        self.para_dict = para_dict
        self.num_classes = para_dict["num_classes"]
        # self.device = para_dict['device']
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, feat_s, feat_t, eps=1e-6):
        r"""
        Args:
            inputs:
            feat_s: student feature (N, C)
            feat_t: teacher feature (N, C)
            output:
            KL divergence
        """
        # Normalize each vector by its norm
        feat_s_norm = msnp.sqrt(msnp.sum(feat_s ** 2, axis=1, keepdims=True))
        feat_s = feat_s / (feat_s_norm + eps)
        feat_s[feat_s != feat_s] = 0

        feat_t_norm = msnp.sqrt(msnp.sum(feat_t ** 2, axis=1, keepdims=True))
        feat_t = feat_t / (feat_t_norm + eps)
        feat_t[feat_t != feat_t] = 0

        # Calculate the cosine similarity
        feat_s_cos_sim = self.matmul(feat_s, feat_s.transpose(0, 1))
        feat_t_cos_sim = self.matmul(feat_t, feat_t.transpose(0, 1))

        # Scale cosine similarity to [0,1]
        feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
        feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

        # Transform them into probabilities
        feat_s_cond_prob = feat_s_cos_sim / msnp.sum(feat_s_cos_sim, axis=1, keepdims=True)
        feat_t_cond_prob = feat_t_cos_sim / msnp.sum(feat_t_cos_sim, axis=1, keepdims=True)

        # Calculate the KL-divergence
        loss = msnp.mean(feat_t_cond_prob * msnp.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps)))

        return loss


class RKD(nn.Cell):
    r"""
    Reference:
        Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho. Relational Knowledge Distillation. CVPR2019.
        https://arxiv.org/pdf/1904.05068.pdf
        https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/rkd.py

    contain distance relation & angle relation
    """

    def __init__(self, para_dict=None):
        super(RKD, self).__init__()
        # self.para_dict = para_dict
        # set default
        self.w_dist = 25.0
        self.w_angle = 50.0
        self.smooth_l1_loss = ops.SmoothL1Loss()
        self.normalize = ops.L2Normalize(2)
        self.bmm = ops.BatchMatMul(transpose_b=True)
        self.mm = ops.MatMul(transpose_b=True)
        self.unsqueeze0 = ops.ExpandDims(0)
        self.unsqueeze1 = ops.ExpandDims(1)

    def construct(self, feat_s, feat_t):
        loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
               self.w_angle * self.rkd_angle(feat_s, feat_t)

        return loss

    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = self.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
        norm_feat_t_vd = self.normalize(feat_t_vd)
        feat_t_angle = self.bmm(norm_feat_t_vd, norm_feat_t_vd).view(-1)

        feat_s_vd = (self.unsqueeze0(feat_s) - self.unsqueeze1(feat_s))
        norm_feat_s_vd = self.normalize(feat_s_vd)
        feat_s_angle = self.bmm(norm_feat_s_vd, norm_feat_s_vd).view(-1)

        loss = self.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        # """
        # Equation:
        #     dist(i, j) = ||feat_i - feat_j||_2
        # Args:
        #     Inputs:
        #     - feat: (N,C)
        #     - squared: whether not do sqrt
        #     targets:
        #     - feature distance matrix
        # """
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = self.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod)

        if not squared:
            feat_dist = msnp.sqrt(feat_dist)

        # feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist
