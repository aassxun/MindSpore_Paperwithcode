import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore.ops import operations as P

# for LDAM Loss
class FCNorm(nn.Cell):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = ms.Parameter(ms.Tensor(num_classes, num_features))
        self.normalize = ops.L2Normalize(1)
        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x):
        out = self.matmul(self.normalize(x), self.normalize(self.weight))
        return out


class LWS(nn.Cell):

    def __init__(self, num_features, num_classes, bias=True):
        super(LWS, self).__init__()
        self.fc = nn.Dense(num_features, num_classes, has_bias=bias)
        self.scales = ms.Parameter(msnp.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def construct(self, x):
        x = self.fc(x)
        x *= self.scales
        return x
