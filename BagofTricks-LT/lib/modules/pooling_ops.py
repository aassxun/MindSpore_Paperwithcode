import mindspore.nn as nn
import mindspore.ops as ops


class GAP(nn.Cell):
    """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """

    def __init__(self):
        super(GAP, self).__init__()
        self.mean = ops.ReduceMean(True)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x
