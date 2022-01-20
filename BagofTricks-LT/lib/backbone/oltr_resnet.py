import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='same',
                     has_bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='same',
                               has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, use_modulatedatt=False, use_fc=False, dropout=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='same',
                               has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_fc = use_fc
        self.use_dropout = True if dropout else False

        if self.use_fc:
            self.fc_add = nn.Dense(512 * block.expansion, 512)

        if self.use_dropout:
            self.dropout = nn.Dropout(keep_prob=dropout)

        self.use_modulatedatt = use_modulatedatt
        if self.use_modulatedatt:
            self.modulatedatt = ModulatedAttLayer(in_channels=512 * block.expansion)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer1':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]
        out = self.layer2(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer2':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]
        out = self.layer3(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer3':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]
        out = self.layer4(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer4':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]

        return out


def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = load_checkpoint(weights_path)
    # if not classifier:
    #     if caffe:
    #         weights = {k: weights[k] if k in weights else model.state_dict()[k]
    #                    for k in model.state_dict()}
    #     else:
    #         weights = weights['state_dict_best']['feat_model']
    #         weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
    #                    for k in model.state_dict()}
    # else:
    #     weights = weights['state_dict_best']['classifier']
    #     weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
    #                for k in model.state_dict()}
    load_param_into_net(model, weights)
    return model


def res10(cfg, use_selfatt=False, use_fc=False, stage1_weights=False, dataset=None, test=False):
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(BasicBlock, [1, 1, 1, 1], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)
    if not test:
        if stage1_weights:
            assert (dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            resnet10 = init_weights(model=resnet10,
                                    weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet10
