import cv2
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import context
import mindspore.ops as ops
from mindspore.ops import functional as F
from resnet import resnet50

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def conv3x3(in_channel, out_channel, stride=1, use_se=False):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def conv1x1(in_channel, out_channel, stride=1, use_se=False):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)

def _fc(in_channel, out_channel, use_se=False):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0, pad_mode='same')
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.SequentialCell(layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def construct(self, x):
        return self._forward_impl(x)
#

def A_2_net_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    return model


class ResNet_Refine(nn.Cell):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        # self.avgpool = ops.AdaptiveAvgPool2D((1, 1))
        self.avgpool = ops.AvgPool(pad_mode="VALID", kernel_size=7, strides=7)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.SequentialCell(layers)

    def _forward_impl(self, x):
        x = self.layer4(x)
        pool_x = self.avgpool(x)
        pool_x = pool_x.flatten()
        # pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def construct(self, x):
        return self._forward_impl(x)


def A_2_net_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    return model


class ResNet_Attention(nn.Cell):

    def __init__(self, block, layer, att_size=4, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_Attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.att_size = att_size
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layer, stride=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        att_expansion = 0.25
        layers = []
        layers.append(block(self.inplanes, int(self.inplanes * att_expansion), stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(nn.SequentialCell(
                conv1x1(self.inplanes, int(self.inplanes * att_expansion)),
                nn.BatchNorm2d(int(self.inplanes * att_expansion))
            ))
            self.inplanes = int(self.inplanes * att_expansion)
            layers.append(block(self.inplanes, int(self.inplanes * att_expansion), groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        layers.append(nn.SequentialCell(
            conv1x1(self.inplanes, self.att_size),
            nn.BatchNorm2d(self.att_size),
            nn.ReLU()
        ))
        return nn.SequentialCell(layers)

    def _forward_impl(self, x):
        x = self.layer4(x)
        return x

    def construct(self, x):
        return self._forward_impl(x)


def A_2_net_attention(att_size=4, pretrained=False, progress=True, **kwargs):
    model = ResNet_Attention(Bottleneck, 3, att_size=att_size, **kwargs)
    return model


class A_2_net(nn.Cell):
    def __init__(self, code_length=12, num_classes=200, att_size=4, feat_size=2048, pretrained=False,
                 finetune=False):
        super(A_2_net, self).__init__()
        self.backbone = A_2_net_backbone(pretrained=pretrained)
        self.refine_global = A_2_net_refine(is_local=False, pretrained=pretrained)
        self.refine_local = A_2_net_refine(pretrained=pretrained)
        self.attention = A_2_net_attention(att_size)

        self.finetune = finetune
        self.hash_layer_active = nn.Tanh()
        self.unsqueeze = ops.ExpandDims()
        self.mul = ops.Mul()
        self.normlize = ops.L2Normalize()
        self.concat = ops.Concat(1)

        self.linear = ops.MatMul()
        self.W = ms.Parameter(Tensor(np.random.uniform(-1, 1, (code_length, (att_size + 1) * feat_size)), ms.float32), name="w", requires_grad=True)

    def construct(self, x):
        out = self.backbone(x)
        batch_size, channels, h, w = out.shape
        global_f = self.refine_global(out)
        att_map = self.attention(out)
        att_size = att_map.shape[1]
        att_map_rep = self.unsqueeze(att_map, 2)
        att_map_rep = att_map_rep.repeat(channels, axis=2)
        out_rep = self.unsqueeze(out, 1)
        out_rep = out_rep.repeat(att_size, axis=1)

        out_local = self.mul(att_map_rep, out_rep).view(batch_size * att_size, channels, h, w)
        local_f, avg_local_f = self.refine_local(out_local)

        _, channels, h, w = local_f.shape
        local_f = local_f.view(batch_size, att_size, channels, h, w)
        avg_local_f = avg_local_f.view(batch_size, att_size, channels)

        global_f = global_f.view(batch_size, channels)
        avg_local_f = self.normlize(avg_local_f)
        global_f = self.normlize(global_f)

        all_f = self.concat((avg_local_f.view(batch_size, -1), global_f))
        deep_S = self.linear(all_f, self.W.T)
        binary_like_code = self.hash_layer_active(deep_S)
        if self.finetune:
            after_f = self.linear(binary_like_code, self.W)
            return binary_like_code, all_f, after_f
        else:
            return binary_like_code

def a_2_net(code_length, num_classes, att_size, feat_size, pretrained=False, finetune = False, progress=True, **kwargs):
    model = A_2_net(code_length, num_classes, att_size, feat_size, pretrained, finetune, **kwargs)
    return model

if __name__ == '__main__':
    device = 'CPU'
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
    x = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
    x = ms.Tensor(x)
    net = a_2_net(code_length=12, num_classes=200, att_size=4, feat_size=2048, finetune=False)
    y = net(x)
