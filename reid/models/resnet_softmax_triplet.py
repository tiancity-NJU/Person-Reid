from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


class ResNet_s_t(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=2048, dropout=0.1, num_classes=0):
        super(ResNet_s_t, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet_s_t.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet_s_t.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features  # 2048

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                self.relu = nn.ReLU(inplace=True)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier_s = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier_s.weight, std=0.001)
                init.constant(self.classifier_s.bias, 0)


        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x_t = F.avg_pool2d(x, x.size()[2:])
        x_t = x_t.view(x_t.size(0), -1)

        if self.has_embedding:
            x_t = self.feat(x_t)
            x_t = self.feat_bn(x_t)
            x_t = self.relu(x_t)
        if self.dropout > 0:
            x_t = self.drop(x_t)
        if self.num_classes > 0:
            x_s = self.classifier_s(x_t)
            return x_t, x_s

        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18_s_t(**kwargs):
    return ResNet_s_t(18, **kwargs)


def resnet34_s_t(**kwargs):
    return ResNet_s_t(34, **kwargs)


def resnet50_s_t(**kwargs):
    return ResNet_s_t(50, **kwargs)


def resnet101_s_t(**kwargs):
    return ResNet_s_t(101, **kwargs)


def resnet152_s_t(**kwargs):
    return ResNet_s_t(152, **kwargs)
