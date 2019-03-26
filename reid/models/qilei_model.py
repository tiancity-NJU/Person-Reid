import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50
from torch.nn import init


class STModel(nn.Module):
  def __init__(self, last_conv_stride=2, cut_at_pooling=False,
                 num_features=1024, norm=False, dropout=0.5, num_classes=0, triplet_features=128, num_cam=0):
    super(STModel, self).__init__()
    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)

    self.cut_at_pooling = cut_at_pooling

    if not self.cut_at_pooling:
      self.num_features = num_features
      self.norm = norm
      self.dropout = dropout
      self.has_embedding = num_features > 0
      self.num_classes = num_classes
      self.triplet_features = triplet_features
      self.cam_features = num_cam > 0
      #out_planes = self.base.fc.in_features

      # Append new layers
      if self.has_embedding:
        self.feat = nn.Linear(2048, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
      else:
        # Change the num_features to CNN output channels
        self.num_features = 2048
      if self.dropout > 0:
        self.drop = nn.Dropout(self.dropout)
      if self.num_classes > 0:
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)
      if self.triplet_features > 0:
        self.triplet = nn.Linear(self.num_features, self.triplet_features)
        init.normal(self.triplet.weight, std=0.001)
        init.constant(self.triplet.bias, 0)

      if self.cam_features > 0:
        self.cam_fc = nn.Linear(1024, 128)
        self.cam_bn = nn.BatchNorm1d(128)
        self.cam_drop = nn.Dropout(0.5)
        init.kaiming_normal(self.cam_fc.weight, mode='fan_out')
        init.constant(self.cam_fc.bias, 0)
        init.constant(self.cam_fc.weight, 1)
        init.constant(self.cam_fc.bias, 0)

        self.cam_classifier = nn.Linear(128, num_cam)
        init.normal(self.cam_classifier.weight, std=0.001)
        init.constant(self.cam_classifier.bias, 0)



  def forward(self, x, output_feature=None):
    # shape [N, C, H, W]
    x = self.base(x)

    if self.cut_at_pooling:
      return x

    x = F.avg_pool2d(x, x.size()[2:])
    # shape [N, C]
    x = x.view(x.size(0), -1)


    if output_feature == 'pool5':
      x = F.normalize(x)      # resize to (0,1)
      return x
    if self.has_embedding:
      x = self.feat(x)
      x = self.feat_bn(x)


    if self.cam_features > 0:
      x_cam = self.cam_fc(x)
      x_cam = self.cam_bn(x_cam)
      x_cam = self.cam_drop(F.relu(x_cam))
      x_cam = self.cam_classifier(x_cam)

#####
    if self.norm:
      x = F.normalize(x)
    elif self.has_embedding:
      x = F.relu(x)
    # triplet feature
    if self.triplet_features > 0:
      x_triplet = self.triplet(x)

    if self.dropout > 0:
      x = self.drop(x)
    if self.num_classes > 0:
      x_class = self.classifier(x)
    # three outputs
    if self.cam_features > 0:
      return x_class, x_triplet, x_cam   #   1024->PID softmax    1024->128->cam softmax  1024->128->triplet loss
    # two outputs
    if self.triplet_features > 0:
      return x_class, x_triplet


    return x_class
