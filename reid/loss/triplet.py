from __future__ import absolute_import

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


# class TripletLoss(nn.Module):
#     def __init__(self, margin=0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#
#     def forward(self, inputs, targets):
#         n = inputs.size(0)
#         print(n)
#         # Compute pairwise distance, replace by the official when merged
#         dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         dist.addmm_(1, -2, inputs, inputs.t())
#         dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             #print(dist[i][mask[i]].max().size())
#             dist_ap.append(dist[i][mask[i]].max())
#             dist_an.append(dist[i][mask[i] == 0].min())
#         dist_ap = torch.stack(dist_ap)
#
#         dist_an = torch.stack(dist_an)
#         y = torch.ones_like(dist_an)
#         # Compute ranking hinge loss
#
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#
#         print('prec',dist_an.data > dist_ap.data.sum()*1.)
#         p=input()
#
#         prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
#         return loss, prec

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)


    def forward(self, inputs, targets):

        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)    # all postive
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)   #   max(0, p-n+margin)

        prec = (dist_an.data > dist_ap.data).sum().float() * 1. / y.size(0)

        return loss, prec
