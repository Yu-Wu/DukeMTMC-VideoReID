from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class TripletClusteringLoss(nn.Module):
    def __init__(self, clusters, margin=0,):
        super(TripletClusteringLoss, self).__init__()
        assert isinstance(clusters, torch.autograd.Variable)
        self.clusters = clusters
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.num_classes = clusters.size(0)
        self.num_features = clusters.size(1)
        self.dist = torch.pow(self.clusters, 2).sum(dim=1, keepdim=True)

    def forward(self, inputs, targets):
        assert self.num_features == input.size(1)
        n = inputs.size(0)
        dist = self.dist.expand(self.num_classes, n)
        dist += torch.pow(inputs, 2).sum(dim=1).t()
        dist.addmm_(1, -2, self.clusters, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        dist = dist.t()
        # For each anchor, find the hardest positive and negative
        mask = torch.zeros(n,self.num_classes,out=torch.ByteTensor())
        target_ids = targets.data.numpy().astype(int)
        mask[np.arange(n),target_ids] = 1
        dist_ap = dist[mask == 1]
        dist_an = dist[mask == 0].view(n, -1).min(dim=1)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

    def update_clusters(self,clusters):
        self.clusters = clusters
