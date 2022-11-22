# -*- encoding: utf-8 -*-
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=138, classes=2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.Sigmoid()
        self.linear3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.act3 = nn.Sigmoid()
        self.linear4 = nn.Linear(512, 1024)
        self.act4 = nn.ReLU()
        self.linear5 = nn.Linear(512, classes)
        self.act5 = nn.ReLU()
        self.sfm = nn.Softmax(1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.act3(out)
        # out = self.linear4(out)
        # out = self.act4(out)
        out = self.linear5(out)
        # out = self.act5(out)
        # result = self.sfm(out)

        return out


class MLPFeature(nn.Module):
    def __init__(self):
        super(MLPFeature, self).__init__()
        self.linear1 = nn.Linear(14, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(128, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 512, bias=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.act2(out)
        out = self.linear3(out)
        out = self.act3(out)

        return out