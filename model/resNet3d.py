# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
# from torchviz import make_dot
import torchvision


def make_layer(block, num_layers, in_channel):
    layers_list = []
    strides = [[2, 1], [1, 1]]
#     is_first_layer = False
    for i in range(num_layers):
        for j, stride in enumerate(strides):
            is_first_layer = (i == 0 and j == 0)
            out_channel = (in_channel * 2 if j == 0 else in_channel)
            layers_list.append(block(in_channel, out_channel, stride, is_first_layer))
            # print(in_channel, out_channel)
            if not is_first_layer:
                in_channel = out_channel
    return nn.Sequential(*layers_list)


class ResNet3d(nn.Module):
    def __init__(self, block, num_layer, classes):
        super(ResNet3d, self).__init__()
        out_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channel, kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.MaxPool3d([2, 2, 2])
        )
        # out_channel = out_channel // 4
#         self.first_layer = None
#         resblock = ResBlock()
        self.resblock = make_layer(block, num_layer, out_channel)
        out_channel = out_channel * (2 ** (num_layer - 1))
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        # self.avg_pool = nn.AvgPool3d()
        # self.flatten = None
        self.linear = nn.Linear(512, classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out = self.bn(out)
        out = self.relu(out)
        out = F.avg_pool3d(out, (out.shape[2], out.shape[3], out.shape[4]))
        out = out.view(out.size(0), -1)
        # out = self.avg_pool((out.shape[2], out.shape[3], out.shape[4]))
        out = self.dropout(out)
        out = self.linear(out)
        # out = F.softmax(out, dim=1)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, is_first_layer=False):
        super(ResBlock, self).__init__()
        if is_first_layer:
            stride = [1, 1]
            self.left = nn.Sequential(
                nn.Conv3d(in_channel, in_channel, kernel_size=[3, 3, 3], stride=[stride[0], stride[0], stride[0]],
                          padding=[1, 1, 1], bias=True),
                nn.BatchNorm3d(in_channel),
                nn.ReLU(),
                nn.Conv3d(in_channel, in_channel, kernel_size=[3, 3, 3], stride=[stride[1], stride[1], stride[1]],
                          padding=[1, 1, 1], bias=True)
            )
        else:
            self.left = nn.Sequential(
                nn.BatchNorm3d(in_channel),
                nn.ReLU(),
                nn.Conv3d(in_channel, out_channel, kernel_size=[3, 3, 3], stride=[stride[0], stride[0], stride[0]],
                          padding=[1, 1, 1], bias=True),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
                nn.Conv3d(out_channel, out_channel, kernel_size=[3, 3, 3], stride=[stride[1], stride[1], stride[1]],
                          padding=[1, 1, 1], bias=True)
            )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel and not is_first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=[1, 1, 1], stride=[stride[0], stride[0], stride[0]],
                          padding=0, bias=True)
            )

    def forward(self, x):
        out = self.left(x)
        out = self.shortcut(x) + out
        return out


class ResNet3dFeature(nn.Module):
    def __init__(self, block, num_layer, is_pooling=True):
        super(ResNet3dFeature, self).__init__()
        out_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channel, kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.MaxPool3d([2, 2, 2])
        )
        # out_channel = out_channel // 4
#         self.first_layer = None
#         resblock = ResBlock()
        self.resblock = make_layer(block, num_layer, out_channel)
        out_channel = out_channel * (2 ** (num_layer - 1))
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        self.output_dim = out_channel
        self.is_pooling = is_pooling
        # self.avg_pool = nn.AvgPool3d()
        # self.flatten = None
        # self.linear = nn.Linear(512, classes)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out = self.bn(out)
        out = self.relu(out)
        if self.is_pooling:
            out = F.avg_pool3d(out, (out.shape[2], out.shape[3], out.shape[4]))
            out = out.view(out.size(0), -1)

        return out


class ResNet3dFeatureV2(nn.Module):
    def __init__(self, block, num_layer):
        super(ResNet3dFeatureV2, self).__init__()
        out_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channel, kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.MaxPool3d([2, 2, 2])
        )
        # out_channel = out_channel // 4
#         self.first_layer = None
#         resblock = ResBlock()
        self.resblock = make_layer(block, num_layer, out_channel)
        out_channel = out_channel * (2 ** (num_layer - 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel // 2)
        self.bn3 = nn.BatchNorm3d(out_channel // 4)
        self.relu = nn.ReLU()
        self.output_dim = out_channel
        # self.avg_pool = nn.AvgPool3d()
        # self.flatten = None
        # self.linear = nn.Linear(512, classes)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out1 = self.conv1(x)
        # out_block = self.resblock(out1)
        out_list = list()
        for i in range(len(self.resblock)):
            out1 = self.resblock[i](out1)
            out_list.append(out1)
        out_list2 = list()
        for i in range(3):
            # out = self.bn(out_list[-(i * 2) - 1])
            out = eval(f"self.bn{i+1}(out_list[-(i * 2) - 1])")
            out = self.relu(out)
            out = F.avg_pool3d(out, (out.shape[2], out.shape[3], out.shape[4]))
            out = out.view(out.size(0), -1)
            out_list2.append(out)
        return out_list2[0], out_list2[1], out_list2[2]


if __name__ == '__main__':
    # model = torchvision.models.resnet18(pretrained=False).to(torch.device('cuda:0'))
    model = ResNet3dFeatureV2(ResBlock, 3).to(torch.device('cuda:0'))
    inp = torch.zeros((2, 1, 64, 128, 128)).to(torch.device('cuda:0'))
    output = model(inp)

    # summary(model, (3,  28, 28))
# device = torch.device("cuda:0")
# # model1 = ResBlock(64, 128, [2, 1], is_first_layer=False).to(device)
# model1 = ResNet3d(ResBlock, 4).to(device)
# x = torch.rand((1, 1, 64, 128, 128)).to(device)
# y = model1(x)
# g = make_dot(y)
# g.render('res_model.pdf', view=False)
# g = hl.build_graph(model1, x, transforms=None)
# g.save('network_architecture.pdf')