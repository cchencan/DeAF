# -*- encoding: utf-8 -*-
"""
@File    :   fusion_1.py    
@Contact :   425077099@qq.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/3/7 9:58   C.Chen     1.0         None
"""
import torch
import torch.nn as nn
from model.resNet3d import ResNet3dFeature
from model.mlp import MLPFeature
from model.non_local_embedded_gaussian import NONLocalBlock1D, NONLocalBlock2D
from training_utils.load_pretrained_model import load_pretrained_model
from model.CMSA import MGATECell
from training_utils.loss_function import MMDLoss
import torch.nn.functional as F


# fusion 1: single linear classification
class Fusion1(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion1, self).__init__()
        # self.linear_1 = nn.Linear(in_feature, in_feature)
        # self.bn1 = nn.BatchNorm1d(in_feature)
        # self.act1 = nn.ReLU()
        # self.linear_2 = nn.Linear(in_feature, in_feature)
        # self.bn2 = nn.BatchNorm1d(in_feature)
        # self.act2 = nn.ReLU()
        # self.linear_3 = nn.Linear(in_feature, num_class)
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

    def forward(self, x_ct, x_tb):
        out_ct = self.ct_model(x_ct)
        out_tb = self.tb_model(x_tb)
        out = torch.cat((out_ct, out_tb), 1)

        # out = self.linear_1(out)
        # out = self.bn1(out)
        # out = self.act1(out)
        # out = self.linear_2(out)
        # out = self.bn2(out)
        # out = self.act2(out)
        # out = self.dropout(out)
        # out = self.linear_3(out)
        out = self.fusion_layer(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out


# fusion 2: conv1d
class Fusion2(nn.Module):
    def __init__(self, num_class):
        super(Fusion2, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, 64, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear(out)
        out = self.dropout(out)

        return out


class Fusion3(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model_mlp, tb_model_conv=None):
        super(Fusion3, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)
        self.nl1 = NONLocalBlock1D(32)
        # self.nl2 = NONLocalBlock2D(in_feature)
        self.ct_model = ct_model
        self.tb_model_mlp = tb_model_mlp
        self.tb_model_conv = tb_model_conv
        self.conv1d_1 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            nn.Sigmoid()
        )

    def forward(self, x_ct, x_tb1, x_tb2):
        out_ct = self.ct_model(x_ct)

        out_ct = self.projection_layer(out_ct)

        out_tb2 = self.tb_model_mlp(x_tb2)
        if self.tb_model_conv is not None:
            out_tb1 = self.tb_model_conv(x_tb1)
            out = torch.cat((out_ct, out_tb1, out_tb2), 1)
        else:
            out = torch.cat((out_ct, out_tb2), 1)
        out = out.view(out.shape[0], 1, -1)
        out = self.conv1d_1(out)
        out = self.nl1(out)
        out = self.conv1d_2(out)
        out = out.view(out.shape[0], -1)
        # out = self.nl2(out)
        out = self.fusion_layer(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out


# 14-d CT features and 14-d clinical data
class Fusion4(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model, fc):
        super(Fusion4, self).__init__()
        # in_feature = in_feature // 2
        self.fc = fc
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            # nn.Linear(65536, 2048),
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.non_local = False
        self.nl1 = NONLocalBlock1D(32)
        self.conv1d_1 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)

        self.zero_type = 'null'

    def forward(self, x_ct, x_tb):
        # out_ct = self.ct_model(x_ct)
        # out_ct = self.projection_layer(out_ct)
        # out_tb = self.projection_layer_tb(x_tb)
        # out_tb = self.tb_model(x_tb)
        if self.zero_type == 'tb':
            out_ct = self.ct_model(x_ct)
            out_ct = self.projection_layer(out_ct)
            out = torch.cat((out_ct, torch.zeros((out_ct.shape[0], 14), device=out_ct.device)), 1)
        elif self.zero_type == 'ct':
            out_tb = self.projection_layer_tb(x_tb)
            out = torch.cat((torch.zeros((out_tb.shape[0], 14), device=out_tb.device), out_tb), 1)
        else:
            out_ct = self.ct_model(x_ct)
            out_ct = out_ct.view(out_ct.shape[0], -1)
            out_ct = self.projectionMLP(out_ct)
            out_ct = self.projection_layer(out_ct)
            if self.fc:
                out_tb = self.projection_layer_tb(x_tb)
            else:
                out_tb = x_tb
            # cat
            out = torch.cat((out_ct, out_tb), 1)

            # add
            # out = out_ct + out_tb
            # print(out.shape)

        if self.non_local:
            out = out.view(out.shape[0], 1, -1)
            out = self.conv1d_1(out)
            out = self.nl1(out)
            out = self.conv1d_2(out)
            out = out.view(out.shape[0], -1)
        out = self.fusion_layer(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out


class Fusion5(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion5, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        ct_projection_num = 14

        self.projection_layer_ct = nn.Sequential(
            nn.Linear(256, ct_projection_num),
            nn.BatchNorm1d(ct_projection_num),
            nn.Sigmoid()
        )
        self.projection_layer_fusion = nn.Sequential(
            nn.Linear(14 + ct_projection_num + 14, 28),
            nn.BatchNorm1d(28),
            nn.Sigmoid()
        )

    def forward(self, x_ct, x_tb):
        out_ct = self.ct_model(x_ct)
        out_ct = self.projection_layer_ct(out_ct)
        # out_tb = self.tb_model(x_tb)
        mid_ct_tb = torch.cat((out_ct, x_tb), 1)

        mid_x = torch.cat((mid_ct_tb, x_tb), 1)
        mid_x = self.projection_layer_fusion(mid_x)

        out = self.fusion_layer(mid_x)
        out = self.dropout(out)
        out = self.classification(out)
        return out


# Ensemble Learning
class Fusion6(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion6, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, num_class),
            nn.BatchNorm1d(num_class),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )

        self.non_local = False
        self.nl1 = NONLocalBlock1D(32)
        self.conv1d_1 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)

        # self.ensemble_weight = nn.Linear(2, 1)
        self.ensemble_weight = nn.ParameterDict(
            {
                'weight1': nn.Parameter(torch.randn(1)),
                'weight2': nn.Parameter(torch.randn(1))

            }
        )

    def forward(self, x_ct, x_tb):
        out_ct = self.ct_model(x_ct)
        out_ct = self.projection_layer(out_ct)
        out_tb = self.projection_layer_tb(x_tb)

        out_tb = out_tb * self.ensemble_weight['weight1']

        out_ct = out_ct * self.ensemble_weight['weight2']

        out = out_ct + out_tb

        return out


class Fusion7(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion7, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        ct_projection_num = 14

        self.projection_layer_ct1 = nn.Sequential(
            nn.Linear(256, ct_projection_num),
            nn.BatchNorm1d(ct_projection_num),
            nn.ReLU()
        )

        self.projection_layer_ct2 = nn.Sequential(
            nn.Linear(128, ct_projection_num),
            nn.BatchNorm1d(ct_projection_num),
            nn.ReLU()
        )

        self.projection_layer_ct3 = nn.Sequential(
            nn.Linear(64, ct_projection_num),
            nn.BatchNorm1d(ct_projection_num),
            nn.ReLU()
        )

        self.projection_layer_fusion = nn.Sequential(
            nn.Linear(14 + ct_projection_num + 14, 28),
            nn.BatchNorm1d(28),
            nn.ReLU()
        )

        self.conv1d_11 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_12 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_13 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)

        self.conv1d_2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)
        # self.conv1d_22 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)
        # self.conv1d_23 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)

        self.nl1 = NONLocalBlock1D(32)
        self.nl2 = NONLocalBlock1D(32)
        self.nl3 = NONLocalBlock1D(32)
        self.mgat = MGATECell(32)

    def forward(self, x_ct, x_tb):
        out_ct1, out_ct2, out_ct3 = self.ct_model(x_ct)

        out_ct1 = self.projection_layer_ct1(out_ct1)
        out_ct2 = self.projection_layer_ct2(out_ct2)
        out_ct3 = self.projection_layer_ct3(out_ct3)
        # out_tb = self.tb_model(x_tb)

        mid_ct_tb1 = torch.cat((out_ct1, x_tb), 1)
        mid_ct_tb2 = torch.cat((out_ct2, x_tb), 1)
        mid_ct_tb3 = torch.cat((out_ct3, x_tb), 1)

        out1 = mid_ct_tb1.view(mid_ct_tb1.shape[0], 1, -1)
        out1 = self.conv1d_11(out1)
        out1 = self.nl1(out1)

        out2 = mid_ct_tb2.view(mid_ct_tb2.shape[0], 1, -1)
        out2 = self.conv1d_12(out2)
        out2 = self.nl2(out2)

        out3 = mid_ct_tb3.view(mid_ct_tb3.shape[0], 1, -1)
        out3 = self.conv1d_13(out3)
        out3 = self.nl3(out3)

        mix_out = self.mgat(out1, out2, out3)

        mix_out = self.conv1d_2(mix_out)

        mix_out = mix_out.view(mix_out.shape[0], -1)

        out = self.fusion_layer(mix_out)
        out = self.dropout(out)
        out = self.classification(out)
        return out


# constraint for CT features by clinical data
class Fusion8(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion8, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 28),
            nn.BatchNorm1d(28)
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.non_local = False
        self.nl1 = NONLocalBlock1D(32)
        self.conv1d_1 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)

        self.zero_type = 'null'

    def forward(self, x_ct, x_tb):
        if self.zero_type == 'tb':
            out_ct = self.ct_model(x_ct)
            out_ct = self.projection_layer(out_ct)
            out = torch.cat((out_ct, torch.zeros((out_ct.shape[0], 14), device=out_ct.device)), 1)
        elif self.zero_type == 'ct':
            out_tb = self.projection_layer_tb(x_tb)
            out = torch.cat((torch.zeros((out_tb.shape[0], 14), device=out_tb.device), out_tb), 1)
        else:
            out_ct = self.ct_model(x_ct)
            out_ct = self.projectionMLP(out_ct)
            # out_ct = self.projection_layer(out_ct)
            out_tb = x_tb
            out_ct_imitation = out_ct[:, : 14]
            out_ct = out_ct[:, 14:]
            out = torch.cat((out_ct, out_tb), 1)

        if self.non_local:
            out = out.view(out.shape[0], 1, -1)
            out = self.conv1d_1(out)
            out = self.nl1(out)
            out = self.conv1d_2(out)
            out = out.view(out.shape[0], -1)
        out = self.fusion_layer(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out, out_ct_imitation


class Fusion9(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion9, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 28),
            nn.BatchNorm1d(28),
            nn.ReLU()
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.non_local = False
        self.nl1 = NONLocalBlock1D(32)
        self.conv1d_1 = nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)

        self.zero_type = 'null'

    def forward(self, x_ct, x_tb):
        if self.zero_type == 'tb':
            out_ct = self.ct_model(x_ct)
            out_ct = self.projection_layer(out_ct)
            out = torch.cat((out_ct, torch.zeros((out_ct.shape[0], 14), device=out_ct.device)), 1)
        elif self.zero_type == 'ct':
            out_tb = self.projection_layer_tb(x_tb)
            out = torch.cat((torch.zeros((out_tb.shape[0], 14), device=out_tb.device), out_tb), 1)
        else:
            out_ct = self.ct_model(x_ct)
            out_ct_imitation = out_ct[:, : 14]
            out = out_ct

        if self.non_local:
            out = out.view(out.shape[0], 1, -1)
            out = self.conv1d_1(out)
            out = self.nl1(out)
            out = self.conv1d_2(out)
            out = out.view(out.shape[0], -1)
        out = self.projectionMLP(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out, out_ct_imitation


# alignment
class Fusion10(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion10, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        ct_projection_num = 14

        self.projection_layer_ct = nn.Sequential(
            nn.Linear(2048, ct_projection_num),
            nn.BatchNorm1d(ct_projection_num),
            nn.Sigmoid()
        )
        self.projection_layer_fusion = nn.Sequential(
            nn.Linear(14 + ct_projection_num, 28),
            nn.BatchNorm1d(28),
            nn.Sigmoid()
        )

        self.adaption_layer = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )

        self.mmd_function = MMDLoss()

    def forward(self, x_ct, x_tb):
        out_ct = self.ct_model(x_ct)
        out_ct = self.projectionMLP(out_ct)
        out_ct = self.projection_layer_ct(out_ct)
        # out_tb = self.tb_model(x_tb)

        out_ct_adaption = self.adaption_layer(out_ct)

        mmd_loss = self.mmd_function(out_ct_adaption, x_tb)

        out_ct = out_ct_adaption + out_ct

        mid_ct_tb = torch.cat((out_ct, x_tb), 1)

        # mid_x = torch.cat((mid_ct_tb, x_tb), 1)
        mid_x = self.projection_layer_fusion(mid_ct_tb)

        out = self.fusion_layer(mid_x)
        out = self.dropout(out)
        out = self.classification(out)
        return out, mmd_loss


# new non local
class Fusion11(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion11, self).__init__()
        # in_feature = in_feature // 2
        self.fusion_layer = nn.Sequential(
            # nn.Linear(36352, in_feature),  # 128 channels
            nn.Linear(19968, in_feature),  # 64 channels
            # nn.Linear(11776, in_feature),  # 32 channels
            # nn.Linear(7680, in_feature),  # 16 channels
            # nn.Linear(19456, in_feature),  # fusion channel
            # nn.Linear(2496, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.projectionMLP1 = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.non_local = True
        self.nl1 = NONLocalBlock1D(78)
        self.nl2 = NONLocalBlock1D(78)
        self.conv1d_1 = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x_ct, x_tb):
        out_ct_ori = self.ct_model(x_ct)
        out_ct = self.projectionMLP(out_ct_ori)  # >= 0
        # out_ct1 = self.projectionMLP1(out_ct_ori)

        out_ct = out_ct.view(out_ct.shape[0], out_ct.shape[1], -1)
        out_ct = out_ct.permute(0, 2, 1)

        # out_ct1 = out_ct1.view(out_ct1.shape[0], out_ct1.shape[1], -1)
        # out_ct1 = out_ct1.permute(0, 2, 1)

        x_tb = x_tb.unsqueeze(1)
        zero_tensor = torch.zeros((out_ct.shape[0], out_ct.shape[1], x_tb.shape[2]), device=x_tb.device)
        x_tb = x_tb + zero_tensor
        # x_tb = self.conv1d_1(x_tb)

        out_ori = torch.cat((out_ct, x_tb), 2)
        out_ori = out_ori.permute(0, 2, 1)  # g
        # out_ori1 = torch.cat((out_ct1, x_tb), 2)

        if self.non_local:
            out = self.nl1(out_ori)
            # out = out_ori + out  # g
            # print(out.shape)
            out = out.view(out.shape[0], -1)
            # out1 = self.nl2(out_ori1)
            # out1 = out_ori1 + out1
            # out1 = out1.view(out1.shape[0], -1)
            # out = torch.cat((out, out1), 1)
        out = self.fusion_layer(out)

        out = self.dropout(out)
        out = self.classification(out)
        return out


class Fusion11One(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion11One, self).__init__()
        # in_feature = in_feature // 2
        self.fusion_layer = nn.Sequential(
            # nn.Linear(36352, in_feature),  # 128 channels
            nn.Linear(78, in_feature),  # 64 channels
            # nn.Linear(11776, in_feature),  # 32 channels
            # nn.Linear(7680, in_feature),  # 16 channels
            # nn.Linear(19456, in_feature),  # fusion channel
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.projectionMLP1 = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.non_local = True
        self.nl1 = NONLocalBlock1D(78, sub_sample=False)
        self.nl2 = NONLocalBlock1D(78, sub_sample=False)
        self.conv1d_1 = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)


    def forward(self, x_ct, x_tb):
        out_ct_ori = self.ct_model(x_ct)
        out_ct = self.projectionMLP(out_ct_ori)
        # out_ct1 = self.projectionMLP1(out_ct_ori)
        out_ct = F.avg_pool3d(out_ct, (out_ct.shape[2], out_ct.shape[3], out_ct.shape[4]))
        out_ct = out_ct.view(out_ct.shape[0], out_ct.shape[1], -1)
        out_ct = out_ct.permute(0, 2, 1)

        # out_ct1 = out_ct1.view(out_ct1.shape[0], out_ct1.shape[1], -1)
        # out_ct1 = out_ct1.permute(0, 2, 1)

        x_tb = x_tb.unsqueeze(1)
        zero_tensor = torch.zeros((out_ct.shape[0], out_ct.shape[1], x_tb.shape[2]), device=x_tb.device)
        x_tb = x_tb + zero_tensor
        # x_tb = self.conv1d_1(x_tb)

        out_ori = torch.cat((out_ct, x_tb), 2)
        out_ori = out_ori.permute(0, 2, 1)  # g
        # out_ori1 = torch.cat((out_ct1, x_tb), 2)

        if self.non_local:
            out = self.nl1(out_ori)
            # out = out_ori + out
            # print(out.shape)
            out = out.view(out.shape[0], -1)
            # out1 = self.nl2(out_ori1)
            # out1 = out_ori1 + out1
            # out1 = out1.view(out1.shape[0], -1)
            # out = torch.cat((out, out1), 1)
        out = self.fusion_layer(out)

        out = self.dropout(out)
        out = self.classification(out)
        return out


class Fusion11Weight(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion11Weight, self).__init__()
        # in_feature = in_feature // 2
        self.fusion_layer = nn.Sequential(
            # nn.Linear(36352, in_feature),  # 128 channels
            nn.Linear(19968, in_feature),  # 64 channels
            # nn.Linear(11776, in_feature),  # 32 channels
            # nn.Linear(7680, in_feature),  # 16 channels
            # nn.Linear(19456, in_feature),  # fusion channel
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.projectionMLP1 = nn.Sequential(
            nn.Conv3d(256, 32, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

        self.non_local = True
        self.nl1 = NONLocalBlock1D(78, fc_weight=True)
        self.nl2 = NONLocalBlock1D(78)
        self.conv1d_1 = nn.Conv1d(1, 256, kernel_size=1, stride=1, padding=0)
        self.conv1d_2 = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x_ct, x_tb):
        out_ct_ori = self.ct_model(x_ct)
        out_ct = self.projectionMLP(out_ct_ori)

        out_ct = out_ct.view(out_ct.shape[0], out_ct.shape[1], -1)
        out_ct = out_ct.permute(0, 2, 1)

        x_tb = x_tb.unsqueeze(1)
        zero_tensor = torch.zeros((out_ct.shape[0], out_ct.shape[1], x_tb.shape[2]), device=x_tb.device)
        x_tb = x_tb + zero_tensor

        out_ori = torch.cat((out_ct, x_tb), 2)
        out_ori = out_ori.permute(0, 2, 1)  # g
        if self.non_local:
            out = self.nl1(out_ori)
            out = out_ori + out
            # print(out.shape)
            out = out.contiguous().view(out.shape[0], -1)
            # out1 = self.nl2(out_ori1)
            # out1 = out_ori1 + out1
            # out1 = out1.view(out1.shape[0], -1)
            # out = torch.cat((out, out1), 1)
        out = self.fusion_layer(out)

        out = self.dropout(out)
        out = self.classification(out)
        return out


class Fusion12(nn.Module):
    def __init__(self, in_feature, num_class, ct_model, tb_model):
        super(Fusion12, self).__init__()
        # in_feature = in_feature // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            # nn.Linear(in_feature, num_class)
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.ct_model = ct_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, 14),
            nn.BatchNorm1d(14),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(14, 14),
            nn.BatchNorm1d(14),
            nn.ReLU()
        )

    def forward(self, x_ct, x_tb):
        out_ct = self.ct_model(x_ct)
        out_ct = self.projectionMLP(out_ct)
        out_ct = self.projection_layer(out_ct)
        out_tb = self.projection_layer_tb(x_tb)
        # cat
        out = torch.cat((out_ct, out_tb), 1)

        out = self.fusion_layer(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:1')

    mlp_path = '/home/lab632/data_win/632_workspace/ChenCan/CRS_Model/fusion_model/saved_model/' \
               'ML_CCR/202203112041/epoch441_acc0.38.pth.tar'
    resnet_path = '/home/lab632/data_win/632_workspace/ChenCan/CRS_Model/dl_model/SimSiam/saved_model/' \
                  'SimSiam_model_colon_mixUp_best.pth.tar'

    mlp_feature = MLPFeature().to(device)
    _, res_feature = load_pretrained_model('SimSiam', 'MLPFeature', mlp_path, resnet_path, device)
    fusion_model = Fusion11Weight(14 + 14, 2, res_feature, mlp_feature)
    fusion_model = fusion_model.to(device)

    test_tbx = torch.randn((2, 14)).to(device)
    test_ctx = torch.randn((2, 1, 64, 128, 128)).to(device)
    output = fusion_model(test_ctx, test_tbx)
    print(output.shape)


