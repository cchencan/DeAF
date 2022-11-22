# -*- encoding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.resNet3d import ResNet3dFeature, ResBlock
from training_utils.loss_function import MMDLoss


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone, dim=2048, pred_dim=2048):
        super(SimSiam, self).__init__()

        self.backbone = backbone
        self.projection = ProjectionMLP(backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projection
        )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return p1, p2, z1.detach(), z2.detach()
        # return loss


class SimSiamMultiModal(nn.Module):
    def __init__(self, backbone, dim=2048, pred_dim=2048, tb_dim=14):
        super(SimSiamMultiModal, self).__init__()

        self.backbone = backbone

        self.backbone_tb = nn.Sequential(
            nn.Linear(tb_dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.projection_ct = ProjectionMLP(backbone.output_dim)
        self.projection_tb = ProjectionMLP(dim)
        self.projection = ProjectionMLP(dim + dim)

        self.encoder_ct = nn.Sequential(
            self.backbone,
            self.projection_ct
        )

        self.encoder_tb = nn.Sequential(
            self.backbone_tb,
            self.projection_tb
        )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1_ct, x2_ct, x1_tb, x2_tb):
        z1_ct = self.encoder_ct(x1_ct)
        z1_tb = self.encoder_tb(x1_tb)
        z1 = self.projection(torch.cat((z1_ct, z1_tb), 1))

        z2_ct = self.encoder_ct(x2_ct)
        z2_tb = self.encoder_tb(x2_tb)
        z2 = self.projection(torch.cat((z2_ct, z2_tb), 1))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return p1, p2, z1.detach(), z2.detach()
        # return loss


class SimSiamMultiModalV2(nn.Module):
    def __init__(self, backbone, dim=2048, pred_dim=2048, tb_dim=14):
        super(SimSiamMultiModalV2, self).__init__()

        self.backbone = backbone

        self.backbone_tb = nn.Sequential(
            nn.Linear(tb_dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.projection_ct = ProjectionMLP(backbone.output_dim)
        self.projection_tb = ProjectionMLP(dim)
        self.projection = ProjectionMLP(dim*2)

        self.encoder_ct = nn.Sequential(
            self.backbone,
            self.projection_ct
        )

        self.encoder_tb = nn.Sequential(
            self.backbone_tb,
            self.projection_tb
        )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim, tb_dim, bias=True),
            nn.BatchNorm1d(tb_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(tb_dim, dim, bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.mmd = MMDLoss()


    def forward(self, x1_ct, x2_ct, x1_tb, x2_tb):
        z1_ct = self.encoder_ct(x1_ct)
        # z1_tb = self.encoder_tb(x1_tb)

        z1_ct_fc1 = self.fc1(z1_ct)

        mmd_loss1 = self.mmd(x1_tb, z1_ct_fc1)

        z1_ct_fc2 = self.fc2(z1_ct_fc1)

        z1_ct = z1_ct + z1_ct_fc2
        z1 = self.projection(z1_ct)

        z2_ct = self.encoder_ct(x2_ct)
        # z2_tb = self.encoder_tb(x2_tb)
        z2_ct_fc1 = self.fc1(z1_ct)

        mmd_loss2 = self.mmd(x2_tb, z2_ct_fc1)

        z2_ct_fc2 = self.fc2(z2_ct_fc1)

        z2_ct = z2_ct + z2_ct_fc2

        z2 = self.projection(z2_ct)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        mmd_loss = mmd_loss1 + mmd_loss2

        # loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return p1, p2, z1.detach(), z2.detach(), mmd_loss
        # return loss


if __name__ == '__main__':
    device = torch.device('cuda:1')
    bb = ResNet3dFeature(ResBlock, 3).to(device)
    model = SimSiam(bb).to(device)
    x = torch.zeros((2, 1, 64, 128, 128)).to(device)
    out1, out2, z1, z2 = model(x, x)
