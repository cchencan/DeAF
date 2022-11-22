# -*- encoding: utf-8 -*-
"""
@File    :   fusion_model.py    
@Contact :   425077099@qq.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/10/22 21:03   C.Chen     1.0         None
"""
import torch
import torch.nn as nn
from model.resNet3d import ResNet3dFeature, ResBlock
from model.mlp import MLPFeature
from model.non_local_embedded_gaussian import NONLocalBlock1D, NONLocalBlock2D


class FusionCat(nn.Module):
    def __init__(self, in_feature, num_class, image_model, tb_model, fc, tb_dim=3, img_chn=64):
        super(FusionCat, self).__init__()
        # in_feature = in_feature // 2
        self.fc = fc
        self.fusion_layer = nn.Sequential(
            nn.Linear(tb_dim + tb_dim, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.5)

        self.image_model = image_model
        self.tb_model = tb_model

        self.projection_layer = nn.Sequential(
            nn.Linear(2048, tb_dim),
            nn.BatchNorm1d(tb_dim),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projectionMLP = nn.Sequential(
            # nn.Linear(65536, 2048),
            nn.Linear(13824, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            # nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(inplace=True)
            nn.Linear(2048, tb_dim),
            nn.BatchNorm1d(tb_dim),
            # nn.Sigmoid()
            nn.ReLU()
        )

        self.projection_layer_tb = nn.Sequential(
            nn.Linear(tb_dim, tb_dim),
            nn.BatchNorm1d(tb_dim),
            nn.ReLU()
        )

    def forward(self, x_img, x_tb):

        out_img = self.image_model(x_img)
        out_img = out_img.view(out_img.shape[0], -1)
        out_img = self.projectionMLP(out_img)

        if self.fc:
            out_tb = self.projection_layer_tb(x_tb)
        else:
            out_tb = x_tb
        # cat
        out = torch.cat((out_img, out_tb), 1)

        out = self.fusion_layer(out)
        out = self.dropout(out)
        out = self.classification(out)
        return out


class FusionSA(nn.Module):
    def __init__(self, in_feature, num_class, image_model, tb_model, tb_dim=3, img_chn=64):
        super(FusionSA, self).__init__()
        self.fusion_layer = nn.Sequential(
            # nn.Linear(36352, in_feature),  # 128 channels
            # nn.Linear(1809, in_feature),  # 64 channels - tb_dim=3
            nn.Linear(27 * (tb_dim + img_chn), in_feature),  # 64 channels - tb_dim=138
            # nn.Linear(11776, in_feature),  # 32 channels
            # nn.Linear(7680, in_feature),  # 16 channels
            # nn.Linear(19456, in_feature),  # fusion channel
            # nn.Linear(2496, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, in_feature),
            nn.BatchNorm1d(in_feature),
            nn.ReLU(),
        )
        self.classification = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(p=0.6)

        self.image_model = image_model
        self.tb_model = tb_model

        self.projectionMLP = nn.Sequential(
            nn.Conv3d(512, img_chn, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(img_chn),
            # nn.ReLU()
            nn.Sigmoid()
        )

        self.nl1 = NONLocalBlock1D(img_chn+tb_dim)

    def forward(self, x_image, x_tb):
        out_image = self.image_model(x_image)
        out_image = self.projectionMLP(out_image)  # >= 0

        # transformation
        out_image = out_image.view(out_image.shape[0], out_image.shape[1], -1)
        out_image = out_image.permute(0, 2, 1)

        # broadcast
        x_tb = x_tb.unsqueeze(1)
        zero_tensor = torch.zeros((out_image.shape[0], out_image.shape[1], x_tb.shape[2]), device=x_tb.device)
        x_tb = x_tb + zero_tensor

        out_ori = torch.cat((out_image, x_tb), 2)
        out_ori = out_ori.permute(0, 2, 1)  # g

        out = self.nl1(out_ori)
        out = out.view(out.shape[0], -1)
        out = self.fusion_layer(out)

        out = self.dropout(out)
        out = self.classification(out)
        return out


def main():
    device = torch.device('cuda:{}'.format(1) if torch.cuda.is_available() else 'cpu')
    # (model_type, model_name, resnet_path, device, option)
    image_model = ResNet3dFeature(ResBlock, 4, False).to(device)
    mlp = MLPFeature().to(device)
    model = FusionCat(1024, 2, image_model, mlp, False, 138).to(device)

    x_image = torch.zeros((2, 1, 79, 95, 95)).to(device)
    x_tb = torch.zeros((2, 138)).to(device)

    y = model(x_image, x_tb)
    print(y)


if __name__ == '__main__':
    main()