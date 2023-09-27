import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath
from nets.convnext_bn import ConvNeXt_BN
from config import cfg

class DeConv(nn.Sequential):
    def __init__(self, inplanes, planes, upscale_factor=2, kernel_size = 3, up = True):
        super().__init__()
        size = kernel_size
        pad = 1
        if kernel_size == 7:
            pad = 3
        elif kernel_size == 5:
            pad = 2
        else:
            pad = 1
        self.dwconv = nn.Conv2d(inplanes, inplanes, kernel_size=size, stride=1, padding=pad, groups=inplanes)
        self.norm = nn.BatchNorm2d(inplanes)
        self.pwconv = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=upscale_factor) if up else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = self.upsample1(x)

        return x
        

class HeadNet(nn.Module):

    def __init__(self, joint_num, in_channel):
        self.inplanes = in_channel # 2048, 768
        super(HeadNet, self).__init__()

        self.deconv_layers_1 = DeConv(inplanes=self.inplanes,planes=cfg.depth, kernel_size = 3)
        self.deconv_layers_2 = DeConv(inplanes=cfg.depth, planes=cfg.depth, kernel_size = 3)
        self.deconv_layers_3 = DeConv(inplanes=cfg.depth, planes=cfg.depth, kernel_size = 3, up = False)
        self.final_layer = nn.Conv2d(
            in_channels=cfg.depth,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.deconv_layers_1(x)
        x = self.deconv_layers_2(x)
        x = self.deconv_layers_3(x)
        x = self.final_layer(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def soft_argmax(heatmaps, joint_num):

    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(cfg.output_shape[1]).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(cfg.output_shape[0]).float().cuda()[None,None,:]
    accu_z = accu_z * torch.arange(cfg.depth_dim).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out

class ConvNeXtPose(nn.Module):
    def __init__(self, backbone,joint_num, head):
        super(ConvNeXtPose, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num
        # self.loss = nn.MSELoss()

    def forward(self, input_img, target=None):
        hm= self.backbone(input_img)
        if self.head != None:
            hm = self.head(hm)
        coord = soft_argmax(hm, self.joint_num)
        
        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
        
            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            # loss_coord = self.loss(coord, target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.
            return loss_coord

def get_pose_net(cfg, is_train, joint_num):
    drop_rate = 0
    if is_train:
        drop_rate = 0.1
    backbone = ConvNeXt_BN(depths=cfg.backbone_cfg[0], dims=cfg.backbone_cfg[1],drop_path_rate=drop_rate) 
    head_net = HeadNet(joint_num, in_channel = cfg.backbone_cfg[1][-1])

    model = ConvNeXtPose(backbone, joint_num, head =head_net)
    return model

