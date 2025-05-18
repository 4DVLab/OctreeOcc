import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
import numpy as np
import mmcv
from mmdet.models.builder import build_loss
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16

BatchNorm = nn.BatchNorm2d
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@HEADS.register_module()
class NaiveSegHead(BaseModule):
    def __init__(self,
                loss_pred=None,
                loss_aux=None,
                **kwargs):
        super(NaiveSegHead, self).__init__()
        self.downsample_factors = [4, 8, 16, 32, 64]
        self.up4 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1), BatchNorm(1024), nn.ReLU())
        self.up3 = nn.Sequential(nn.Conv2d(1024,512, kernel_size=3, stride=1, padding=1), BatchNorm(512), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), BatchNorm(256), nn.ReLU())
        self.delayer4 = self.make_layer(BasicBlock, 1024, 1, inplanes=2048)
        self.delayer3 = self.make_layer(BasicBlock, 512, 1, inplanes=1024)
        self.delayer2 = self.make_layer(BasicBlock, 256, 1, inplanes=512)
        self.cls = nn.Sequential(
            nn.Conv2d(256,128, kernel_size=3, padding=1, bias=False),
            BatchNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 17, kernel_size=1)
        )
        if loss_pred != None:
            self.loss_pred = build_loss(loss_pred)
        if loss_aux is not None:
            self.aux_loss = build_loss(loss_aux)

    def make_layer(self, block, planes, blocks, stride=1, inplanes=128):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    @auto_fp16(apply_to=('img_feats'))
    def forward(self, img_feats):
        # for i in range(len(img_feats)):
        #     B, N, C, H, W = img_feats[i].shape
        #     img_feats[i] = img_feats[i].reshape(B*N, C, H, W)

        x1 = img_feats[0]
        x2 = img_feats[1]
        x3 = img_feats[2]
        x4 = img_feats[3]
        _, _, h, w = x1.shape

        p4 = self.up4(F.interpolate(x4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p4 = torch.cat([p4, x3], dim=1)
        p4 = self.delayer4(p4)
        p3 = self.up3(F.interpolate(p4, x2.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, x2], dim=1)
        p3 = self.delayer3(p3)
        p2 = self.up2(F.interpolate(p3, x1.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, x1], dim=1)
        p2 = self.delayer2(p2)
        x = self.cls(p2)
        x = F.interpolate(x, size=(72*4, 120*4), mode='bilinear', align_corners=True)

        return x
    
    @force_fp32(apply_to=('depth_preds'))
    def loss(self, seg_labels, seg_preds):
        # seg_preds: bs*6, C, h, w
        # seg_labels: bs, 6, h, w
        loss_dict=dict()
        if isinstance(seg_labels, list):
            seg_labels = torch.stack(seg_labels, dim=0)

        bs, N, h, w = seg_labels.shape
        seg_labels = seg_labels.reshape(bs*N, h, w)
        seg_preds = seg_preds.permute(0,2,3,1)
        mask = (seg_labels != 255).bool()

        # CE loss
        labels_for_CE = seg_labels[mask]
        preds_for_CE = seg_preds[mask]
        assert labels_for_CE.min()>=0 and labels_for_CE.max()<=16

        loss_dict['loss_seg_CE'] = self.loss_pred(preds_for_CE,labels_for_CE.long())

        # DiceLoss
        visible_pred_voxels = preds_for_CE
        visible_target_voxels = labels_for_CE
        visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 17)
        loss_dict['loss_seg_dice'] = self.aux_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict

    def get_seg_pred(self, seg_preds):
        seg_preds = torch.argmax(seg_preds, dim=1)
        seg_res = torch.argmax(seg_preds, dim=1)
        return seg_res