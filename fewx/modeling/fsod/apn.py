import torch.nn as nn
import torch.nn.functional as F
import torch as t
from math import *
import logging

__all__ = ["APN", "Interp", "GConv"]
logger = logging.getLogger(__name__)
class Interp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Interp, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )
        for modules in [self.conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)

    def forward(self, feature, target_size, interp_mode="bicubic"):
        return self.conv(F.interpolate(feature, target_size, mode=interp_mode))

class GConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, k_size):
        super(GConv, self).__init__()

        #layer definition
        self.conv_g=nn.Sequential(
                nn.Conv2d(in_channels, out_channels,  kernel_size=k_size, stride=k_size),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )
        self.conv=nn.Sequential(
                nn.Conv2d(in_channels, out_channels,  kernel_size=k_size, stride=k_size, groups=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )
        
        #layer init
        for modules in [self.conv_g, self.conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
    
    def forward(self, x):
        res = self.conv(x)+self.conv_g(x)
        return res



class APN(nn.Module):
    def __init__(self, channels):
        super(APN, self).__init__()

        #layer definition
        self.r1=Interp(channels*2, channels)
        # self.r2=Interp(channels*4, channels)
        self.gconv1=GConv(512, 512, 1)
        # self.gconv2=GConv(512, 512, 1)
        self.r3=Interp(channels*2, channels*4)
        self.v_2 = nn.Parameter(t.FloatTensor(1))
        # self.r4=Interp(channels*2, channels*4)
    
    def forward(self, correlation_feat):
        res2, res3, res4 = correlation_feat["res2"], correlation_feat["res3"], correlation_feat["res4"]
        w_2, h_2 = res2.size(2), res2.size(3)
        w_4, h_4 = res4.size(2), res4.size(3)
        res3_2 = t.cat([self.r1(res3, (w_2, h_2), interp_mode="nearest"), res2], dim=1)
        # res4_2 = t.cat([self.r2(res4, (w_2, h_2), interp_mode="nearest"), res2], dim=1)
        res3_2_g = self.gconv1(res3_2)
        # res4_2_g = self.gconv2(res4_2)
        res = self.r3(res3_2_g, (w_4, h_4))*self.v_2+res4
        return res