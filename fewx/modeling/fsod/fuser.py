import torch.nn as nn
import torch.nn.functional as F
import torch as t
import logging

__all__ = ["FuseNet"]
logger = logging.getLogger(__name__)

class FuseNet(nn.Module):
    
    def __init__(self, channels, representation_size):
        super(FuseNet, self).__init__()

        #layer definition
        self.conv1=nn.Sequential(
                nn.Conv2d(channels, channels,  kernel_size=1, stride=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                )
        self.conv2=nn.Sequential(
                nn.Conv2d(channels*2, representation_size,  kernel_size=1, stride=1),
                nn.BatchNorm2d(representation_size),
                nn.ReLU(inplace=True),
                )
        self.dw_conv=nn.Sequential(
                nn.Conv2d(channels*2, representation_size,  kernel_size=1, stride=1, groups=2),
                nn.BatchNorm2d(representation_size),
                nn.ReLU(inplace=True),
                )
        self.mlp=nn.Sequential(
                nn.Linear(representation_size, representation_size),
                nn.ReLU(inplace=True),
                nn.Linear(representation_size, 1),
                nn.ReLU(inplace=True),
                nn.Softmax(dim=0),
                )
        
        #layer init
        for modules in [self.conv1, self.conv2, self.dw_conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
        for modules in [self.mlp]:
            for l in modules.modules():
                if isinstance(l, nn.Linear):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
    
    def forward(self,supports):
        n = supports.size(0) 
        x = self.conv1(supports)
        avg_x = x.mean(0, keepdim=True)
        x_cat = t.cat([x, avg_x.repeat(n,1,1,1)], dim=1)
        x = self.conv2(x_cat) + self.dw_conv(x_cat)
        x = x.permute(0,2,3,1)
        att = self.mlp(x)
        fused_supports = (att.permute(0,3,1,2))*supports
        fused_supports = fused_supports.sum(dim=0, keepdim=True)
        return fused_supports