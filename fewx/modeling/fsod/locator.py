import torch.nn as nn
import torch.nn.functional as F
import torch as t
import logging

__all__ = ["LocatorNet"]
logger = logging.getLogger(__name__)

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class LocatorNet(nn.Module):
    
    def __init__(self, pro_size, channels, representation_size):
        super(LocatorNet, self).__init__()

        #layer definition
        self.support_fc = nn.Sequential(
            nn.Linear(pro_size ** 2, channels),
            nn.Linear(channels, 1)
        )
        self.conv1=nn.Sequential(
                nn.Conv2d(channels, channels,  kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                )
        self.mlp=nn.Sequential(
                nn.Linear(channels * pro_size ** 2, representation_size),
                nn.ReLU(inplace=True),
                nn.Linear(representation_size, representation_size),
                nn.ReLU(inplace=True),
                )
        self.loc_fc = nn.Linear(representation_size, 4)
        
        #layer init
        for modules in [self.conv1, self.support_fc]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
                if isinstance(l, nn.Linear):
                    t.nn.init.normal_(l.weight, std=0.001)
                    t.nn.init.constant_(l.bias, 0)
        for modules in [self.mlp]:
            for l in modules.modules():
                if isinstance(l, nn.Linear):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
        t.nn.init.normal_(self.loc_fc.weight, std=0.01)
        t.nn.init.constant_(self.loc_fc.bias, 0)

    def relate_locator(self,x, kernel):
        """depthwise cross correlation
        """
        num = kernel.size(0)
        channel = kernel.size(1)
        a = kernel.size(2)
        padding = int((a-1)/2)
        # can't use view
        x = x.reshape(-1, num*channel, x.size(3), x.size(4))
        kernel = kernel.reshape(num*channel, 1, kernel.size(2), kernel.size(2))
        out = F.conv2d(x, kernel, groups=num*channel, padding=padding)
        return out+x
    
    def forward(self,x,z):

        x1 = x.view(x.size(0), -1, x.size(1), x.size(2), x.size(3)).expand(x.size(0), z.size(0), x.size(1), x.size(2), x.size(3))
        z1 = self.conv1(z)
        z2 = z1.view(z1.size(0), z1.size(1), z1.size(2)*z1.size(3))
        k = self.support_fc(z2)
        res=self.relate_locator(x1,k)
        res = res.flatten(start_dim=1)
        v = self.mlp(res)
        bbox_delta = self.loc_fc(v)

        return bbox_delta