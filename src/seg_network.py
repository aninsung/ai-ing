"""UNet++ segmentation backbone for lesion-mask supervision.

The dataset must contain approved pixel masks (or explicitly reviewed
negatives); disease-level labels alone are not used as segmentation targets.
"""
import torch
from torch import nn


def build_model():
    return UNetPlusPlus()


class ConvBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.InstanceNorm2d(out), nn.ReLU(inplace=True), nn.Conv2d(out, out, 3, padding=1), nn.InstanceNorm2d(out), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, base=32):
        super().__init__(); self.pool = nn.MaxPool2d(2)
        self.x00=ConvBlock(1,base); self.x10=ConvBlock(base,base*2); self.x20=ConvBlock(base*2,base*4); self.x30=ConvBlock(base*4,base*8)
        self.x01=ConvBlock(base*3,base); self.x11=ConvBlock(base*7,base*2); self.x21=ConvBlock(base*14,base*4)
        self.x02=ConvBlock(base*4,base); self.x12=ConvBlock(base*9,base*2); self.x03=ConvBlock(base*5,base); self.out=nn.Conv2d(base,1,1)
    def up(self,x,ref): return nn.functional.interpolate(x,size=ref.shape[-2:],mode='bilinear',align_corners=False)
    def forward(self,x):
        x00=self.x00(x); x10=self.x10(self.pool(x00)); x20=self.x20(self.pool(x10)); x30=self.x30(self.pool(x20))
        x01=self.x01(torch.cat([x00,self.up(x10,x00)],1)); x11=self.x11(torch.cat([x10,self.up(x20,x10),self.pool(x01)],1)); x21=self.x21(torch.cat([x20,self.up(x30,x20),self.pool(x11)],1))
        x02=self.x02(torch.cat([x00,x01,self.up(x11,x00)],1)); x12=self.x12(torch.cat([x10,x11,self.up(x21,x10),self.pool(x02)],1)); x03=self.x03(torch.cat([x00,x01,x02,self.up(x12,x00)],1))
        return self.out(x03)
