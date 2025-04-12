import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *

class objectHead(nn.Module) : 

    def __init__(self, num_classes, bilinear = False) : 
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 63, bilinear))
        self.outc = (OutConv(63, num_classes)) # num_classes : 2
    
    def forward(self, x) :
        out = self.up1(x[4], x[3])
        out = self.up2(out, x[2])
        out = self.up3(out, x[1])
        out1 = self.up4(out, x[0])
        logits = self.outc(out1)
        return logits, out1
    
class shadowHead(nn.Module) : 

    def __init__(self, num_classes, bilinear = False) : 
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 63, bilinear))
        self.outc = (OutConv(63, num_classes))
    
    def forward(self, x) :
        feats, atten = x[0], x[1]
        out = self.up1(feats[4], feats[3])
        out = self.up2(out, feats[2])
        out = self.up3(out, feats[1])
        out = self.up4(out, feats[0])
        out = out * atten
        logits = self.outc(out)
        return logits