from .backbone import *
from .head import *
from .attention import *

class SLAN(nn.Module) :
    def __init__(self, num_classes, in_channels, bilinear=False) :
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear

        self.backbone = backbone(self.in_channels, self.bilinear)
        self.object_head = objectHead(self.num_classes, self.bilinear)
        self.shadow_head = shadowHead(self.num_classes, self.bilinear)
        self.attention_module = AttentionModule()

    def forward(self, x) :
        input, light = x[0], x[1]
        backbone_feats = self.backbone(input)
        object_pred, out1 = self.object_head(backbone_feats)
        out1 = out1.clone().detach()
        atten = self.attention_module([out1, light])
        shadow_pred = self.shadow_head([backbone_feats, atten])
        return object_pred, shadow_pred