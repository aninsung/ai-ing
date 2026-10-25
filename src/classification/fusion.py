import torch
from torch import nn
import torchvision.models as models

class FusionResNet50(nn.Module):
    """Global-image + ROI feature fusion classifier."""
    def __init__(self, num_classes=15, auxiliary_classes=14, pretrained=False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        a, b = models.resnet50(weights=weights), models.resnet50(weights=weights)
        dim = a.fc.in_features
        self.global_encoder = nn.Sequential(*list(a.children())[:-1])
        self.roi_encoder = nn.Sequential(*list(b.children())[:-1])
        self.primary_head = nn.Linear(dim * 2, num_classes)
        self.auxiliary_head = nn.Linear(dim * 2, auxiliary_classes)
    def forward(self, image, roi):
        g = self.global_encoder(image.repeat(1,3,1,1)).flatten(1)
        r = self.roi_encoder(roi.repeat(1,3,1,1)).flatten(1)
        fused = torch.cat([g,r],1)
        return self.primary_head(fused), self.auxiliary_head(fused)
