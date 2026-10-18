import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskResNet50(nn.Module):
    """Shared ResNet representation with multiclass and multilabel heads."""
    def __init__(self, num_classes=15, auxiliary_classes=14, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        features = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.primary_head = nn.Linear(features, num_classes)
        self.auxiliary_head = nn.Linear(features, auxiliary_classes)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        x = images.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        features = self.encoder(x).flatten(1)
        return self.primary_head(features), self.auxiliary_head(features)
