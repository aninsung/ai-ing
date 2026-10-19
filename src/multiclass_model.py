import torch
import torch.nn as nn
import torchvision.models as models


def _encoder(backbone, pretrained):
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        network = models.resnet50(weights=weights)
        encoder = nn.Sequential(*list(network.children())[:-1])
        return encoder, network.fc.in_features
    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        network = models.densenet121(weights=weights)
        # DenseNet applies ReLU after features and before global pooling.
        encoder = nn.Sequential(network.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        return encoder, network.classifier.in_features
    raise ValueError(f"Unsupported backbone: {backbone}. Use resnet50 or densenet121.")


class MultiTaskClassifier(nn.Module):
    """Shared image encoder with a 15-class primary head and a 14-label auxiliary head."""
    def __init__(self, num_classes=15, auxiliary_classes=14, pretrained=True, backbone="resnet50"):
        super().__init__()
        self.backbone_name = backbone
        self.encoder, features = _encoder(backbone, pretrained)
        self.primary_head = nn.Linear(features, num_classes)
        self.auxiliary_head = nn.Linear(features, auxiliary_classes)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, images):
        x = images.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.encoder(x).flatten(1)

    def forward(self, images):
        features = self.encode(images)
        return self.primary_head(features), self.auxiliary_head(features)


class MultiTaskResNet50(MultiTaskClassifier):
    """ResNet-50 multitask classifier. Kept so existing checkpoints and scripts load unchanged."""
    def __init__(self, num_classes=15, auxiliary_classes=14, pretrained=True):
        super().__init__(num_classes, auxiliary_classes, pretrained, backbone="resnet50")


def build_multitask_model(backbone="resnet50", num_classes=15, auxiliary_classes=14, pretrained=True):
    return MultiTaskClassifier(num_classes, auxiliary_classes, pretrained, backbone=backbone)
