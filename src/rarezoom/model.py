"""Global classifier and the RareZoom fusion network."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def _encoder():
    network = resnet50(weights=None)
    network.fc = nn.Identity()
    return network


class GlobalClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        network = resnet50(weights=weights)
        features = network.fc.in_features
        network.fc = nn.Linear(features, num_classes)
        self.network = network
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images):
        pixels = images.repeat(1, 3, 1, 1)
        pixels = (pixels - self.mean) / self.std
        return self.network(pixels)


class RareZoomNet(nn.Module):
    """Frozen or trainable global encoder, local encoder, and GRU fusion head."""

    def __init__(self, num_classes=14, hidden=512):
        super().__init__()
        self.hidden = hidden
        self.global_encoder = _encoder()
        self.local_encoder = _encoder()
        self.gru = nn.GRU(2048, hidden, batch_first=True)
        self.global_head = nn.Linear(2048, num_classes)
        self.delta_head = nn.Linear(hidden, num_classes)
        self.gate_head = nn.Linear(hidden, num_classes)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -5.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def train(self, mode=True):
        super().train(mode)
        self.global_encoder.eval()
        return self

    def load_stage1(self, checkpoint):
        encoder = {}
        for key, value in checkpoint["model"].items():
            if not key.startswith("network."):
                continue
            name = key[len("network."):]
            if name.startswith("fc."):
                continue
            encoder[name] = value
        self.global_encoder.load_state_dict(encoder)
        self.local_encoder.load_state_dict(encoder)
        with torch.no_grad():
            self.global_head.weight.copy_(checkpoint["model"]["network.fc.weight"])
            self.global_head.bias.copy_(checkpoint["model"]["network.fc.bias"])
            self.mean.copy_(checkpoint["model"]["mean"])
            self.std.copy_(checkpoint["model"]["std"])
        for parameter in self.global_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.global_head.parameters():
            parameter.requires_grad_(False)

    def _encode(self, encoder, images):
        pixels = images.repeat(1, 3, 1, 1)
        pixels = (pixels - self.mean) / self.std
        features = encoder.conv1(pixels)
        features = encoder.bn1(features)
        features = encoder.relu(features)
        features = encoder.maxpool(features)
        features = encoder.layer1(features)
        features = encoder.layer2(features)
        features = encoder.layer3(features)
        features = encoder.layer4(features)
        pooled = torch.flatten(encoder.avgpool(features), 1)
        spatial = F.adaptive_avg_pool2d(features, (3, 3)).mean(dim=1).flatten(1)
        return pooled, spatial

    def global_features(self, images):
        pooled, _ = self._encode(self.global_encoder, images)
        return pooled

    def global_observation(self, images):
        return self._encode(self.global_encoder, images)

    def local_features(self, images):
        pooled, _ = self._encode(self.local_encoder, images)
        return pooled

    def initial_state(self, batch, like):
        return like.new_zeros(1, batch, self.hidden)

    def gru_step(self, features, hidden):
        _, hidden = self.gru(features.unsqueeze(1), hidden)
        return hidden

    def predict(self, global_features, hidden):
        state = hidden.squeeze(0)
        base = self.global_head(global_features)
        gate = torch.sigmoid(self.gate_head(state))
        return base + gate * self.delta_head(state)

    def residual_logits(self, global_features, hidden):
        """Ungated residual. Trains the local path while the gate can stay closed."""
        state = hidden.squeeze(0)
        return self.global_head(global_features) + self.delta_head(state)

    def forward_sequence(self, global_images, crops, lengths):
        with torch.no_grad():
            global_features, spatial = self.global_observation(global_images)
        global_features = global_features.detach()
        spatial = spatial.detach()
        batch, steps = crops.shape[:2]
        local = self.local_features(crops.reshape(batch * steps, *crops.shape[2:]))
        local = local.view(batch, steps, -1)
        hidden = self.initial_state(batch, global_features)
        outputs = [self.predict(global_features, hidden)]
        for step in range(steps):
            updated = self.gru_step(local[:, step], hidden)
            active = (lengths > step).view(1, batch, 1)
            hidden = torch.where(active, updated, hidden)
            outputs.append(self.predict(global_features, hidden))
        return outputs, hidden, global_features, spatial


def per_sample_bce(logits, target, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return loss.mean(dim=1)


def focus_bce(logits, target, pos_weight, index):
    loss = F.binary_cross_entropy_with_logits(
        logits[:, index], target[:, index], pos_weight=pos_weight[index], reduction="none",
    )
    return loss.mean(dim=1)


class QNetwork(nn.Module):
    def __init__(self, state_dim, actions=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions),
        )

    def forward(self, state):
        return self.net(state)
