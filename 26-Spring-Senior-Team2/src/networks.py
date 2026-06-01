import torch
import torch.nn as nn
import torchvision.models as models

class DDQNNetwork(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet50"):
        super(DDQNNetwork, self).__init__()
        
        # Phase 3: CNN 백본 기반 특징 추출 (ResNet50)
        if backbone_name == "resnet50":
            # torchvision 제공 resnet50 활용 (MONAI 대체 가능)
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # X-ray 이미지는 보통 1채널이므로 첫 번째 Conv 계층 수정
            # (만약 EnsureChannelFirstd에서 3채널로 변환했다면 수정 생략 가능)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # 최종 FC 분류 레이어를 행동 공간(num_classes) 크기의 Q-value 출력으로 대체
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} is not implemented.")

    def forward(self, x):
        """
        Input: 상태 (State) 이미지 텐서 [B, C, H, W]
        Output: 행동 공간에 대한 Q-value 벡터 [B, num_classes]
        """
        return self.backbone(x)
