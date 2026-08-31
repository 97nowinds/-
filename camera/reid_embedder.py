import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class IBN(nn.Module):
    """与现有 Re-ID 权重匹配的实例归一化/批归一化混合层。"""

    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.half = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(channels - half)

    def forward(self, inputs):
        first, second = torch.split(inputs, [self.half, inputs.size(1) - self.half], 1)
        return torch.cat((self.IN(first.contiguous()), self.BN(second.contiguous())), 1)


class ReIDNetwork(nn.Module):
    """ResNet50-IBN Re-ID 推理网络，输出 2048 维身份特征。"""

    def __init__(self):
        super().__init__()
        self.base = resnet50(weights=None)
        for layer in (self.base.layer1, self.base.layer2, self.base.layer3):
            for block in layer:
                block.bn1 = IBN(block.bn1.num_features)
        # 人物重识别保留更多空间细节，最后一个 stage 使用 last_stride=1。
        self.base.layer4[0].conv2.stride = (1, 1)
        self.base.layer4[0].downsample[0].stride = (1, 1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, inputs):
        model = self.base
        features = model.conv1(inputs)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)
        return F.normalize(self.bottleneck(features), p=2, dim=1)


class ReIDEmbedder:
    """从完整人体框提取归一化 Re-ID 特征，供跨摄像头余弦匹配。"""

    def __init__(self, weight_path, device=None):
        self.weight_path = Path(weight_path)
        if not self.weight_path.is_file():
            raise FileNotFoundError(f"Re-ID 权重不存在：{self.weight_path}")
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = ReIDNetwork()
        state = torch.load(self.weight_path, map_location="cpu")
        usable = {
            key: value
            for key, value in state.items()
            if key.startswith("base.") or key.startswith("bottleneck.")
        }
        missing, unexpected = self.model.load_state_dict(usable, strict=False)
        allowed_missing = {"base.fc.weight", "base.fc.bias"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                f"Re-ID 权重与网络不匹配，missing={missing}, unexpected={unexpected}"
            )
        self.model.to(self.device).eval()
        self.lock = threading.RLock()

    @staticmethod
    def crop_person(frame, box):
        if frame is None or box is None:
            return None
        x, y, width, height = [int(round(value)) for value in box]
        frame_height, frame_width = frame.shape[:2]
        left = max(0, min(x, frame_width - 1))
        top = max(0, min(y, frame_height - 1))
        right = max(left + 1, min(x + width, frame_width))
        bottom = max(top + 1, min(y + height, frame_height))
        crop = frame[top:bottom, left:right]
        return crop if crop.size else None

    @staticmethod
    def preprocess(crop):
        resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return tensor.sub_(mean).div_(std).unsqueeze(0)

    def extract(self, frame, box):
        crop = self.crop_person(frame, box)
        if crop is None or crop.shape[0] < 32 or crop.shape[1] < 12:
            return None
        inputs = self.preprocess(crop).to(self.device)
        with self.lock, torch.inference_mode():
            feature = self.model(inputs)[0].cpu().numpy().astype(np.float32)
        return feature
