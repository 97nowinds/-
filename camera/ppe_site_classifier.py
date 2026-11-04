"""Experimental PPE classifier heads; outputs never authorize an alert.

The checkpoint contains trainable heads, not a fine-tuned language model.
Visibility must be checked separately before using a classification as evidence.
"""
from pathlib import Path


class SitePPEClassifier:
    def __init__(self, checkpoint, device="cuda:0"):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6VisionModel

        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Site PPE inference requires a cluster CUDA GPU")
        self.device = device
        self.checkpoint = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if self.checkpoint.get("schema") != "site_ppe_linear_v2":
            raise ValueError("Unsupported site PPE checkpoint")
        model_path = Path(self.checkpoint["vision_model"])
        config = AutoConfig.from_pretrained(model_path, local_files_only=True).vision_config
        config._attn_implementation = "sdpa"
        self.vision = MiniCPMV4_6VisionModel(config)
        with safe_open(model_path / "model.safetensors", framework="pt") as weights:
            state = {key.removeprefix("model.vision_tower."): weights.get_tensor(key)
                     for key in weights.keys() if key.startswith("model.vision_tower.")}
        self.vision.load_state_dict(state, strict=True)
        self.vision.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.heads = {item: {key: value.to(device) for key, value in head.items()}
                      for item, head in self.checkpoint["heads"].items()}

    def predict_crop(self, image, item):
        """Accept an RGB PIL head-region crop for mask, person crop for coat."""
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        if item not in self.heads:
            raise ValueError("item must be mask or lab_coat")
        image = ImageOps.pad(image.convert("RGB"), (224, 224), color=(127, 127, 127),
                             method=Image.Resampling.BICUBIC)
        array = np.asarray(image).astype(np.float32) / 127.5 - 1
        pixels = torch.from_numpy(array.transpose(2, 0, 1).copy())[None]
        pixels = pixels.to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            feature = self.vision(
                pixels, target_sizes=torch.tensor([[16, 16]], device=self.device, dtype=torch.int32),
                use_vit_merger=False).last_hidden_state.float().mean(dim=1)
            feature = torch.nn.functional.normalize(feature, dim=-1)
            head = self.heads[item]
            standardized = (feature - head["mean"]) / head["scale"]
            score = torch.sigmoid(standardized @ head["weight"].T + head["bias"]).item()
        state = "worn" if score >= .5 else "not_worn"
        if max(score, 1 - score) < self.checkpoint["threshold"]:
            state = "unknown"
        return {"state": state, "worn_score": score,
                "score_is_calibrated_probability": False, "visibility_checked": False,
                "mode": "experimental", "notification_eligible": False}
