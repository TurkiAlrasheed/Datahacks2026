"""
convert_to_tflite.py
====================
Run on your dev machine. Converts .pth → TFLite directly via ai-edge-torch.

Setup:
    # ai-edge-torch requires Python 3.10 or 3.11
    conda create -n aiedge python=3.11
    conda activate aiedge
    pip install torch torchvision ai-edge-torch

Usage:
    python convert_to_tflite.py \
        --weights your_model.pth \
        --num-classes 10 \
        --output species_classifier.tflite
"""

import argparse
import os
import torch
import torch.nn as nn


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        self.backbone.eval()
        self.head = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--output", default="species_classifier.tflite")
    args = parser.parse_args()

    print(f"Building DINOv2-ViT-Base + Linear(768, {args.num_classes}) ...")
    model = DINOv2Classifier(num_classes=args.num_classes)

    print(f"Loading weights from {args.weights} ...")
    state_dict = torch.load(args.weights, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("  ✓ Weights loaded")

    import litert_torch as ai_edge_torch

    sample_input = (torch.randn(1, 3, args.input_size, args.input_size),)
    print("Converting to TFLite (this may take a few minutes) ...")
    edge_model = ai_edge_torch.convert(model, sample_input)
    edge_model.export(args.output)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  ✓ Saved → {args.output} ({size_mb:.1f} MB)")

    # Sanity check
    import numpy as np
    print("Verifying ...")
    torch_out = model(sample_input[0])
    edge_out = edge_model(*sample_input)
    max_diff = np.abs(torch_out.detach().numpy() - edge_out).max()
    print(f"  ✓ Max diff between PyTorch and TFLite: {max_diff:.6f}")
    print(f"  ✓ Output shape: {edge_out.shape}")


if __name__ == "__main__":
    main()