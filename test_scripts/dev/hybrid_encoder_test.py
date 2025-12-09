import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.pipeline.encoder.mixed_encoder import HybridEncoder

import torch

def test_hybrid_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridEncoder(
        in_ch=1,
        base_ch=32,
        depths=[2, 2, 3],
        window_sizes=[8, 8, 16],
        num_heads=8,
        state_dim=16,
        contrast_dim=128,
        enable_local_contrast=True,
        local_patch_size=8
    ).to(device)

    x = torch.randn(2, 1, 256, 256).to(device)  # Batch of 2, 1 channel, 64x64 images

    features, global_repr, local_repr = model(x)

    print("=" * 50)
    print("Hybrid Encoder Test Results:")
    print("=" * 50)

    print(f"Input shape: {x.shape}")
    print(f"\nMulti-scale features:")
    for i, feat in enumerate(features):
        print(f"  Feature {i + 1}: shape = {feat.shape}")
    print(f"\nGlobal representation shape: {global_repr.shape}")
    if local_repr is not None:
        print(f"Local representation shape: {local_repr.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")
    print("=" * 50)


if __name__ == "__main__":
    test_hybrid_encoder()