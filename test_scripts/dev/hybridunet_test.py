import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from model.HybridUNet import HybridUNet


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")

    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    model = HybridUNet(
        in_ch=1,
        num_classes=1,
        base_ch=32,
    ).to(device)
    
    model.print_model_info()
    
    x = torch.randn(2, 1, 256, 256).to(device)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        logits, aux = model(x)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Global repr: {aux['global'].shape}")
    if aux['local'] is not None:
        print(f"  Local repr: {aux['local'].shape}")
    else:
        print(f"  Local repr: None")
    
    print(f"\nEncoder features:")
    for i, f in enumerate(aux['features']):
        print(f"  f{i+1}: {f.shape}")
    
    print(f"\nDecoder intermediate features:")
    print(f"  x3 (bottleneck): {aux['x3'].shape}")
    print(f"  x2 (mid-level): {aux['x2'].shape}")
    print(f"  x1 (high-res): {aux['x1'].shape}")
    

    print("\n" + "=" * 60)
    print("Test 2: Different Input Sizes")
    print("=" * 60)
    
    test_sizes = [(128, 128), (256, 256), (384, 384), (512, 512)]
    for h, w in test_sizes:
        x_test = torch.randn(1, 1, h, w).to(device)
        with torch.no_grad():
            logits_test, _ = model(x_test)
        print(f"Input ({h}x{w}) -> Logits: {logits_test.shape}")
        assert logits_test.shape == (1, 1, h, w), f"Shape mismatch for {h}x{w}!"
    

    print("\n" + "=" * 60)
    print("Test 3: Multi-class Segmentation")
    print("=" * 60)
    
    model_multiclass = HybridUNet(
        in_ch=1,
        num_classes=3,  # 3 类分割
        base_ch=32,
    ).to(device)
    
    x = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        logits, _ = model_multiclass(x)
    
    print(f"Input: {x.shape}")
    print(f"Logits (3 classes): {logits.shape}")
    assert logits.shape == (2, 3, 256, 256), "Multi-class output shape mismatch!"
    

    print("\n" + "=" * 60)
    print("Test 4: Freeze/Unfreeze Parameters")
    print("=" * 60)
    

    model.freeze_encoder()
    encoder_frozen = all(not p.requires_grad for p in model.encoder.parameters())
    print(f"After freeze_encoder: all encoder params frozen = {encoder_frozen}")
    assert encoder_frozen, "Encoder freeze failed!"
    

    model.unfreeze_encoder()
    encoder_unfrozen = all(p.requires_grad for p in model.encoder.parameters())
    print(f"After unfreeze_encoder: all encoder params trainable = {encoder_unfrozen}")
    assert encoder_unfrozen, "Encoder unfreeze failed!"
    

    model.freeze_decoder()
    decoder_frozen = all(not p.requires_grad for p in model.decoder.parameters())
    print(f"After freeze_decoder: all decoder params frozen = {decoder_frozen}")
    assert decoder_frozen, "Decoder freeze failed!"
    

    model.unfreeze_decoder()
    decoder_unfrozen = all(p.requires_grad for p in model.decoder.parameters())
    print(f"After unfreeze_decoder: all decoder params trainable = {decoder_unfrozen}")
    assert decoder_unfrozen, "Decoder unfreeze failed!"
    

    print("\n" + "=" * 60)
    print("Test 5: Custom Configuration")
    print("=" * 60)
    
    model_custom = HybridUNet(
        in_ch=1,
        num_classes=2,
        base_ch=16,
        encoder_kwargs={
            'depths': [1, 1, 2],
            'enable_local_contrast': False, 
        },
        decoder_kwargs={
            'up_mode': 'nearest', 
            'norm_type': 'group', 
        }
    ).to(device)
    
    model_custom.print_model_info()
    
    x = torch.randn(2, 1, 128, 128).to(device)
    with torch.no_grad():
        logits, aux = model_custom(x)
    
    print(f"\nCustom model output:")
    print(f"  Logits: {logits.shape}")
    print(f"  Local repr: {aux['local']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)