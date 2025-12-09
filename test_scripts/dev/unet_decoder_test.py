import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from model.pipeline.utils.unet import ConvBlock2d, UpBlock2d
from model.pipeline.decoder.unet_decoder import UNetDecoder2D


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # ConvBlock2d
    print("\n" + "="*60)
    print("Testing ConvBlock2d")
    print("="*60)
    conv_block = ConvBlock2d(64, 128, norm_type='batch').to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = conv_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 128, 32, 32), "ConvBlock2d output shape mismatch!"
    print("✅ ConvBlock2d test passed!")
    
    # UpBlock2d
    print("\n" + "="*60)
    print("Testing UpBlock2d")
    print("="*60)
    up_block = UpBlock2d(128, 64, 64, mode='bilinear').to(device)
    low_res = torch.randn(2, 128, 16, 16).to(device)
    skip = torch.randn(2, 64, 32, 32).to(device)
    out = up_block(low_res, skip)
    print(f"Low-res input: {low_res.shape}")
    print(f"Skip input: {skip.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 64, 32, 32), "UpBlock2d output shape mismatch!"
    print("✅ UpBlock2d test passed!")
    
    # UNetDecoder2d
    print("\n" + "="*60)
    print("Testing UNetDecoder2d")
    print("="*60)
    
    decoder = UNetDecoder2D(
        enc_channels=(32, 64, 128),
        dec_channels=(128, 64, 32),
        num_classes=1,
        up_mode='bilinear',
        align_corners=False,
        norm_type='batch',
        use_bottleneck=True
    ).to(device)
    
    # simulate encoder outputs features
    f1 = torch.randn(2, 32, 256, 256).to(device)
    f2 = torch.randn(2, 64, 128, 128).to(device)
    f3 = torch.randn(2, 128, 64, 64).to(device)
    features = [f1, f2, f3]
    
    print(f"Encoder features:")
    print(f"  f1: {f1.shape}")
    print(f"  f2: {f2.shape}")
    print(f"  f3: {f3.shape}")
    
    # test normal output
    logits = decoder(features)
    print(f"\nDecoder output (logits): {logits.shape}")
    assert logits.shape == (2, 1, 256, 256), "Decoder output shape mismatch!"
    
    # test intermediate outputs
    logits, intermediates = decoder(features, return_intermediate=True)
    print(f"\nIntermediate features:")
    for k, v in intermediates.items():
        print(f"  {k}: {v.shape}")
    
    # model parameter statistics
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nDecoder statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # test different input sizes
    print("\n" + "="*60)
    print("Testing different input sizes")
    print("="*60)
    test_sizes = [(128, 128), (256, 256), (384, 384), (512, 512)]
    for h, w in test_sizes:
        f1_test = torch.randn(1, 32, h, w).to(device)
        f2_test = torch.randn(1, 64, h//2, w//2).to(device)
        f3_test = torch.randn(1, 128, h//4, w//4).to(device)
        features_test = [f1_test, f2_test, f3_test]
        
        with torch.no_grad():
            logits_test = decoder(features_test)
        
        print(f"Input size ({h}x{w}): logits shape = {logits_test.shape}")
        assert logits_test.shape == (1, 1, h, w), f"Output shape mismatch for size {h}x{w}!"
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)