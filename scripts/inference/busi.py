import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Import existing dataset creators
from dataset.dataset_creators import create_segmentation_dataloaders
from dataset.meta.UltrasoundSample import load_ultrasound_metadata

from scripts.inference.vis_util import visualize_segmentation_case


# ============================================================
# Helper Functions
# ============================================================

def compute_dice_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算每个样本的 Dice 系数。
    
    Args:
        pred: (B, 1, H, W) - 预测 mask
        target: (B, 1, H, W) - GT mask
        epsilon: 数值稳定性常数
    
    Returns:
        dice_scores: (B,) - 每个样本的 Dice
    """
    B = pred.size(0)
    
    # Flatten: (B, H*W)
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)
    
    # Per-sample Dice
    intersection = (pred_flat * target_flat).sum(dim=1)  # (B,)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # (B,)
    
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    
    return dice


# ============================================================
# Main Inference Function
# ============================================================

def main():
    # ==================== Argument Parsing ====================
    parser = argparse.ArgumentParser(
        description="BUSI Inference - Simplified"
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='BUSI_processed',
        help='BUSI preprocessed data root directory'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Which split to run inference on'
    )
    
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Input image size'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='outputs_busi_infer',
        help='Output directory'
    )
    
    parser.add_argument(
        '--num_visualize',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader num_workers'
    )
    
    args = parser.parse_args()
    
    # ==================== Setup ====================
    print("=" * 70)
    print("BUSI Inference - Simplified")
    print("=" * 70)
    print()
    
    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    print(f"Split: {args.split}")
    print()
    
    # ==================== Load Data ====================
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    # Construct meta paths
    train_meta_path = os.path.join(args.data_root, 'train_meta.json')
    val_meta_path = os.path.join(args.data_root, 'val_meta.json')
    
    # Use existing dataloader creator
    # Note: We'll use val_loader regardless of split for simplicity
    # You can modify this to support test set if needed
    _, dataloader = create_segmentation_dataloaders(
        train_meta_path=train_meta_path,
        val_meta_path=val_meta_path,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=False,
    )
    
    print(f"Total batches: {len(dataloader)}")
    print()
    
    # ==================== Load Model ====================
    print("=" * 70)
    print("Loading Model")
    print("=" * 70)
    
    try:
        from model.HybridUNet import HybridUNet
    except ImportError:
        raise ImportError("Failed to import HybridUNet")
    
    # Create model
    model = HybridUNet(
        in_ch=1,
        num_classes=1,
        base_ch=32,
        encoder_kwargs=None,
        decoder_kwargs=None,
    )
    
    # Load checkpoint
    print(f"Loading: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'best_val_dice' in checkpoint:
        print(f"Best val dice: {checkpoint['best_val_dice']:.4f}")
    print()
    
    # ==================== Inference ====================
    print("=" * 70)
    print("Running Inference")
    print("=" * 70)
    print()
    
    all_dice_scores = []
    visualize_count = 0
    sample_idx = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(device)  # (B, 1, H, W)
            masks = batch['mask'].to(device)    # (B, 1, H, W)
            
            # Forward
            outputs = model(images)
            
            # Extract logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Predictions
            prob_map = torch.sigmoid(logits)
            pred_mask = (prob_map > 0.5).float()
            
            # Compute Dice
            dice_scores = compute_dice_per_sample(pred_mask, masks)
            all_dice_scores.extend(dice_scores.cpu().numpy().tolist())
            
            # Visualize
            batch_size = images.size(0)
            for i in range(batch_size):
                img = images[i]
                pred = pred_mask[i]
                gt = masks[i]
                dice = dice_scores[i].item()
                
                # Save visualization
                if visualize_count < args.num_visualize:
                    save_name = f"{args.split}_case_{sample_idx:04d}_dice_{dice:.3f}.png"
                    save_path = save_dir / save_name
                    
                    visualize_segmentation_case(
                        image_tensor=img,
                        gt_mask_tensor=gt,
                        pred_mask_tensor=pred,
                        save_path=str(save_path),
                        title=f"Case {sample_idx}",
                        dice_score=dice,
                        show_prob=False,
                    )
                    
                    visualize_count += 1
                
                # Save predictions
                pred_save_dir = save_dir / "predictions"
                pred_save_dir.mkdir(exist_ok=True)
                
                pred_np = pred.squeeze(0).cpu().numpy()
                prob_np = prob_map[i].squeeze(0).cpu().numpy()
                
                np.save(pred_save_dir / f"pred_mask_{sample_idx:04d}.npy", pred_np)
                np.save(pred_save_dir / f"pred_prob_{sample_idx:04d}.npy", prob_np)
                
                sample_idx += 1
            
            # Update progress
            pbar.set_postfix({'dice': f'{np.mean(all_dice_scores):.4f}'})
    
    # ==================== Results ====================
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    mean_dice = np.mean(all_dice_scores)
    std_dice = np.std(all_dice_scores)
    median_dice = np.median(all_dice_scores)
    min_dice = np.min(all_dice_scores)
    max_dice = np.max(all_dice_scores)
    
    print(f"Total samples: {sample_idx}")
    print(f"Visualizations: {visualize_count}")
    print()
    print(f"Dice Statistics:")
    print(f"  Mean:   {mean_dice:.4f}")
    print(f"  Std:    {std_dice:.4f}")
    print(f"  Median: {median_dice:.4f}")
    print(f"  Min:    {min_dice:.4f}")
    print(f"  Max:    {max_dice:.4f}")
    
    # Save metrics
    metrics_path = save_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Samples: {sample_idx}\n")
        f.write(f"\n")
        f.write(f"Dice:\n")
        f.write(f"  Mean:   {mean_dice:.4f}\n")
        f.write(f"  Std:    {std_dice:.4f}\n")
        f.write(f"  Median: {median_dice:.4f}\n")
        f.write(f"  Min:    {min_dice:.4f}\n")
        f.write(f"  Max:    {max_dice:.4f}\n")
    
    print(f"\nSaved to: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()