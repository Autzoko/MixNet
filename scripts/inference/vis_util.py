import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Tuple, Union


def denormalize_image(image: np.ndarray, method: str = 'auto') -> np.ndarray:

    if method == 'auto':
        img_min = image.min()
        img_max = image.max()
        
        # 自动判断范围
        if img_min >= -1.1 and img_max <= 1.1:
            # 可能是 [-1, 1] 或 [0, 1]
            if img_min < -0.1:
                # [-1, 1] -> [0, 1]
                image = (image + 1.0) / 2.0
            # else: already in [0, 1]
        
        # Normalize to [0, 1]
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
    
    elif method == 'minmax':
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
    
    elif method == 'standard':
        # Assume mean=0, std=1 normalization
        # Reverse: x = x * std + mean
        # For display, clip to [0, 1]
        image = np.clip(image, -3, 3)
        image = (image + 3) / 6.0
    
    # Scale to [0, 255]
    image = (image * 255).astype(np.uint8)
    
    return image


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:

    # Convert grayscale to RGB
    image_rgb = np.stack([image, image, image], axis=-1)  # (H, W, 3)
    
    # Create colored mask
    mask_rgb = np.zeros_like(image_rgb)
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    for i, c in enumerate(color):
        mask_rgb[:, :, i] = mask_binary * c
    
    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * mask_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


def visualize_segmentation_case(
    image_tensor: torch.Tensor,
    gt_mask_tensor: Optional[torch.Tensor],
    pred_mask_tensor: torch.Tensor,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    dice_score: Optional[float] = None,
    show_prob: bool = False,
    pred_prob_tensor: Optional[torch.Tensor] = None,
):

    # Convert to numpy
    image = image_tensor.squeeze(0).cpu().numpy()  # (H, W)
    pred_mask = pred_mask_tensor.squeeze(0).cpu().numpy()  # (H, W)
    
    if gt_mask_tensor is not None:
        gt_mask = gt_mask_tensor.squeeze(0).cpu().numpy()  # (H, W)
    else:
        gt_mask = None
    
    if pred_prob_tensor is not None:
        pred_prob = pred_prob_tensor.squeeze(0).cpu().numpy()  # (H, W)
    else:
        pred_prob = None
    
    # Denormalize image
    image_vis = denormalize_image(image)
    
    # Create figure
    n_plots = 2  # 至少有原图和预测
    if gt_mask is not None:
        n_plots += 1
    if show_prob and pred_prob is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Original image
    axes[plot_idx].imshow(image_vis, cmap='gray')
    axes[plot_idx].set_title('Original Image', fontsize=12)
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Plot 2: GT mask (if available)
    if gt_mask is not None:
        gt_overlay = create_overlay(image_vis, gt_mask, color=(0, 255, 0), alpha=0.5)
        axes[plot_idx].imshow(gt_overlay)
        axes[plot_idx].set_title('Ground Truth', fontsize=12)
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Plot 3: Prediction mask
    pred_overlay = create_overlay(image_vis, pred_mask, color=(255, 0, 0), alpha=0.5)
    axes[plot_idx].imshow(pred_overlay)
    if dice_score is not None:
        axes[plot_idx].set_title(f'Prediction (Dice: {dice_score:.3f})', fontsize=12)
    else:
        axes[plot_idx].set_title('Prediction', fontsize=12)
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Plot 4: Probability map (optional)
    if show_prob and pred_prob is not None:
        im = axes[plot_idx].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[plot_idx].set_title('Probability Map', fontsize=12)
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def visualize_batch(
    images: torch.Tensor,
    gt_masks: Optional[torch.Tensor],
    pred_masks: torch.Tensor,
    save_path: Union[str, Path],
    max_samples: int = 8,
    dice_scores: Optional[torch.Tensor] = None,
):

    B = images.size(0)
    n_samples = min(B, max_samples)
    
    # Create grid
    n_cols = 3 if gt_masks is not None else 2
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Get data
        image = images[i].squeeze(0).cpu().numpy()  # (H, W)
        pred = pred_masks[i].squeeze(0).cpu().numpy()  # (H, W)
        
        if gt_masks is not None:
            gt = gt_masks[i].squeeze(0).cpu().numpy()  # (H, W)
        else:
            gt = None
        
        if dice_scores is not None:
            dice = dice_scores[i].item()
        else:
            dice = None
        
        # Denormalize
        image_vis = denormalize_image(image)
        
        col = 0
        
        # Original
        axes[i, col].imshow(image_vis, cmap='gray')
        axes[i, col].set_title(f'Sample {i}', fontsize=10)
        axes[i, col].axis('off')
        col += 1
        
        # GT
        if gt is not None:
            gt_overlay = create_overlay(image_vis, gt, color=(0, 255, 0), alpha=0.5)
            axes[i, col].imshow(gt_overlay)
            axes[i, col].set_title('GT', fontsize=10)
            axes[i, col].axis('off')
            col += 1
        
        # Prediction
        pred_overlay = create_overlay(image_vis, pred, color=(255, 0, 0), alpha=0.5)
        axes[i, col].imshow(pred_overlay)
        if dice is not None:
            axes[i, col].set_title(f'Pred (Dice: {dice:.3f})', fontsize=10)
        else:
            axes[i, col].set_title('Pred', fontsize=10)
        axes[i, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def visualize_comparison(
    image_tensor: torch.Tensor,
    masks_dict: dict,
    save_path: Union[str, Path],
    title: Optional[str] = None,
):
    image = image_tensor.squeeze(0).cpu().numpy()
    image_vis = denormalize_image(image)
    
    n_models = len(masks_dict)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
    
    # Original image
    axes[0].imshow(image_vis, cmap='gray')
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    # Each model
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for idx, (model_name, mask_tensor) in enumerate(masks_dict.items(), 1):
        mask = mask_tensor.squeeze(0).cpu().numpy()
        color = colors[idx % len(colors)]
        
        overlay = create_overlay(image_vis, mask, color=color, alpha=0.5)
        axes[idx].imshow(overlay)
        axes[idx].set_title(model_name, fontsize=12)
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

